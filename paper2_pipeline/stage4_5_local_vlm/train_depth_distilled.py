"""
Paper 2 — Distilled depth classifier (Stage 4 alternative).

Student: timm CNN fine-tuned on pothole crops.
Teacher labels: Gemini 2.5 Pro depth categories for all 3,092 ENPAD potholes.
Split: BY SCENE (no leakage), with the 12 human-eval scenes fully held out.
Judge: kappa vs human consensus on the frozen 38-set (same bar as VLMs: 0.4795).

Run (paper2 env, GPU 1 while GPU 0 serves VLMs):
    CUDA_VISIBLE_DEVICES=1 python train_depth_distilled.py --arch convnext_tiny

Output: results/distill_depth_<arch>.json + checkpoint in stage4_5_local_vlm/ckpt/
"""
import argparse
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
GEMINI = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/gemini-2_5-pro_full_dataset.json"
PATCHES = ROOT / "Main_Work/image_patches"
EVAL_SET = P2 / "results/human_depth_eval_set.json"
CKPT_DIR = P2 / "stage4_5_local_vlm/ckpt"

CLASSES = ["low", "medium", "high"]
SEED = 42
SCALE_AUG = False


class CropDataset(Dataset):
    """Yields (image, geo_features[6], label). Geometry = the Paper-1 vector
    [area(log10), circularity, solidity, aspect_ratio, dist_idx, lat_in_path]."""

    def __init__(self, items, train):
        import torchvision.transforms as T

        self.items = items
        # --scale-aug adds aggressive scale/crop jitter. Motivation: the student
        # collapses on close-range imagery because every training crop is a small,
        # distant dashcam defect. Randomised zoom simulates other capture
        # geometries at training time.
        if train and SCALE_AUG:
            aug = [
                T.RandomResizedCrop(224, scale=(0.25, 1.0), ratio=(0.7, 1.4)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.15),
            ]
        elif train:
            aug = [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.1),
            ]
        else:
            aug = [T.Resize((224, 224))]
        self.tf = T.Compose(
            aug + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, geo, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(geo, dtype=torch.float32), label


DIST_IDX = {"Near": 0.0, "Medium Distance": 0.5, "Far": 1.0}


def geo_vector(metrics, positional):
    area = max(float(metrics.get("area", 10)), 10.0)
    lat = positional.get("Lateral_Position", "")
    return [
        (np.log10(area) - 2.0) / 2.5,                     # same normalization as risk
        float(metrics.get("circularity", 0)),
        float(metrics.get("solidity", 0)),
        min(float(metrics.get("aspect_ratio", 0)) / 5.0, 1.0),
        DIST_IDX.get(positional.get("Distance_Category"), 0.5),
        1.0 if lat.startswith("in_path") else 0.0,
    ]


class TwoTower(nn.Module):
    """CNN image tower + optional MLP geometry tower -> classifier head."""

    def __init__(self, arch, use_geo=True):
        super().__init__()
        import timm

        self.use_geo = use_geo
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        if use_geo:
            self.geo = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            feat_dim += 64
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 3)
        )

    def forward(self, img, geo):
        z = self.backbone(img)
        if self.use_geo:
            z = torch.cat([z, self.geo(geo)], dim=1)
        return self.head(z)


def load_items():
    """(scene, crop_path, geo6, class_idx) for every Gemini-labeled pothole."""
    gem = json.loads(GEMINI.read_text())
    items, missing = [], 0
    for pid, rec in gem.items():
        scene = rec["scene"]
        crop = PATCHES / scene / f"{scene}_mask{rec['bbox_id']}.jpg"
        if not crop.exists():
            missing += 1
            continue
        geo = geo_vector(rec.get("geometric_features", {}),
                         rec.get("positional_features", {}))
        items.append((scene, crop, geo, CLASSES.index(rec["depth_category"])))
    print(f"labeled crops: {len(items)}  (missing files: {missing})")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--variant", choices=["image", "twotower"], default="twotower")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale-aug", action="store_true",
                    help="aggressive scale jitter for cross-geometry robustness")
    args = ap.parse_args()

    global SCALE_AUG
    SCALE_AUG = args.scale_aug
    # Full determinism. Without this, two runs with the same seed produced models
    # differing by kappa 0.13 vs 0.33 on the 38-item human set: cuDNN autotuning and
    # unseeded DataLoader workers make training irreproducible, and on a test set
    # this small the resulting spread swamps any effect being measured.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda:0"  # remapped by CUDA_VISIBLE_DEVICES

    eval_set = json.loads(EVAL_SET.read_text())
    heldout_scenes = {pid.rsplit("_bbox", 1)[0] for pid in eval_set["records"]}
    print(f"held-out human-eval scenes: {len(heldout_scenes)}")

    all_items = load_items()
    trainable = [it for it in all_items if it[0] not in heldout_scenes]

    # scene-level split, stratified-ish by shuffling scenes
    scenes = sorted({it[0] for it in trainable})
    random.shuffle(scenes)
    n_val = max(1, int(0.15 * len(scenes)))
    val_scenes = set(scenes[:n_val])
    train_items = [(p, g, y) for s, p, g, y in trainable if s not in val_scenes]
    val_items = [(p, g, y) for s, p, g, y in trainable if s in val_scenes]
    print(f"train: {len(train_items)} potholes / {len(scenes)-n_val} scenes | "
          f"val: {len(val_items)} potholes / {n_val} scenes")

    model = TwoTower(args.arch, use_geo=(args.variant == "twotower")).to(device)

    # class weights for imbalance (medium is ~half the data)
    counts = np.bincount([y for _, _, y in train_items], minlength=3)
    weights = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    g = torch.Generator(); g.manual_seed(args.seed)
    def _wi(wid):
        ws = args.seed + wid
        np.random.seed(ws); random.seed(ws)
    dl_train = DataLoader(CropDataset(train_items, True), batch_size=args.batch,
                          shuffle=True, num_workers=8, pin_memory=True,
                          generator=g, worker_init_fn=_wi)
    dl_val = DataLoader(CropDataset(val_items, False), batch_size=args.batch,
                        shuffle=False, num_workers=8, pin_memory=True)

    from sklearn.metrics import cohen_kappa_score

    def evaluate(dl):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, g, y in dl:
                logits = model(x.to(device), g.to(device))
                ps.extend(logits.argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        acc = float(np.mean(np.array(ys) == np.array(ps)))
        kap = float(cohen_kappa_score(ys, ps))
        return acc, kap

    best_val_kappa, best_state = -1, None
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0
        for x, g, y in dl_train:
            opt.zero_grad()
            loss = crit(model(x.to(device), g.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot_loss += float(loss)
        sched.step()
        acc, kap = evaluate(dl_val)
        print(f"ep {ep+1}/{args.epochs} loss {tot_loss/len(dl_train):.3f} "
              f"val acc {acc:.3f} val kappa(vs gemini) {kap:.3f}")
        if kap > best_val_kappa:
            best_val_kappa = kap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    train_s = time.perf_counter() - t0

    suffix = '_scaleaug' if args.scale_aug else ''
    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(best_state, CKPT_DIR / f"distill_{args.arch}_{args.variant}_s{args.seed}{suffix}.pt")

    # ---- final exam: frozen human 38-consensus set ----
    import torchvision.transforms as T

    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model.eval()
    gem = json.loads(GEMINI.read_text())
    preds, y_true, per_item, infer_times = [], [], {}, []
    for pid, rec in eval_set["records"].items():
        crop_path = ROOT / "Main_Work/test_data/image_patches" / rec["crop_image"].split("image_patches/")[1]
        img = tf(Image.open(crop_path).convert("RGB"))[None].to(device)
        grec = gem[pid]
        geo = torch.tensor(
            [geo_vector(grec.get("geometric_features", {}),
                        grec.get("positional_features", {}))],
            dtype=torch.float32,
        ).to(device)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad():
            p = int(model(img, geo).argmax(1))
        torch.cuda.synchronize(); infer_times.append(time.perf_counter() - t1)
        per_item[pid] = CLASSES[p]
        if rec["human_consensus"]:
            preds.append(CLASSES[p])
            y_true.append(rec["human_consensus"])

    kappa_human = float(cohen_kappa_score(y_true, preds, labels=CLASSES))
    agree = float(np.mean([t == p for t, p in zip(y_true, preds)]))

    result = {
        "config": vars(args) | {"seed": SEED, "teacher": "gemini-2.5-pro labels (3092)",
                                "split": "by scene; 12 human-eval scenes fully held out"},
        "train_seconds": round(train_s, 1),
        "best_val_kappa_vs_gemini": round(best_val_kappa, 4),
        "human_eval": {
            "kappa_vs_human_consensus": round(kappa_human, 4),
            "agreement_percent": round(100 * agree, 2),
            "n": len(y_true),
            "bar_gemini_pro_rescored": 0.4795,
            "clears_bar": kappa_human >= 0.4795,
            "s_per_pothole": round(float(np.mean(infer_times)), 5),
        },
        "predictions_38set": per_item,
    }
    out = P2 / "results" / f"distill_depth_{args.arch}_{args.variant}_s{args.seed}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "predictions_38set"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
