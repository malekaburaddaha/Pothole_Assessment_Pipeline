"""
Paper 2 — Unified pipeline, Phase A (fully local, no API):

  Stage 1: perspective-transformed image + GT boxes (as Paper 1)
  Stage 2: SAM ViT-H, fp16 autocast encoder + torch.compile   (decided Step 4)
  Stage 3: geometric features — Paper-1 convention preserved for model/risk
           inputs, PLUS corrected solidity emitted for the label corpus
  Stage 4a: depth = 6-model distilled ConvNeXt ensemble        (decided Step 7)
  Stage 4b: composite risk, production formula (fusion_and_risk_scoring.py)

Output: one VLA-ready assessment record per scene (warnings merged in Phase B).

  python paper2_pipeline.py --limit 20          # smoke test
  python paper2_pipeline.py                     # full 1,119 scenes

Output: results/paper2_full_assessments.json (+ timing block)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Paths resolve through repo_config so a fresh clone runs on the bundled sample
# with no configuration; set POTHOLE_DATA_ROOT to point at the full corpus.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import repo_config as _cfg

IMAGES_DIR = _cfg.images_dir()
BBOXES_DIR = _cfg.bboxes_dir()
SAM_CKPT = _cfg.SAM_CKPT
CNN_CKPT_DIR = _cfg.CKPT_DIR / "distilled_severity"
GEMINI = _cfg.labels_path()
ROOT = _cfg.ROOT
P2 = _cfg.REPO

DEVICE = "cuda:0"
CLASSES = ["low", "medium", "high"]
ENSEMBLE = [("image", 42), ("image", 43), ("image", 44),
            ("twotower", 42), ("twotower", 43), ("twotower", 44)]

# ---------------------------------------------------------------- Stage 2
from segment_anything import sam_model_registry, SamPredictor


class AutocastEncoderPredictor(SamPredictor):
    @torch.no_grad()
    def set_torch_image(self, transformed_image, original_image_size):
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = self.model.image_encoder(input_image)
        self.features = feats.float()
        self.is_image_set = True


# ---------------------------------------------------------------- Stage 3
def stage3_features(mask):
    """Paper-1 convention + corrected solidity (both emitted)."""
    area = int(np.sum(mask))  # full-mask pixels — risk formula calibrated on this
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
        solidity_p1 = area / hull_area if hull_area > 0 else 0.0        # Paper-1 (buggy: mixed sources)
        solidity_fix = contour_area / hull_area if hull_area > 0 else 0.0  # consistent contour/hull
    else:
        perimeter = circularity = aspect_ratio = solidity_p1 = solidity_fix = 0.0
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity_p1),            # kept: model input + comparability
        "solidity_corrected": float(solidity_fix),  # emitted for VLA labels
    }


def positional(bbox):
    """Paper-1 thresholds (Code_Files/main.py)."""
    y2 = bbox[3]
    dist = "Far" if y2 <= 449 else "Medium Distance" if y2 <= 514 else "Near"
    # Reference coordinate: the box LEFT EDGE (x_min), not the centre.
    # Paper 1's released main.py uses the centre, but that does not reproduce the
    # released labels; x_min reproduces them exactly (verified: non-overlapping
    # zone ranges over all 3,092 potholes, whereas centre-x ranges overlap).
    # We match the published data for comparability and document the discrepancy.
    x = bbox[0]
    if 1000 <= x <= 1839:
        lat = "in_path_left"
    elif 1840 <= x <= 2700:
        lat = "in_path_right"
    elif x < 1000:
        lat = "in_lane_far_left"
    else:
        lat = "in_lane_far_right"
    return {"Distance_Category": dist, "Lateral_Position": lat}


# ---------------------------------------------------------------- Stage 4a
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage4_5_local_vlm"))
from train_depth_distilled import TwoTower, geo_vector  # noqa: E402

import torchvision.transforms as T  # noqa: E402

CNN_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_ensemble():
    models = []
    for variant, seed in ENSEMBLE:
        m = TwoTower("convnext_tiny", use_geo=(variant == "twotower"))
        sd = torch.load(CNN_CKPT_DIR / f"distill_convnext_tiny_{variant}_s{seed}.pt",
                        map_location="cpu")
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        models.append(m)
    return models


@torch.no_grad()
def ensemble_depth(models, crops_rgb, geos):
    """Majority vote over the 6 models for a batch of potholes."""
    imgs = torch.stack([CNN_TF(Image.fromarray(c)) for c in crops_rgb]).to(DEVICE)
    g = torch.tensor(geos, dtype=torch.float32).to(DEVICE)
    votes = np.zeros((len(crops_rgb), 3), dtype=int)
    for m in models:
        pred = m(imgs, g).argmax(1).cpu().numpy()
        for i, p in enumerate(pred):
            votes[i, p] += 1
    return [CLASSES[int(v.argmax())] for v in votes], votes.tolist()


# ---------------------------------------------------------------- Stage 4b
def risk(depth_cat, area_pixels, lateral_pos):
    """Production formula (VLM_Code/fusion_and_risk_scoring.py), verbatim."""
    s_depth = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(depth_cat, 0.2) * 100
    a = max(float(area_pixels), 10.0)
    s_area = float(np.clip((np.log10(a) - 2.0) / 2.5 * 100, 0, 100))
    if lateral_pos in ("in_path_left", "in_path_right"):
        s_pos = 100
    elif lateral_pos in ("in_lane_far_left", "in_lane_far_right"):
        s_pos = 70
    else:
        s_pos = 30
    score = round(0.5 * s_depth + 0.3 * s_area + 0.2 * s_pos, 1)
    sev = ("Critical" if score >= 80 else "High" if score >= 60
           else "Medium" if score >= 40 else "Low")
    return score, sev, {"s_depth": s_depth, "s_area": s_area, "s_pos": s_pos}


def load_bboxes(scene):
    boxes = []
    for line in (BBOXES_DIR / f"{scene}.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        x, y, w, h = map(float, line.split())
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="paper2_full_assessments.json")
    args = ap.parse_args()

    gemini = json.loads(GEMINI.read_text())  # merged in as reference labels

    print("Loading SAM ViT-H (fp16+compile) ...")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(DEVICE).eval()
    sam.image_encoder = torch.compile(sam.image_encoder)
    predictor = AutocastEncoderPredictor(sam)
    print("Loading distilled depth ensemble (6 models) ...")
    models = load_ensemble()

    scenes = sorted(p.stem for p in IMAGES_DIR.glob("*.JPG"))
    if args.limit:
        scenes = scenes[: args.limit]

    out, t_stage = {}, {"load": 0.0, "sam": 0.0, "feat": 0.0, "depth": 0.0, "risk": 0.0}
    t_all = time.perf_counter()
    for si, scene in enumerate(scenes):
        boxes = load_bboxes(scene)
        if not boxes:
            continue

        t0 = time.perf_counter()
        img = cv2.imread(str(IMAGES_DIR / f"{scene}.JPG"))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_stage["load"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        predictor.set_image(rgb)
        masks = []
        for box in boxes:
            m, s, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=True)
            masks.append((m[int(np.argmax(s))], float(np.max(s))))
        torch.cuda.synchronize()
        t_stage["sam"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        feats = [stage3_features(m) for m, _ in masks]
        poss = [positional(b) for b in boxes]
        t_stage["feat"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        crops = [rgb[b[1]:b[3], b[0]:b[2]] for b in boxes]
        geos = [geo_vector(f, p) for f, p in zip(feats, poss)]
        depths, votes = ensemble_depth(models, crops, geos)
        torch.cuda.synchronize()
        t_stage["depth"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        potholes = []
        for i, (box, (mask, sam_score), f, p, d, v) in enumerate(
            zip(boxes, masks, feats, poss, depths, votes)
        ):
            score, sev, comp = risk(d, f["area"], p["Lateral_Position"])
            pid = f"{scene}_bbox{i}"
            g1 = gemini.get(pid, {})
            potholes.append({
                "pothole_id": pid,
                "bbox_xyxy": box,
                "sam_score": round(sam_score, 4),
                "geometric": f,
                "positional": p,
                "depth": {
                    "paper2_distilled": d,
                    "ensemble_votes_low_med_high": v,
                    "paper1_gemini": g1.get("depth_category"),
                    "paper1_gemini_confidence": g1.get("depth_confidence"),
                },
                "risk_assessment": {
                    "composite_score": score,
                    "severity_level": sev,
                    "component_scores": comp,
                },
            })
        max_risk = max(p["risk_assessment"]["composite_score"] for p in potholes)
        out[scene] = {
            "scene_id": scene,
            "image": str(IMAGES_DIR / f"{scene}.JPG"),
            "num_potholes": len(potholes),
            "max_risk_score": max_risk,
            "highest_severity": ("Critical" if max_risk >= 80 else "High" if max_risk >= 60
                                 else "Medium" if max_risk >= 40 else "Low"),
            "potholes": potholes,
        }
        t_stage["risk"] += time.perf_counter() - t0

        if (si + 1) % 100 == 0:
            print(f"[{si+1}/{len(scenes)}] elapsed {time.perf_counter()-t_all:.0f}s")

    total_s = time.perf_counter() - t_all
    n = len(out)
    timing = {k: round(v / n, 4) for k, v in t_stage.items()} | {
        "total_s_per_scene": round(total_s / n, 4),
        "n_scenes": n,
        "wall_total_s": round(total_s, 1),
    }
    result = {"config": {"stage2": "vith fp16+compile", "stage4": "distilled 6-ensemble",
                         "risk": "paper1 production formula"},
              "timing_per_scene": timing, "scenes": out}
    out_path = P2 / "results" / args.out
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(timing, indent=1))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
