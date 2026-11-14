"""
Paper 2 — Stage 2 candidate benchmark: efficient SAM variants vs Paper-1 ViT-H.

One model per process (namespace + GPU isolation):
    python bench_stage2_candidates.py --model mobilesam
    python bench_stage2_candidates.py --model tinysam
    python bench_stage2_candidates.py --model efficientvit-l2
    python bench_stage2_candidates.py --model efficientvit-xl1
    python bench_stage2_candidates.py --model sam2.1-small

Identical protocol to bench_stage2_baseline.py (same 50 scenes after 3 warmup,
CUDA-synced timers, GT-box prompts, multimask -> best score).

Measured per candidate:
  speed    : t_load / t_encoder / t_decoder / t_total
  fidelity : IoU of best-mask vs stored Paper-1 ViT-H mask PNG
  drift    : relative error of Stage-3 features (area, circularity, solidity, AR)
  impact   : composite-risk recomputed with candidate area (depth & position held
             at Paper-1 values) -> risk category flip rate vs Paper-1 categories

Output: Paper2_Improvements/results/bench_stage2_<model>.json
"""
import argparse
import json
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
IMAGES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images"
BBOXES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_bboxes"
PAPER1_MASKS_DIR = ROOT / "Main_Work/Dataset/With_Distance/sam_predictions_large"
RISK_FLAT = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/final_risk_scores_flat.json"
CKPT_DIR = P2 / "stage2_segmentation/checkpoints"
TP = P2 / "stage2_segmentation/third_party"

N_SCENES = 50
N_WARMUP = 3
DEVICE = "cuda:0"


# --------------------------------------------------------------------------
# Adapters: identical interface per model so the measurement loop is shared.
# --------------------------------------------------------------------------
class Adapter:
    name = "base"

    def load(self):
        raise NotImplementedError

    def set_image(self, rgb):
        self.predictor.set_image(rgb)

    def predict_box(self, box_xyxy):
        """Returns (masks[N,H,W] bool, scores[N]) with multimask output."""
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box_xyxy)[None, :],
            multimask_output=True,
        )
        return masks, scores


class MobileSAMAdapter(Adapter):
    name = "mobilesam"

    def load(self):
        sys.path.insert(0, str(TP / "MobileSAM"))
        from mobile_sam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_DIR / "mobile_sam.pt"))
        sam.to(DEVICE).eval()
        self.predictor = SamPredictor(sam)


class TinySAMAdapter(Adapter):
    name = "tinysam"

    def load(self):
        sys.path.insert(0, str(TP / "TinySAM"))
        from tinysam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_DIR / "tinysam.pth"))
        sam.to(DEVICE).eval()
        self.predictor = SamPredictor(sam)

    def predict_box(self, box_xyxy):
        # TinySAM's predict() has no multimask_output flag; it always returns
        # multiple candidates (masks, iou_predictions, low_res_logits).
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box_xyxy)[None, :],
        )
        return masks, scores


class EfficientViTAdapter(Adapter):
    def __init__(self, variant):  # "l2" or "xl1"
        self.variant = variant
        self.name = f"efficientvit-{variant}"

    def load(self):
        sys.path.insert(0, str(TP / "efficientvit"))
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

        model = create_efficientvit_sam_model(
            name=f"efficientvit-sam-{self.variant}",
            weight_url=str(CKPT_DIR / f"efficientvit_sam_{self.variant}.pt"),
        )
        model.to(DEVICE).eval()
        self.predictor = EfficientViTSamPredictor(model)


class SAM2Adapter(Adapter):
    name = "sam2.1-small"

    def load(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(CKPT_DIR / "sam2.1_hiera_small.pt"),
            device=DEVICE,
        )
        self.predictor = SAM2ImagePredictor(model)

    def predict_box(self, box_xyxy):
        masks, scores, _ = self.predictor.predict(
            box=np.array(box_xyxy)[None, :], multimask_output=True
        )
        return masks.astype(bool), scores


ADAPTERS = {
    "mobilesam": MobileSAMAdapter,
    "tinysam": TinySAMAdapter,
    "efficientvit-l2": lambda: EfficientViTAdapter("l2"),
    "efficientvit-xl1": lambda: EfficientViTAdapter("xl1"),
    "sam2.1-small": SAM2Adapter,
}


# --------------------------------------------------------------------------
# Stage 3 + risk (verbatim Paper-1 logic)
# --------------------------------------------------------------------------
def calculate_mask_metrics(mask):
    area = np.sum(mask)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
    else:
        perimeter = circularity = aspect_ratio = solidity = 0
    return {
        "area": int(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
    }


def area_score(area_pixels):
    if area_pixels < 10:
        area_pixels = 10
    return float(np.clip((np.log10(area_pixels) - 2.0) / 2.5 * 100, 0, 100))


def composite(s_depth, s_area, s_pos):
    return round(0.5 * s_depth + 0.3 * s_area + 0.2 * s_pos, 1)


def categorize(score):
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def load_bboxes(scene):
    boxes = []
    for line in (BBOXES_DIR / f"{scene}.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        x, y, w, h = map(float, line.split())
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def rel_err(new, ref):
    if ref == 0:
        return None
    return float(abs(new - ref) / abs(ref))


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS.keys()))
    args = ap.parse_args()

    risk_flat = json.loads(RISK_FLAT.read_text())

    adapter = ADAPTERS[args.model]()
    print(f"Loading {adapter.name} ...")
    t0 = time.perf_counter()
    adapter.load()
    sync()
    model_load_s = time.perf_counter() - t0
    print(f"Model load: {model_load_s:.1f}s")

    scenes = sorted(p.stem for p in IMAGES_DIR.glob("*.JPG"))[: N_SCENES + N_WARMUP]

    per_scene, ious, drifts, flips = [], [], [], []
    n_boxes_total = 0

    for idx, scene in enumerate(scenes):
        warmup = idx < N_WARMUP
        boxes = load_bboxes(scene)
        if not boxes:
            continue

        t0 = time.perf_counter()
        image = cv2.imread(str(IMAGES_DIR / f"{scene}.JPG"))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_load = time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        adapter.set_image(rgb)
        sync()
        t_encoder = time.perf_counter() - t0

        t_decoder = 0.0
        for i, box in enumerate(boxes):
            sync()
            t0 = time.perf_counter()
            masks, scores = adapter.predict_box(box)
            sync()
            t_decoder += time.perf_counter() - t0
            if warmup:
                continue

            best = masks[int(np.argmax(scores))]
            feats = calculate_mask_metrics(best)

            # fidelity vs stored Paper-1 ViT-H mask
            stored_path = PAPER1_MASKS_DIR / scene / f"mask_{i}.png"
            if stored_path.exists():
                ref_mask = cv2.imread(str(stored_path), cv2.IMREAD_GRAYSCALE) > 127
                iou = mask_iou(best, ref_mask)
                ious.append(iou)

            # feature drift + risk flip vs Paper-1 record
            rec = risk_flat.get(f"{scene}_bbox{i}")
            if rec is not None:
                ref_area = rec["geometric"]["area"]
                ref_circ = rec["geometric"]["circularity"]
                ref_sol = rec["geometric"]["solidity"]
                drifts.append(
                    {
                        "area": rel_err(feats["area"], ref_area),
                        "circularity": rel_err(feats["circularity"], ref_circ),
                        "solidity": rel_err(feats["solidity"], ref_sol),
                    }
                )
                s_depth = rec["risk_assessment"]["component_scores"]["s_depth"]
                s_pos = rec["risk_assessment"]["component_scores"]["s_pos"]
                ref_cat = rec["risk_assessment"]["severity_level"]
                new_cat = categorize(composite(s_depth, area_score(feats["area"]), s_pos))
                flips.append(int(new_cat != ref_cat))

        if warmup:
            print(f"[warmup {idx+1}/{N_WARMUP}] {scene}")
            continue

        n_boxes_total += len(boxes)
        per_scene.append(
            {
                "scene": scene,
                "n_boxes": len(boxes),
                "t_load": round(t_load, 4),
                "t_encoder": round(t_encoder, 4),
                "t_decoder": round(t_decoder, 4),
                "t_total": round(t_load + t_encoder + t_decoder, 4),
            }
        )

    def agg(key):
        vals = [r[key] for r in per_scene]
        return {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
        }

    drift_agg = {}
    for k in ["area", "circularity", "solidity"]:
        vals = [d[k] for d in drifts if d[k] is not None]
        drift_agg[k] = {
            "mean_rel_err": round(float(np.mean(vals)), 4),
            "median_rel_err": round(float(np.median(vals)), 4),
            "p95_rel_err": round(float(np.percentile(vals, 95)), 4),
        }

    summary = {
        "config": {
            "model": adapter.name,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "n_scenes": len(per_scene),
            "n_warmup": N_WARMUP,
            "protocol": "identical to bench_stage2_baseline.py",
        },
        "model_load_s": round(model_load_s, 2),
        "timing": {k: agg(k) for k in ["t_load", "t_encoder", "t_decoder", "t_total"]},
        "total_boxes": n_boxes_total,
        "fidelity_iou_vs_vith": {
            "mean": round(float(np.mean(ious)), 4),
            "median": round(float(np.median(ious)), 4),
            "min": round(float(np.min(ious)), 4),
            "frac_above_0.8": round(float(np.mean(np.array(ious) > 0.8)), 4),
            "frac_above_0.9": round(float(np.mean(np.array(ious) > 0.9)), 4),
            "n": len(ious),
        },
        "feature_drift": drift_agg,
        "risk_category_flip_rate": round(float(np.mean(flips)), 4) if flips else None,
        "n_risk_checked": len(flips),
        "per_scene": per_scene,
    }

    out = P2 / "results" / f"bench_stage2_{adapter.name.replace('.', '_')}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scene"}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
