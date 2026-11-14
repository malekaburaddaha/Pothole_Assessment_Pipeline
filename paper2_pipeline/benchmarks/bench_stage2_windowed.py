"""
Paper 2 — Plan B: prompt-local windowed segmentation.

Instead of encoding the full 3680x1965 frame, crop a square context window around
each GT box (side = max(WINDOW_FACTOR * max(bw,bh), MIN_WINDOW)), segment inside
the window, and paste the mask back. Gives small potholes ~10x more effective
resolution inside the model's 1024 input budget.

    python bench_stage2_windowed.py --model tinysam
    python bench_stage2_windowed.py --model mobilesam
    python bench_stage2_windowed.py --model vith        # upper-bound sanity check

Metrics identical to the candidate benchmark (IoU vs stored Paper-1 ViT-H
full-frame masks, feature drift, risk flips). Timing: per-box encoder+decoder
summed per scene (windowing moves encode cost from per-scene to per-box).

Output: results/bench_stage2_windowed_<model>.json
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
WINDOW_FACTOR = 4.0
MIN_WINDOW = 384


def load_model(name):
    if name == "tinysam":
        sys.path.insert(0, str(TP / "TinySAM"))
        from tinysam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_DIR / "tinysam.pth"))
        sam.to(DEVICE).eval()
        return SamPredictor(sam), False
    if name == "mobilesam":
        sys.path.insert(0, str(TP / "MobileSAM"))
        from mobile_sam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_DIR / "mobile_sam.pt"))
        sam.to(DEVICE).eval()
        return SamPredictor(sam), True
    if name == "vith":
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_h"](
            checkpoint=str(ROOT / "Code_Files/sam_vit_h.pth")
        )
        sam.to(DEVICE).eval()
        return SamPredictor(sam), True
    raise ValueError(name)


def predict_box(predictor, supports_multimask_kw, box):
    if supports_multimask_kw:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box)[None, :],
            multimask_output=True,
        )
    else:  # TinySAM: always multimask, no kwarg
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=np.array(box)[None, :]
        )
    return masks, scores


def window_for_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = int(max(WINDOW_FACTOR * max(bw, bh), MIN_WINDOW))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    wx1 = max(0, cx - side // 2)
    wy1 = max(0, cy - side // 2)
    wx2 = min(img_w, wx1 + side)
    wy2 = min(img_h, wy1 + side)
    wx1 = max(0, wx2 - side)  # re-anchor if clamped at right/bottom edge
    wy1 = max(0, wy2 - side)
    return wx1, wy1, wx2, wy2


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


def area_score(a):
    if a < 10:
        a = 10
    return float(np.clip((np.log10(a) - 2.0) / 2.5 * 100, 0, 100))


def composite(sd, sa, sp):
    return round(0.5 * sd + 0.3 * sa + 0.2 * sp, 1)


def categorize(s):
    return "Critical" if s >= 80 else "High" if s >= 60 else "Medium" if s >= 40 else "Low"


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


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tinysam", "mobilesam", "vith"])
    args = ap.parse_args()

    risk_flat = json.loads(RISK_FLAT.read_text())
    predictor, multimask_kw = load_model(args.model)

    scenes = sorted(p.stem for p in IMAGES_DIR.glob("*.JPG"))[: N_SCENES + N_WARMUP]
    per_scene, ious, area_errs, flips = [], [], [], []
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
        img_h, img_w = rgb.shape[:2]

        t_seg = 0.0
        for i, box in enumerate(boxes):
            sync()
            t0 = time.perf_counter()
            wx1, wy1, wx2, wy2 = window_for_box(box, img_w, img_h)
            crop = rgb[wy1:wy2, wx1:wx2]
            predictor.set_image(crop)
            local_box = [box[0] - wx1, box[1] - wy1, box[2] - wx1, box[3] - wy1]
            masks, scores = predict_box(predictor, multimask_kw, local_box)
            sync()
            t_seg += time.perf_counter() - t0
            if warmup:
                continue

            best_local = masks[int(np.argmax(scores))]
            full = np.zeros((img_h, img_w), dtype=bool)
            full[wy1:wy2, wx1:wx2] = best_local
            feats = calculate_mask_metrics(full)

            stored_path = PAPER1_MASKS_DIR / scene / f"mask_{i}.png"
            if stored_path.exists():
                ref = cv2.imread(str(stored_path), cv2.IMREAD_GRAYSCALE) > 127
                ious.append(mask_iou(full, ref))

            rec = risk_flat.get(f"{scene}_bbox{i}")
            if rec is not None:
                ref_area = rec["geometric"]["area"]
                if ref_area:
                    area_errs.append(abs(feats["area"] - ref_area) / ref_area)
                cs = rec["risk_assessment"]["component_scores"]
                new_cat = categorize(
                    composite(cs["s_depth"], area_score(feats["area"]), cs["s_pos"])
                )
                flips.append(int(new_cat != rec["risk_assessment"]["severity_level"]))

        if warmup:
            print(f"[warmup {idx+1}/{N_WARMUP}] {scene}")
            continue

        n_boxes_total += len(boxes)
        per_scene.append(
            {
                "scene": scene,
                "n_boxes": len(boxes),
                "t_load": round(t_load, 4),
                "t_seg_all_boxes": round(t_seg, 4),
                "t_total": round(t_load + t_seg, 4),
            }
        )

    def agg(key):
        vals = [r[key] for r in per_scene]
        return {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
        }

    summary = {
        "config": {
            "model": f"{args.model} + prompt-local windowing",
            "window_factor": WINDOW_FACTOR,
            "min_window": MIN_WINDOW,
            "device": DEVICE,
            "torch": torch.__version__,
            "n_scenes": len(per_scene),
        },
        "timing": {k: agg(k) for k in ["t_load", "t_seg_all_boxes", "t_total"]},
        "total_boxes": n_boxes_total,
        "fidelity_iou_vs_vith_fullframe": {
            "mean": round(float(np.mean(ious)), 4),
            "median": round(float(np.median(ious)), 4),
            "min": round(float(np.min(ious)), 4),
            "frac_above_0.8": round(float(np.mean(np.array(ious) > 0.8)), 4),
            "frac_above_0.9": round(float(np.mean(np.array(ious) > 0.9)), 4),
            "n": len(ious),
        },
        "area_rel_err_mean": round(float(np.mean(area_errs)), 4),
        "risk_category_flip_rate": round(float(np.mean(flips)), 4),
        "per_scene": per_scene,
    }

    out = P2 / "results" / f"bench_stage2_windowed_{args.model}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scene"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
