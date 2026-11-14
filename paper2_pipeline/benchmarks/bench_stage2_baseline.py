"""
Paper 2 — Stage 2+3 Baseline Benchmark (SAM ViT-H, exactly as Paper 1 ran it)

Measures, per scene, with proper CUDA synchronization:
  - t_load     : imread + BGR->RGB conversion (CPU, disk-bound)
  - t_encoder  : predictor.set_image()  (ViT-H image encoder, once per image)
  - t_decoder  : predictor.predict() summed over all GT boxes (prompt+mask decoder)
  - t_metrics  : Stage 3 geometric feature computation summed over boxes
Also verifies reproduction: IoU of freshly computed best-mask vs the stored
Paper-1 mask PNG for the same scene/bbox_id.

Output: Paper2_Improvements/results/bench_stage2_baseline.json
"""
import json
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
IMAGES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images"
BBOXES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_bboxes"
PAPER1_MASKS_DIR = ROOT / "Main_Work/Dataset/With_Distance/sam_predictions_large"
CHECKPOINT = ROOT / "Code_Files/sam_vit_h.pth"
OUT_PATH = ROOT / "Paper2_Improvements/results/bench_stage2_baseline.json"

N_SCENES = 50          # deterministic sample: first 50 sorted scenes
N_WARMUP = 3           # scenes run before timing starts (CUDA init, autotune)
DEVICE = "cuda:0"


def sync():
    torch.cuda.synchronize()


def calculate_mask_metrics(mask):
    """Stage 3 feature extraction — verbatim logic from Code_Files/main.py."""
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


def load_bboxes(scene):
    path = BBOXES_DIR / f"{scene}.txt"
    boxes = []
    for line in path.read_text().splitlines():
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


def main():
    scenes = sorted(p.stem for p in IMAGES_DIR.glob("*.JPG"))[: N_SCENES + N_WARMUP]

    print(f"Loading SAM ViT-H from {CHECKPOINT} on {DEVICE}")
    t0 = time.perf_counter()
    sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT))
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    sync()
    model_load_s = time.perf_counter() - t0
    print(f"Model load: {model_load_s:.1f}s")

    per_scene = []
    repro_ious = []

    for idx, scene in enumerate(scenes):
        warmup = idx < N_WARMUP
        img_path = IMAGES_DIR / f"{scene}.JPG"
        boxes = load_bboxes(scene)
        if not boxes:
            continue

        # --- t_load: disk + decode + colorspace ---
        t0 = time.perf_counter()
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_load = time.perf_counter() - t0

        # --- t_encoder: ViT-H forward (once per image) ---
        sync()
        t0 = time.perf_counter()
        predictor.set_image(image_rgb)
        sync()
        t_encoder = time.perf_counter() - t0

        # --- t_decoder + t_metrics: per box ---
        t_decoder = 0.0
        t_metrics = 0.0
        scene_repro = []
        for i, box in enumerate(boxes):
            sync()
            t0 = time.perf_counter()
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box)[None, :],
                multimask_output=True,
            )
            sync()
            t_decoder += time.perf_counter() - t0

            best = masks[np.argmax(scores)]

            t0 = time.perf_counter()
            _ = calculate_mask_metrics(best)
            t_metrics += time.perf_counter() - t0

            # Reproduction check vs stored Paper-1 mask
            stored = PAPER1_MASKS_DIR / scene / f"mask_{i}.png"
            if stored.exists() and not warmup:
                ref = cv2.imread(str(stored), cv2.IMREAD_GRAYSCALE) > 127
                scene_repro.append(mask_iou(best, ref))

        if warmup:
            print(f"[warmup {idx+1}/{N_WARMUP}] {scene}")
            continue

        repro_ious.extend(scene_repro)
        rec = {
            "scene": scene,
            "n_boxes": len(boxes),
            "t_load": round(t_load, 4),
            "t_encoder": round(t_encoder, 4),
            "t_decoder": round(t_decoder, 4),
            "t_metrics": round(t_metrics, 4),
            "t_total": round(t_load + t_encoder + t_decoder + t_metrics, 4),
            "repro_iou_mean": round(float(np.mean(scene_repro)), 4) if scene_repro else None,
        }
        per_scene.append(rec)
        print(
            f"[{idx-N_WARMUP+1}/{N_SCENES}] {scene}: total {rec['t_total']:.2f}s "
            f"(load {t_load:.2f} | enc {t_encoder:.2f} | dec {t_decoder:.3f} "
            f"x{len(boxes)} | feat {t_metrics:.4f}) repro IoU "
            f"{rec['repro_iou_mean']}"
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
            "model": "SAM ViT-H (Paper 1 baseline)",
            "checkpoint": str(CHECKPOINT),
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "n_scenes": len(per_scene),
            "n_warmup": N_WARMUP,
            "image_size": "3680x1965",
        },
        "model_load_s": round(model_load_s, 2),
        "timing": {k: agg(k) for k in ["t_load", "t_encoder", "t_decoder", "t_metrics", "t_total"]},
        "total_boxes": int(sum(r["n_boxes"] for r in per_scene)),
        "reproduction_iou": {
            "mean": round(float(np.mean(repro_ious)), 4),
            "median": round(float(np.median(repro_ious)), 4),
            "min": round(float(np.min(repro_ious)), 4),
            "frac_above_0.9": round(float(np.mean(np.array(repro_ious) > 0.9)), 4),
            "n": len(repro_ious),
        },
        "per_scene": per_scene,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scene"}, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
