"""
Paper 2 — Plan A: accelerate SAM ViT-H itself, no architecture change.

Variants:
  --variant fp16      : encoder forward under torch.autocast(float16)
  --variant fp16-comp : fp16 + torch.compile on the image encoder

Same protocol/metrics as candidate benchmark (50 scenes after 3 warmup, fidelity
vs stored Paper-1 masks, risk flips with depth/position held fixed).

Output: results/bench_stage2_vith-<variant>.json
"""
import argparse
import json
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
IMAGES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images"
BBOXES_DIR = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_bboxes"
PAPER1_MASKS_DIR = ROOT / "Main_Work/Dataset/With_Distance/sam_predictions_large"
RISK_FLAT = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/final_risk_scores_flat.json"
CHECKPOINT = ROOT / "Code_Files/sam_vit_h.pth"

N_SCENES = 50
N_WARMUP = 3
DEVICE = "cuda:0"


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


class AutocastEncoderPredictor(SamPredictor):
    """SamPredictor whose image-encoder forward runs under fp16 autocast.

    Only set_image's encoder pass is wrapped; the prompt encoder / mask decoder
    keep fp32 (they are <5% of compute and precision-sensitive thresholds live
    in the decoder's mask upscaling).
    """

    @torch.no_grad()
    def set_torch_image(self, transformed_image, original_image_size):
        # Mirrors SamPredictor.set_torch_image but wraps the encoder in autocast.
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = self.model.image_encoder(input_image)
        self.features = feats.float()
        self.is_image_set = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fp16", "fp16-comp"])
    args = ap.parse_args()

    risk_flat = json.loads(RISK_FLAT.read_text())

    t0 = time.perf_counter()
    sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT))
    sam.to(DEVICE).eval()
    if args.variant == "fp16-comp":
        sam.image_encoder = torch.compile(sam.image_encoder)
    predictor = AutocastEncoderPredictor(sam)
    sync()
    model_load_s = time.perf_counter() - t0
    print(f"Model load: {model_load_s:.1f}s (variant={args.variant})")

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

        sync()
        t0 = time.perf_counter()
        predictor.set_image(rgb)
        sync()
        t_encoder = time.perf_counter() - t0

        t_decoder = 0.0
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
            if warmup:
                continue

            best = masks[int(np.argmax(scores))]
            feats = calculate_mask_metrics(best)

            stored_path = PAPER1_MASKS_DIR / scene / f"mask_{i}.png"
            if stored_path.exists():
                ref = cv2.imread(str(stored_path), cv2.IMREAD_GRAYSCALE) > 127
                ious.append(mask_iou(best, ref))

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
            print(f"[warmup {idx+1}/{N_WARMUP}] {scene} enc={t_encoder:.2f}s")
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

    summary = {
        "config": {
            "model": f"SAM ViT-H optimized ({args.variant})",
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "n_scenes": len(per_scene),
        },
        "model_load_s": round(model_load_s, 2),
        "timing": {k: agg(k) for k in ["t_load", "t_encoder", "t_decoder", "t_total"]},
        "total_boxes": n_boxes_total,
        "fidelity_iou_vs_vith_fp32": {
            "mean": round(float(np.mean(ious)), 4),
            "median": round(float(np.median(ious)), 4),
            "min": round(float(np.min(ious)), 4),
            "frac_above_0.9": round(float(np.mean(np.array(ious) > 0.9)), 4),
            "n": len(ious),
        },
        "area_rel_err_mean": round(float(np.mean(area_errs)), 4),
        "risk_category_flip_rate": round(float(np.mean(flips)), 4),
        "per_scene": per_scene,
    }

    out = P2 / "results" / f"bench_stage2_vith-{args.variant}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scene"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
