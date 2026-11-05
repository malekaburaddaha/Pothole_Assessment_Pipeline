"""
R2-M1 — Metric depth validation on an independent RGB-D pothole dataset.

Reviewer: "validate the categories using physically measured, stereo, LiDAR, or
other metric depth data, or redefine the outputs as perceived visual-severity."
We do the former on PothRGBD (1,000 RGB-D frames, uint16 depth in mm, YOLO
polygon labels), which we hold out entirely from the main study.

Method per pothole polygon:
  1. mask       = filled polygon
  2. ring       = dilate(mask) \\ mask  -> surrounding intact road surface
  3. plane fit  = robust least squares z = ax + by + c on ring pixels
  4. deviation  = z_measured - z_plane inside the mask (positive = below road)
  5. tilt corr  = perpendicular depth = deviation / sqrt(a^2+b^2+1)
  6. depth_mm   = 90th percentile of corrected deviation (robust to sensor spikes)
  -> category via the paper's thresholds: <20 mm low, 20-50 mm medium, >50 mm high

Output: Paper1_Resubmission/m1_metric_depth_gt.json  (per-pothole metric depth)
"""
import os
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
DS = ROOT / "PothRGBD_Dataset/PUBLIC_POTHOLE_DATASET"
OUT = ROOT / "Paper1_Resubmission/m1_metric_depth_gt.json"

MIN_AREA_PX = 400        # ignore tiny polygons (unreliable plane fit)
RING_PX = 25             # width of the surrounding-road ring
MIN_RING_PTS = 300
MAX_VALID_MM = 8000
PCTL = 90


def depth_for_image(stem):
    img_p = next(DS.glob(f"images/{stem}*.jpg"), None)
    lab_p = next(DS.glob(f"labels/{stem}*.txt"), None)
    dep_p = DS / "depths" / f"{stem}_depth.npy"
    if not (img_p and lab_p and dep_p.exists()):
        return []
    Z = np.load(dep_p).astype(np.float32)
    Z[(Z <= 0) | (Z > MAX_VALID_MM)] = np.nan
    H, W = Z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for li, line in enumerate(lab_p.read_text().splitlines()):
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array([float(t) for t in v[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= W
        pts[:, 1] *= H
        poly = pts.astype(np.int32)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        if mask.sum() < MIN_AREA_PX:
            continue
        ring = cv2.dilate(mask, np.ones((RING_PX, RING_PX), np.uint8)) - mask
        rsel = (ring == 1) & np.isfinite(Z)
        msel = (mask == 1) & np.isfinite(Z)
        if rsel.sum() < MIN_RING_PTS or msel.sum() < 50:
            continue
        # robust plane fit on the ring (2 IRLS passes)
        A = np.c_[xx[rsel], yy[rsel], np.ones(rsel.sum())]
        b = Z[rsel]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        for _ in range(2):
            resid = b - A @ coef
            keep = np.abs(resid) < 2.5 * (np.median(np.abs(resid)) * 1.4826 + 1e-6)
            if keep.sum() < MIN_RING_PTS // 2:
                break
            coef, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        a, bb, c = coef
        tilt = np.sqrt(a * a + bb * bb + 1.0)
        plane_in = a * xx[msel] + bb * yy[msel] + c
        dev = (Z[msel] - plane_in) / tilt          # mm below the road plane
        d90 = float(np.percentile(dev, PCTL))
        ring_rms = float(np.std((b - A @ coef)))   # surface-fit noise floor
        out.append({
            "image": img_p.name, "poly_id": li,
            "depth_mm_p90": round(d90, 1),
            "depth_mm_max": round(float(np.percentile(dev, 99)), 1),
            "depth_mm_median": round(float(np.median(dev)), 1),
            "plane_rms_mm": round(ring_rms, 1),
            "area_px": int(msel.sum()),
            "bbox": [int(poly[:, 0].min()), int(poly[:, 1].min()),
                     int(poly[:, 0].max()), int(poly[:, 1].max())],
        })
    return out


def categorize(mm):
    return "low" if mm < 20 else ("medium" if mm <= 50 else "high")


def main():
    stems = sorted({p.name.split("_color")[0] for p in DS.glob("images/*.jpg")})
    print(f"frames: {len(stems)}")
    recs = []
    for i, s in enumerate(stems):
        recs += depth_for_image(s)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(stems)} frames, {len(recs)} potholes")

    # quality filter: fit must be tighter than the depth we claim to measure
    good = [r for r in recs if r["plane_rms_mm"] < max(8.0, 0.5 * abs(r["depth_mm_p90"]))
            and r["depth_mm_p90"] > -20]
    for r in good:
        r["metric_category"] = categorize(r["depth_mm_p90"])

    d = np.array([r["depth_mm_p90"] for r in good])
    summary = {
        "dataset": "PothRGBD (PUBLIC_POTHOLE_DATASET), uint16 depth in mm",
        "method": "robust plane fit on surrounding road ring; tilt-corrected p90 deviation",
        "n_polygons_total": len(recs),
        "n_after_quality_filter": len(good),
        "depth_mm": {"mean": round(float(d.mean()), 1),
                     "median": round(float(np.median(d)), 1),
                     "p10": round(float(np.percentile(d, 10)), 1),
                     "p90": round(float(np.percentile(d, 90)), 1),
                     "max": round(float(d.max()), 1)},
        "category_distribution": dict(Counter(r["metric_category"] for r in good)),
        "thresholds_mm": {"low": "<20", "medium": "20-50", "high": ">50"},
        "median_plane_rms_mm": round(float(np.median([r["plane_rms_mm"] for r in good])), 2),
        "potholes": good,
    }
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "potholes"}, indent=1))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
