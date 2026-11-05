"""
R2-M8 — (b) real segmentation IoU against human masks, and
        (c) quantitative evidence that foundation depth models miss potholes.

R2: "SAM's internally predicted IoU score is not equivalent to actual IoU against
a reference mask; segmentation quality should be evaluated using manually
annotated masks." and "the conclusion that all three foundation depth models fail
is based primarily on qualitative examples in Figure 4 and should be supported
quantitatively."

Both are answered on PothRGBD, which is independent of the study data and carries
(i) human-drawn polygon masks and (ii) sensor depth in millimetres.

  M8b  SAM ViT-H, prompted with the polygon's bounding box (identical protocol to
       the paper), scored against the HUMAN polygon -> true IoU; then correlate
       true IoU with SAM's self-reported predicted IoU.
  M8c  Depth-Anything-V2 monocular depth vs the depth sensor: for each pothole,
       measure the depth step between the pothole interior and the surrounding
       road, in sensor mm and in the model's relative units (z-scored per image),
       and test whether the model reproduces the physical contrast.

Output: Paper1_Resubmission/m8_results.json
"""
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
RES = ROOT / "Paper1_Resubmission"
DS = ROOT / "PothRGBD_Dataset/PUBLIC_POTHOLE_DATASET"
SAM_CKPT = ROOT / "Code_Files/sam_vit_h.pth"
GT = RES / "m1_metric_depth_gt.json"
N_FRAMES = 120
RING_PX = 25
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_polys(stem, H, W):
    lab = next(DS.glob(f"labels/{stem}*.txt"), None)
    if not lab:
        return []
    out = []
    for line in lab.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        p = np.array([float(t) for t in v[1:]], np.float32).reshape(-1, 2)
        p[:, 0] *= W
        p[:, 1] *= H
        out.append(p.astype(np.int32))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-depth", action="store_true")
    args = ap.parse_args()
    dev = args.device

    gt = {(r["image"], r["poly_id"]): r for r in json.loads(GT.read_text())["potholes"]}
    stems = sorted({p.name.split("_color")[0] for p in DS.glob("images/*.jpg")})[:N_FRAMES]
    out = {}

    # ---------------------------------------------------------- M8b
    if not args.skip_sam:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
        sam.to(dev).eval()
        pred = SamPredictor(sam)
        ious, sam_scores = [], []
        for si, stem in enumerate(stems):
            ip = next(DS.glob(f"images/{stem}*.jpg"), None)
            if not ip:
                continue
            im = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            H, W = im.shape[:2]
            polys = load_polys(stem, H, W)
            if not polys:
                continue
            pred.set_image(im)
            for pi, poly in enumerate(polys):
                ref = np.zeros((H, W), np.uint8)
                cv2.fillPoly(ref, [poly], 1)
                if ref.sum() < 400:
                    continue
                x1, y1 = poly[:, 0].min(), poly[:, 1].min()
                x2, y2 = poly[:, 0].max(), poly[:, 1].max()
                m, sc, _ = pred.predict(box=np.array([x1, y1, x2, y2])[None, :],
                                        multimask_output=True)
                k = int(np.argmax(sc))
                pm = m[k].astype(np.uint8)
                inter = int((pm & ref).sum())
                union = int((pm | ref).sum())
                if union:
                    ious.append(inter / union)
                    sam_scores.append(float(sc[k]))
            if (si + 1) % 40 == 0:
                print(f"  SAM {si+1}/{len(stems)} frames, {len(ious)} masks")
        ious = np.array(ious)
        ss = np.array(sam_scores)
        out["M8b_segmentation_vs_human_masks"] = {
            "n_masks": int(len(ious)),
            "true_iou": {"mean": round(float(ious.mean()), 4),
                         "median": round(float(np.median(ious)), 4),
                         "p10": round(float(np.percentile(ious, 10)), 4),
                         "pct_above_0.5": round(100 * float((ious > .5).mean()), 2),
                         "pct_above_0.75": round(100 * float((ious > .75).mean()), 2)},
            "sam_predicted_iou": {"mean": round(float(ss.mean()), 4),
                                  "median": round(float(np.median(ss)), 4)},
            "mean_overestimate_predicted_minus_true": round(float((ss - ious).mean()), 4),
            "pearson_r_predicted_vs_true": round(float(np.corrcoef(ss, ious)[0, 1]), 4),
            "note": "SAM's self-reported IoU is not a substitute for measured IoU; "
                    "reported here on independent human polygon masks.",
        }
        print(json.dumps(out["M8b_segmentation_vs_human_masks"], indent=1))
        del sam, pred
        torch.cuda.empty_cache()

    # ---------------------------------------------------------- M8c
    if not args.skip_depth:
        from transformers import pipeline
        from PIL import Image
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if dev.startswith("cuda") else -1)
        rows = []
        for si, stem in enumerate(stems):
            ip = next(DS.glob(f"images/{stem}*.jpg"), None)
            dp = DS / "depths" / f"{stem}_depth.npy"
            if not (ip and dp.exists()):
                continue
            im = Image.open(ip).convert("RGB")
            W, H = im.size
            Z = np.load(dp).astype(np.float32)
            Z[(Z <= 0) | (Z > 8000)] = np.nan
            if Z.shape != (H, W):
                Z = cv2.resize(Z, (W, H), interpolation=cv2.INTER_NEAREST)
            est = np.array(pipe(im)["predicted_depth"].squeeze().cpu())
            est = cv2.resize(est, (W, H))
            est = (est - np.nanmean(est)) / (np.nanstd(est) + 1e-6)   # z-score per image
            for pi, poly in enumerate(load_polys(stem, H, W)):
                key = (ip.name, pi)
                if key not in gt:
                    continue
                mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                ring = cv2.dilate(mask, np.ones((RING_PX, RING_PX), np.uint8)) - mask
                mi, ri = mask == 1, ring == 1
                if mi.sum() < 50 or ri.sum() < 200:
                    continue
                sensor_step = float(np.nanmedian(Z[mi]) - np.nanmedian(Z[ri]))   # mm, +ve = deeper
                model_step = float(np.nanmedian(est[mi]) - np.nanmedian(est[ri]))  # z units
                ring_noise = float(np.nanstd(est[ri]))
                rows.append({"sensor_step_mm": sensor_step, "model_step_z": model_step,
                             "model_step_over_ring_noise": model_step / (ring_noise + 1e-6),
                             "metric_mm": gt[key]["depth_mm_p90"]})
            if (si + 1) % 40 == 0:
                print(f"  depth {si+1}/{len(stems)} frames, {len(rows)} potholes")
        s = np.array([r["sensor_step_mm"] for r in rows])
        m = np.array([r["model_step_z"] for r in rows])
        snr = np.array([r["model_step_over_ring_noise"] for r in rows])
        gtmm = np.array([r["metric_mm"] for r in rows])
        ok = np.isfinite(s) & np.isfinite(m)
        from scipy import stats as st
        out["M8c_monocular_depth_vs_sensor"] = {
            "model": "depth-anything/Depth-Anything-V2-Small-hf",
            "n_potholes": int(ok.sum()),
            "sensor_step_mm_median": round(float(np.median(s[ok])), 2),
            "model_step_z_median": round(float(np.median(m[ok])), 4),
            "model_step_within_1_ring_sd_pct": round(100 * float((np.abs(snr[ok]) < 1).mean()), 2),
            "pearson_r_model_vs_sensor_step": round(float(np.corrcoef(m[ok], s[ok])[0, 1]), 4),
            "spearman_r_model_step_vs_metric_depth": round(
                float(st.spearmanr(m[ok], gtmm[ok]).statistic), 4),
            "interpretation": "If the model recovered pothole geometry, its interior-vs-road "
                              "step would scale with the sensor step. A near-zero correlation "
                              "and a step smaller than local surface noise quantify the "
                              "qualitative Figure-4 observation.",
        }
        print(json.dumps(out["M8c_monocular_depth_vs_sensor"], indent=1))

    p = RES / "m8_results.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(out)
    p.write_text(json.dumps(prev, indent=1))
    print("Saved:", p)


if __name__ == "__main__":
    main()
