# Pipeline 1 — cloud-assisted hazard assessment

Stages 1-3 run locally; Stages 4-5 call Gemini 2.5 Pro.

## Layout

| Path | Role |
|---|---|
| `main.py` | Stages 1-3: SAM segmentation + six geometric features per pothole |
| `detection/Transformation_Pothole_Detection.py` | perspective rectification + detection |
| `detection/pothole_detection_project.py` | YOLO detection training/eval (notebook-derived) |
| `detection/eval_metrics.py` | detection metrics |
| `detection/bboxes_categorization.py` | distance / lateral-position assignment |
| `detection/images_batch.py` | crop extraction from scenes |
| `vlm/Run_Pipeline.py` | orchestrates Stages 4-5 |
| `vlm/gemini_inference.py` | Gemini client (the production path) |
| `vlm/VLM_Inference.py` | GPT-4o client + retry/backoff |
| `vlm/fusion_and_risk_scoring.py` | geometry + severity -> composite risk |
| `vlm/generate_scene_warnings.py` | advisory generation |
| `vlm/*_analysis*.py`, `calculate_*.py` | ablations, agreement metrics, statistics |

## Running

```bash
export GEMINI_API_KEY=...
python main.py                      # Stages 1-3
python vlm/Run_Pipeline.py          # Stages 4-5
```

## Cost and time, measured

The full 1,119-scene corpus: **8 h 14 min**, 3,092 potholes, 2 failures, ~**$117**.
Throughput held at 0.098-0.115 potholes/s across all thirty checkpoints, which means the
pipeline is **latency bound** — one request in flight, each waiting ~9.6 s — rather than
rate-limited.

## Two known defects

1. **No resume.** A checkpoint is written every 100 records and never read back. A crash
   at hour seven costs the whole run. Fix before any long job: load the checkpoint at
   startup and skip IDs already present, and write it atomically
   (temp file + `os.replace`) so a crash during the write cannot corrupt it.

2. **Redundant decoding.** `detection/images_batch.py` calls `cv2.imread` on the full
   3680×1965 scene *inside* the per-pothole loop, so a scene with four potholes is
   decoded four times. Hoist it out of the loop.

Neither is fixed here, so that the code matches what produced the published numbers.
