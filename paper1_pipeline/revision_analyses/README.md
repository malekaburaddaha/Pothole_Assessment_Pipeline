# Paper 1: analyses added in the revision

Each script below produced a result reported in *Pothole Risk Assessment with
Geometrically Grounded Vision Language Models* (IEEE Access, under review). Its
output is in `results/`.

## Run from the released data alone

These two scripts need only the files in `data/`:

```bash
python paper1_pipeline/revision_analyses/panel_analysis.py     # Tables 6, 7, 8, 16
python paper1_pipeline/revision_analyses/warning_analysis.py   # Tables 9-13
```

| Script | Paper | Reproduces |
|---|---|---|
| `panel_analysis.py` | Sec. IV-B, VI-F | both human severity panels: rater agreement, validation-run and released-label κ, per-class precision/recall, confusion matrices, subset analysis |
| `warning_analysis.py` | Sec. V | Stuart–Maxwell test, method agreement, Fleiss κ per method, scene-level Wilcoxon tests with bootstrap intervals, preference analysis |

## Need the full imagery, an RGB-D dataset, a GPU or an API key

These scripts read the authors' local data layout. Set `POTHOLE_ROOT` to the folder
that holds it.

| Script | Paper | Inputs | Result |
|---|---|---|---|
| `m1_metric_depth_validation.py` | Sec. VI-A | PothRGBD (Yurdakul & Taşdemir, IEEE DataPort, doi:10.21227/z8eq-sf60) | `m1_metric_depth_gt.json` (plane-fit depth for 910 potholes) |
| `m1_vlm_vs_metric.py` | Sec. VI-A, Table 14 | the above plus the Gemini API | `m1_vlm_vs_metric.json` |
| `m4_prompt_ablation_gemini.py` | Sec. VI-B, Table 15 | ENPAD scenes, the published prompt scripts in `paper1_pipeline/vlm/`, and the Gemini API | `m4_ablation_gemini.json` |
| `m6b_scoring_reconciliation.py` | Sec. VI-C | `data/enpad/risk_scores_flat.json` | `m6b_reconciliation.json` |
| `phaseA_statistics.py` | Sec. VI-C, V | ENPAD outputs and the rating sheets | `phaseA_statistics_results.json` (risk-weight sensitivity and the earlier statistics run) |
| `m8_segmentation_and_depthmodels.py` | Sec. VI-D | PothRGBD, SAM ViT-H, Depth Anything v2 | `m8_results.json` |
| `m10_run_variability.py` | Sec. VI-E | the Gemini API | `m10_run_variability.json` |

The prompt ablation in the paper uses Gemini 2.5 Pro in all six arms, with one
generation configuration. Arm 1 is the published image-only prompt and arm 6 the
published hybrid prompt, both verbatim; arms 2–5 remove components from the hybrid
prompt.
