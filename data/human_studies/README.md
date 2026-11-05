# Human studies (Paper 1), anonymized

Raters are identified only as R1–R3. The IDs are assigned per study and do not link
raters across studies. No names, free-text notes or file metadata are included.

| File | Study | Contents |
|---|---|---|
| `depth_panel1.csv` | first severity panel, 42 potholes | each rater's class. The `shown_*` columns are the model predictions (and one model's confidence) that were displayed to these annotators, so their labels are **not** independent of the models. These columns are also the validation-run predictions used for model selection. |
| `depth_panel2.csv` | second, prediction-free severity panel, 102 potholes | each rater's class and difficulty flag. `in_first_panel` marks the 42 potholes shared with panel 1. No model output was shown. |
| `warning_study_ratings.csv` | warning study, 50 scenes × 3 raters | clarity and safety ratings (1–5) for warnings A and B, preference (A/B/TIE), and which method produced A and B |

One safety rating (`R2`, scene `G0011590`, warning B) was entered as 6 on the 1–5
scale. It is kept as recorded. Capping it at 5 changes the mean safety difference
from +0.73 to +0.72.

`paper1_pipeline/revision_analyses/panel_analysis.py` and `warning_analysis.py`
reproduce the paper's Tables 6–13 and Table 16 from these files.

License: CC BY 4.0 (see `../../DATA_LICENSE.md`).
