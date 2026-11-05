# ENPAD: complete pipeline outputs (Paper 1)

These are the outputs of the five-stage pipeline described in *Pothole Risk Assessment
with Geometrically Grounded Vision Language Models* (IEEE Access, under review). They
are released so that every downstream analysis can be reproduced exactly, since a
hosted model cannot be guaranteed to give the same output on a re-run.

| File | Contents | Records |
|---|---|---|
| `severity_gemini_2_5_pro.json` | Stage 4 severity class, confidence and reasoning per pothole, with its geometric features and position | 3,092 potholes |
| `assessments_with_risk.json` | per-scene records: severity, composite risk score and its components, risk level | 1,119 scenes |
| `risk_scores_flat.json` | one record per pothole: severity, geometry, position and composite risk with components | 3,092 potholes |
| `warnings_hybrid.json` | Stage 5 warnings from the multimodal hybrid prompt | 1,119 scenes |
| `warnings_image_only.json` | Stage 5 warnings from the image-only baseline | 1,119 scenes |

**Read before use.** The severity classes are model-generated, perceived visual
categories, not depth measurements. In the paper they agree only weakly with an
independent, prediction-free human panel (κ = 0.139) and with physically measured
depth (κ = 0.098). They are a resource for developing and comparing methods, not a
benchmark of physical severity.

The source dataset has 3,097 annotated pothole boxes. The pipeline assessed 3,092 of
them: three boxes lie outside the transformed frame and two were not processed.

License: CC BY 4.0 (see `../../DATA_LICENSE.md`).
