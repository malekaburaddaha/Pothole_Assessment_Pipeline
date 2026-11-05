# Data

| Folder | Contents |
|---|---|
| `sample/` | 30 perspective-transformed scenes with their ground-truth boxes and Stage 4 labels, enough to run the pipeline without downloading anything |
| `enpad/` | the complete pipeline outputs: Stage 4 severity for all 3,092 assessed potholes, and risk scores and both warning sets for all 1,119 scenes (see `enpad/README.md`) |
| `human_studies/` | anonymized annotations and ratings from the three human studies (see `human_studies/README.md`) |

## Images

The scenes come from the public pothole dataset of Nienaber, Booysen and Kroon (2015).
The dataset has 3,097 annotated pothole boxes in 1,119 scenes. The pipeline assessed
3,092 of them: three boxes lie outside the transformed frame and two were not
processed. Stage 1 (`paper1_pipeline/detection/Transformation_Pothole_Detection.py`)
regenerates the perspective-transformed images from the original dataset. To run the
full corpus, place the transformed images and boxes at

```
data/enpad_images/
├── images/     1119 × .JPG
└── bboxes/     1119 × .txt   (pixel x1 y1 x2 y2, 3680×1965 frame)
```

This folder is gitignored, so it will not be committed by accident.

## Label provenance

The severity labels are **model-generated** perceived visual classes, produced by
Gemini 2.5 Pro in Stage 4. They are not human ground truth and not depth
measurements. In the paper they agree only weakly with a prediction-free human panel
(κ = 0.139) and with physically measured depth (κ = 0.098). The box annotations are
human ground truth.

## Conventions

- Distance band: from the bottom edge y2 of the box, far ≤ 449, medium 450–514,
  near ≥ 515 (3680×1965 frame).
- Lateral zone: from the left edge x_min of the box, with cuts at x = 1000, 1840
  and 2700.

Annotations are licensed CC BY 4.0 (`../DATA_LICENSE.md`). The images remain subject
to the original dataset's terms.
