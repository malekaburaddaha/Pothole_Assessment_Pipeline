# Pothole Detection and Assessment Project

This repository contains a comprehensive implementation of pothole detection and severity assessment using hybrid computer vision and vision-language model (VLM) techniques.

## Overview

This project implements a novel hybrid pipeline that combines deterministic computer vision metrics with VLM-based depth inference to create rich, multi-dimensional pothole assessments for autonomous vehicle applications. The system processes road imagery to detect, segment, characterize, and score potholes based on geometric properties, spatial positioning, and inferred depth characteristics.

## Project Structure

### Core Detection Pipeline

- **pothole_detection_project.py**: Main implementation of the pothole detection system
- **Transformation_Pothole_Detection.py**: Handles perspective transformation and image preprocessing for pothole detection
- **bboxes_categorization.py**: Tools for categorizing bounding boxes around detected potholes by distance and lateral position
- **eval_metrics.py**: Evaluation metrics for measuring detection performance
- **plot_gt_Vs_dt.py**: Visualization tools for comparing ground truth vs. detected potholes
- **gt bbox to cropped bboxes.py**: Utility for cropping images based on ground truth bounding boxes

### VLM Assessment Pipeline (`VLM_Code/`)

The VLM_Code directory contains the complete implementation of our hybrid assessment methodology:

#### Configuration & Setup

- **config.py**: Central configuration file containing all paths, VLM settings, depth category definitions, API parameters, and logging configurations. Manages adaptive dual-image strategy thresholds and checkpoint settings.

- **requirements.txt**: Python dependencies including OpenAI API client, tqdm for progress tracking, and supporting libraries.

- **.env.example**: Template for environment variables, primarily for storing OpenAI API credentials securely.

#### Core Pipeline Scripts

##### vlm_inference.py
Main VLM depth inference engine implementing adaptive multi-image strategy:
- Automatically determines whether to send full scene + crop (for small/distant potholes) or crop only (for near/clear potholes)
- Constructs structured prompts combining pre-computed metrics with visual analysis instructions
- Implements robust error handling with exponential backoff retry logic for API rate limits
- Tracks token usage and cost in real-time with checkpoint saving every 100 potholes
- Uses GPT-5 (or GPT-4o) with structured JSON output for reliable depth classification
- **Output**: `vlm_depth_predictions.json` containing depth categories, confidence scores, and reasoning for each pothole

##### data_fusion.py
Merges VLM depth predictions with original geometric assessments:
- Loads VLM inference results and original summary JSONs
- Enriches each pothole record with depth category, confidence, reasoning, and metadata
- Preserves all original CV-computed metrics while adding VLM-inferred attributes
- Generates fusion statistics (success rate, missing predictions)
- **Output**: Enriched assessment files in `final_assessments/` directory

##### calculate_risk_scores.py
Computes unified pothole risk scores (1-10 scale):
- Implements weighted formula combining depth (40%), area (25%), distance (20%), and solidity (15%)
- Depth scores: low=2.0, medium=5.5, high=9.0
- Distance scores: Near=9.0, Medium=5.0, Far=2.0
- Normalizes area and solidity to 0-10 scale
- Produces risk distribution statistics (low/medium/high categories)
- **Output**: Updated assessment files with `risk_score` field added to each pothole

#### Orchestration

##### run_pipeline.py
Master execution script that runs the complete end-to-end pipeline:
1. VLM depth inference (48-72 hours for full dataset)
2. Data fusion (merges predictions with original data)
3. Risk score calculation (generates final severity rankings)

Includes comprehensive logging and error handling with progress tracking across all pipeline stages.

## Methodology

### Phase 1: Deterministic Feature Extraction
Uses traditional computer vision to calculate objective, measurable properties:
- **Geometric metrics**: Area, perimeter, circularity, aspect ratio, solidity
- **Positional features**: Distance category (Near/Medium/Far), lateral position (in-path, in-lane)
- **Segmentation**: SAM (Segment Anything Model) for precise pothole masks

### Phase 2: VLM Depth Inference
Addresses the "no ground truth depth" problem using vision-language models as weak supervisors:
- **Adaptive strategy**: Sends full scene + crop for small/distant potholes; crop only for clear cases
- **Structured prompting**: Combines pre-computed metrics with visual analysis instructions
- **Depth classification**: Low (<2cm), Medium (2-5cm), High (>5cm)
- **Validation**: Human-in-the-loop evaluation + comparison against RGB-D ground truth

### Phase 3: Risk Synthesis
Combines all metrics into actionable severity assessment:
- **Weighted formula**: Scientifically justified weights based on vehicle damage literature
- **Unified score**: Single 1-10 metric for prioritization and decision-making
- **Distribution analysis**: Characterizes dataset severity profile

## Usage

### Setup

```
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd VLM_Code
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Pipeline

**Complete end-to-end execution:**
```
python VLM_Code/run_pipeline.py
```

**Individual stages:**
```
# Stage 1: VLM depth inference (takes 48-72 hours)
python VLM_Code/vlm_inference.py

# Stage 2: Data fusion
python VLM_Code/data_fusion.py

# Stage 3: Risk score calculation
python VLM_Code/calculate_risk_scores.py
```

### Monitoring Progress

```
# View real-time logs
tail -f data/vlm_outputs/inference.log

# Check cost tracking
cat data/vlm_outputs/cost_tracking.json

# View checkpoint progress
ls -lh data/vlm_outputs/vlm_depth_predictions.json
```

## Output Format

### Final Assessment JSON Structure

Each scene produces an enriched JSON with complete per-pothole assessment:

```
{
  "image_name": "G0010121.JPG",
  "results": [
    {
      "bbox_id": 0,
      "gt_bbox": ,
      "mask_metrics": {
        "area": 7850.0,
        "perimeter": 342.5,
        "circularity": 0.81,
        "aspect_ratio": 1.12,
        "solidity": 0.87
      },
      "Positional_feature": {
        "Distance_Category": "Near",
        "Lateral_Position": "in_path_left"
      },
      "vlm_depth_inference": {
        "depth_category": "high",
        "confidence": "high",
        "reasoning": "Deep shadows and sharp edge definition indicate significant depth",
        "used_dual_image": false,
        "inference_timestamp": "2025-11-06T10:45:23"
      },
      "risk_score": 8.45
    }
  ]
}
```

## Dataset

This pipeline processes the **Road Pothole Images** dataset (11,497 images, 7,588 labeled potholes) and produces the **DPAD (Dearborn Pothole Assessment Dataset)** with 10+ attributes per pothole including VLM-inferred depth and unified risk scores.

## Cost Estimation

- **VLM inference**: ~$150-220 for 8,000-10,000 potholes using GPT-5/GPT-4o
- **Processing time**: 48-72 hours for complete dataset (API rate limits)
- **Checkpoint system**: Automatic resume capability if interrupted

## Research Applications

This codebase supports research on:
1. **Hybrid CV + VLM methodologies** for perception tasks lacking ground truth
2. **Weak supervision** using vision-language models as annotators
3. **Multi-modal risk assessment** for autonomous vehicle decision-making
4. **Dataset creation** for pothole severity benchmarking

## Citation

If you use this code or methodology in your research, please cite:
```
[Paper citation to be added upon publication]
```

## License

[Your license here]

## Contact

[Your contact information]

---

**Note**: The VLM pipeline requires an OpenAI API key and processes images asynchronously. Estimated cost for full dataset processing is $150-220 USD. Checkpoint files are saved every 100 potholes to enable resume functionality.
```
