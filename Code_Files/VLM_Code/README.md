cat > Code_Files/VLM_Code/README.md << 'EOF'
# VLM Pothole Assessment Pipeline

This pipeline uses Vision Language Models (VLM) to assess pothole depth and calculate risk scores.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. Configure paths in `Config.py` according to your data structure.

## Usage

Run the complete pipeline:
```bash
python Run_Pipeline.py
```

Or run individual steps:
```bash
python VLM_Inference.py      # Step 1: VLM depth inference
python Data_Fusion.py         # Step 2: Merge predictions
python calculate_risk_scores.py  # Step 3: Calculate risk scores
```

## Files

- `VLM_Inference.py`: Main VLM inference with adaptive multi-image strategy
- `Data_Fusion.py`: Merges VLM predictions with original data
- `calculate_risk_scores.py`: Calculates unified risk scores
- `Run_Pipeline.py`: Master script to run complete pipeline
- `Config.py`: Configuration settings
- `requirements.txt`: Python dependencies

## Configuration

Edit `Config.py` to adjust:
- Model selection (gpt-4o vs gpt-4o-mini)
- Adaptive strategy thresholds
- Depth category definitions
- Risk score weights
EOF