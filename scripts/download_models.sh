#!/usr/bin/env bash
# Fetch the model weights the pipelines need. Weights are deliberately NOT in git.
set -euo pipefail
CKPT="${1:-checkpoints}"
mkdir -p "$CKPT"

echo "==> SAM ViT-H (2.4 GB) — Stage 2 of both pipelines"
[ -f "$CKPT/sam_vit_h.pth" ] || curl -L -o "$CKPT/sam_vit_h.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "==> YOLO11m (Paper 1 detection stage)"
[ -f "$CKPT/yolo11m.pt" ] || curl -L -o "$CKPT/yolo11m.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

cat <<'NOTE'

Local VLM (Paper 2, Stages 4-5) is pulled from the HuggingFace Hub on first use:
    Qwen/Qwen3-VL-8B-Instruct
Set HF_HOME to a disk with ~20 GB free before the first run.

The six distilled ConvNeXt-Tiny severity checkpoints (6 x 112 MB, too large for git) go in
    $CKPT/distilled_severity/distill_convnext_tiny_{image,twotower}_s{42,43,44}.pt
They are trained by paper2_pipeline/stage4_5_local_vlm/train_depth_distilled.py on the
full ENPAD-A corpus (see data/README.md), or are available from the corresponding author.
NOTE
