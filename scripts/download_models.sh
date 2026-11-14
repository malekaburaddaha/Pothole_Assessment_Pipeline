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

echo "==> Six distilled ConvNeXt-Tiny severity models (6 x 112 MB) — Paper 2, Stage 4"
REL=https://github.com/malekaburaddaha/Pothole_Assessment_Pipeline/releases/download/v2.0-paper2
mkdir -p "$CKPT/distilled_severity"
for v in image twotower; do for s in 42 43 44; do
  f="distill_convnext_tiny_${v}_s${s}.pt"
  [ -f "$CKPT/distilled_severity/$f" ] || curl -fL -o "$CKPT/distilled_severity/$f" "$REL/$f"
done; done
(cd "$CKPT/distilled_severity" && curl -fsL "$REL/SHA256SUMS" | sha256sum -c -)

cat <<'NOTE'

Local VLM (Paper 2, Stage 5) is pulled from the HuggingFace Hub on first use:
    Qwen/Qwen3-VL-8B-Instruct
Set HF_HOME to a disk with ~20 GB free before the first run.
NOTE
