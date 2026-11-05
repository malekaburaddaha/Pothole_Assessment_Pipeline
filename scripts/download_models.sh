#!/usr/bin/env bash
# Fetch the SAM ViT-H weights used in Stage 2. Weights are deliberately not in git.
set -euo pipefail
CKPT="${1:-checkpoints}"
mkdir -p "$CKPT"
echo "==> SAM ViT-H (2.4 GB)"
[ -f "$CKPT/sam_vit_h.pth" ] || curl -L -o "$CKPT/sam_vit_h.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
echo "Stages 4-5 call the Gemini API; put GOOGLE_API_KEY in a .env file (never commit it)."
