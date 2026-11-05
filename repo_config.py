"""Central path resolution, so the code runs outside the machine it was written on.

The original scripts hardcoded absolute paths to the development workstation. This
module replaces that with one resolution rule:

    POTHOLE_DATA_ROOT   environment variable, if set
    otherwise           this repository's own directory

A fresh clone therefore works with no configuration for anything that only needs the
bundled 30-scene sample. To run against the full corpus, lay it out as

    $POTHOLE_DATA_ROOT/data/enpad_images/images/*.JPG
    $POTHOLE_DATA_ROOT/data/enpad_images/bboxes/*.txt

and the same scripts pick it up.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("POTHOLE_DATA_ROOT", REPO))

# bundled sample (always present)
SAMPLE_DIR = REPO / "data/sample"
SAMPLE_IMAGES = SAMPLE_DIR / "images"
SAMPLE_BBOXES = SAMPLE_DIR / "bboxes"
SAMPLE_LABELS = SAMPLE_DIR / "labels_gemini_2_5_pro.json"

# full corpus (gitignored; present only if the user installed it)
FULL_IMAGES = ROOT / "data/enpad_images/images"
FULL_BBOXES = ROOT / "data/enpad_images/bboxes"

# model weights, fetched by scripts/download_models.sh
CKPT_DIR = Path(os.environ.get("POTHOLE_CKPT_DIR", REPO / "checkpoints"))
SAM_CKPT = CKPT_DIR / "sam_vit_h.pth"

# pipeline outputs
RESULTS = Path(os.environ.get("POTHOLE_RESULTS", REPO / "results"))


def images_dir(prefer_full: bool = True) -> Path:
    """Full corpus when installed, otherwise the bundled sample."""
    if prefer_full and FULL_IMAGES.is_dir() and any(FULL_IMAGES.glob("*.JPG")):
        return FULL_IMAGES
    return SAMPLE_IMAGES


def bboxes_dir(prefer_full: bool = True) -> Path:
    if prefer_full and FULL_BBOXES.is_dir() and any(FULL_BBOXES.glob("*.txt")):
        return FULL_BBOXES
    return SAMPLE_BBOXES


def labels_path(prefer_full: bool = True) -> Path:
    full = ROOT / "data/enpad_images/labels_gemini_2_5_pro.json"
    return full if (prefer_full and full.exists()) else SAMPLE_LABELS
