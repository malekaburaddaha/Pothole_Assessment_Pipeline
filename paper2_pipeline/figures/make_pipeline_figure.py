"""Paper 2 main figure: the fully local pipeline on one real scene (G0065615).

Every number is read from a result file or quoted from the paper's own tables; the
frame, SAM masks, severities, risk scores and advisory text are the pipeline's actual
outputs for the scene. Writes a vector PDF for LaTeX and a 300-dpi PNG.
"""
import json
import textwrap
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
SID = "G0065615"
IMG = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images" / f"{SID}.JPG"
MASKS = ROOT / "Main_Work/Dataset/With_Distance/sam_predictions_large" / SID
OUT = P2 / "figures"

_FONT_DIR = os.environ.get("ROBOTO_DIR", "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF")
for w in ("Regular", "Medium", "Bold", "Italic"):
    try:
        fm.fontManager.addfont(f"{_FONT_DIR}/Roboto-{w}.ttf")
    except (FileNotFoundError, OSError):  # falls back to the matplotlib default font
        pass
plt.rcParams["font.family"] = "Roboto"

INK, MUTE = "#14213d", "#5c677d"
C_IN, C_SEG, C_FEAT, C_SEV, C_ADV, C_OUT = "#4a5568", "#2b6cb0", "#117a8b", "#6b46c1", "#276749", "#1a202c"
SEV = {"low": "#2e9e5b", "medium": "#e0a21b", "high": "#c0392b"}

assess = json.loads((P2 / "results/paper2_full_assessments.json").read_text())


def find(o):
    if isinstance(o, dict):
        if o.get("scene_id") == SID:
            return o
        for v in o.values():
            r = find(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r:
                return r


scene = find(assess)
adv = json.loads((P2 / "results/warnings_local_qwen3vl8b-FINAL.json").read_text())["scenes"]
adv = adv[SID] if isinstance(adv, dict) else [x for x in adv if x["scene_id"] == SID][0]
loc = adv["local"]

frame = Image.open(IMG).convert("RGB")
W, H = frame.size
frame_small = frame.copy(); frame_small.thumbnail((1500, 1500))
potholes = scene["potholes"]
masks = [np.array(Image.open(MASKS / f"mask_{i}.png").convert("L")) > 127 for i in range(len(potholes))]
mask_cols = [(231, 76, 60), (241, 196, 15), (46, 204, 113)]

# zoom window around the defects (pixels of the 3680 x 1965 frame)
ZX1, ZY1, ZX2, ZY2 = 1450, 150, 3150, 1587
overlay = np.array(frame).astype(float)
for m, c in zip(masks, mask_cols):
    overlay[m] = 0.45 * overlay[m] + 0.55 * np.array(c)
overlay_zoom = Image.fromarray(overlay.astype(np.uint8)).crop((ZX1, ZY1, ZX2, ZY2))
overlay_zoom.thumbnail((1200, 1200))

FS = 0.93                                   # figure is shown at ~0.65x in the paper
fig = plt.figure(figsize=(11, 4.43), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160); ax.set_ylim(0, 64.5); ax.axis("off")
LS = 2.45                                    # line step in axis units


def T(x, y, s, size=10, color=INK, weight="normal", ha="left", va="center", style="normal", z=6, **k):
    return ax.text(x, y, s, fontsize=size * FS, color=color, fontweight=weight, ha=ha, va=va,
                   fontstyle=style, zorder=z, **k)


def card(x, y, w, h, edge, fill="white", lw=1.8, r=1.0):
    ax.add_patch(FancyBboxPatch((x + .25, y - .25), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc="#0000001a", ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=3))


def arrow(p0, p1, color=INK, lw=2.2, ms=14, z=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 zorder=z, shrinkA=0, shrinkB=0))


def header(x, n, title, color):
    ax.add_patch(plt.Circle((x + 1.6, 57.2), 1.5, fc=color, ec="white", lw=1.2, zorder=8))
    T(x + 1.6, 57.15, str(n), 12, "white", "bold", "center", z=9)
    T(x + 4.0, 57.2, title, 14, color, "bold")


def place(img, x, y_top, w, edge=None):
    h = w * img.height / img.width
    ax.imshow(img, extent=(x, x + w, y_top - h, y_top), aspect="auto", zorder=4)
    if edge:
        ax.add_patch(Rectangle((x, y_top - h), w, h, fc="none", ec=edge, lw=1.5, zorder=5))
    return h


T(80, 62.4, "Fully local pothole severity assessment and driver advisory", 18, INK, "bold", "center")

XS = [1.5, 33.1, 64.7, 96.3, 127.9]
WS = [28.6] * 5
TOP = 53.6
CH = 24.0                                    # common card / image height
GAP = 0.3


def cap(x, title, color, lines):
    T(x, TOP - CH - 2.4, title, 11.5, color, "bold")
    for k, ln in enumerate(lines):
        T(x, TOP - CH - 2.4 - LS * (k + 1), ln, 10.5, MUTE)


# ---- stage 1: input (full frame, boxes the pipeline is prompted with)
header(XS[0], 1, "Input", C_IN)
FX1, FX2 = 700, 3000                          # centre crop shown, px of the 3680-wide frame
frame_c = frame_small.crop((int(FX1 / W * frame_small.width), 0, int(FX2 / W * frame_small.width), frame_small.height))
h = place(frame_c, XS[0], TOP, WS[0], edge=C_IN)
for p in potholes:
    x1, y1, x2, y2 = p["bbox_xyxy"]
    ax.add_patch(Rectangle((XS[0] + (x1 - FX1) / (FX2 - FX1) * WS[0], TOP - y2 / H * h), (x2 - x1) / (FX2 - FX1) * WS[0],
                           (y2 - y1) / H * h, fc="none", ec="#f1c40f", lw=1.8, zorder=6))
cap(XS[0], "Dashcam frame", INK, ["3680 × 1965 px, centre", "crop; box prompts"])

# ---- stage 2: segmentation
header(XS[1], 2, "Segment", C_SEG)
zh = place(overlay_zoom, XS[1], TOP, WS[1], edge=C_SEG)
cap(XS[1], "SAM ViT-H", C_SEG, ["fp16 encoder + compile", "0.137 s · IoU 0.9993"])
arrow((XS[0] + WS[0] + GAP, TOP - CH / 2), (XS[1] - GAP, TOP - CH / 2), C_SEG)
zoom_h = zh

# ---- stage 3: geometry
header(XS[2], 3, "Measure", C_FEAT)
card(XS[2], TOP - CH, WS[2], CH, C_FEAT)
g = potholes[0]["geometric"]
T(XS[2] + 1.4, TOP - 2.2, "pothole 1", 10.5, C_FEAT, "bold")
feat = [("area", f"{g['area']:,} px"), ("circularity", f"{g['circularity']:.2f}"),
        ("aspect ratio", f"{g['aspect_ratio']:.2f}"), ("solidity", f"{g['solidity_corrected']:.2f}"),
        ("distance", potholes[0]["positional"]["Distance_Category"].lower()),
        ("lane zone", potholes[0]["positional"]["Lateral_Position"].replace("_", " "))]
for k, (a, b) in enumerate(feat):
    yy = TOP - 5.4 - 2.9 * k
    T(XS[2] + 1.4, yy, a, 10.5, MUTE)
    T(XS[2] + WS[2] - 1.4, yy, b, 10.5, INK, "bold", "right")
cap(XS[2], "Geometry", C_FEAT, ["6 mask features, band,", "lane zone · 0.009 s"])
arrow((XS[1] + WS[1] + GAP, TOP - CH / 2), (XS[2] - GAP, TOP - CH / 2), C_FEAT)

# ---- stage 4: severity and risk
header(XS[3], 4, "Assess", C_SEV)
card(XS[3], TOP - CH, WS[3], CH, C_SEV)
T(XS[3] + 1.4, TOP - 2.2, "severity · risk", 10.5, C_SEV, "bold")
for k, p in enumerate(potholes):
    yy = TOP - 6.6 - 4.9 * k
    sev = p["depth"]["paper2_distilled"]
    ax.add_patch(plt.Circle((XS[3] + 2.4, yy), .9, fc=tuple(v / 255 for v in mask_cols[k]), ec="none", zorder=6))
    ax.add_patch(FancyBboxPatch((XS[3] + 4.4, yy - 1.3), 10.4, 2.6, boxstyle="round,pad=0,rounding_size=1.1",
                                fc=SEV[sev], ec="none", zorder=6))
    T(XS[3] + 9.6, yy, sev, 10, "white", "bold", "center")
    r = p["risk_assessment"]["composite_score"]
    ax.add_patch(Rectangle((XS[3] + 15.4, yy - .7), 8.8, 1.4, fc="#e6e9ef", ec="none", zorder=5))
    ax.add_patch(Rectangle((XS[3] + 15.4, yy - .7), 8.8 * r / 100, 1.4, fc=C_SEV, ec="none", zorder=6))
    T(XS[3] + WS[3] - 1.2, yy, f"{r:.0f}", 10, INK, "bold", "right")
T(XS[3] + 1.4, TOP - CH + 2.4, "R = 0.5 S$_d$ + 0.3 S$_a$ + 0.2 S$_p$", 9.5, MUTE)
cap(XS[3], "Distilled ConvNeXt", C_SEV, ["6-model ensemble", "+ risk score · 0.030 s"])
arrow((XS[2] + WS[2] + GAP, TOP - CH / 2), (XS[3] - GAP, TOP - CH / 2), C_SEV)

# ---- stage 5: advisory
header(XS[4], 5, "Advise", C_ADV)
card(XS[4], TOP - CH, WS[4], CH, C_ADV)
ax.add_patch(FancyBboxPatch((XS[4] + 1.0, TOP - 4.6), 12.0, 2.7, boxstyle="round,pad=0,rounding_size=1.1",
                            fc=SEV[loc["urgency_level"]], ec="none", zorder=6))
T(XS[4] + 7.0, TOP - 3.25, "URGENCY " + loc["urgency_level"].upper(), 8.6, "white", "bold", "center")
ax.add_patch(FancyBboxPatch((XS[4] + 13.4, TOP - 4.6), 14.2, 2.7, boxstyle="round,pad=0,rounding_size=1.1",
                            fc=C_SEG, ec="none", zorder=6))
T(XS[4] + 20.5, TOP - 3.25, loc["maneuver_type"].replace("_", " ").upper(), 8.6, "white", "bold", "center")
for k, s in enumerate(textwrap.wrap("\u201c" + loc["warning_message"] + "\u201d", 27)[:5]):
    T(XS[4] + 1.4, TOP - 8.2 - 2.7 * k, s, 10, INK, style="italic")
cap(XS[4], "Qwen3-VL-8B, local", C_ADV, ["closed maneuver set", "1.08 s · 94.8% calibrated"])
arrow((XS[3] + WS[3] + GAP, TOP - CH / 2), (XS[4] - GAP, TOP - CH / 2), C_ADV)

# ---- replaced cloud calls
for x0, w0, lab in ((XS[3], WS[3], "cloud VLM: 13.3 s, \\$0.080"),
                    (XS[4], WS[4], "cloud VLM: 3.5 s, \\$0.025")):
    # plain two-line label: a strike-through line drawn over small text smears
    # into an unreadable bar once the figure is scaled to column width
    y = 15.6
    T(x0 + w0 / 2, y, "replaces " + lab, 10, "#8a94a6", "normal", "center")
    T(x0 + w0 / 2, y - 2.6, "now local, no API call", 9.5, "#2e7d32", "bold", "center")

# ---- banner
ax.add_patch(FancyBboxPatch((1.5, 1.2), 157.0, 9.0, boxstyle="round,pad=0,rounding_size=1.2",
                            fc="#f3f6fa", ec=INK, lw=1.6, zorder=3))
T(80, 7.3, "Per scene: 20.1 s and \\$0.105 with cloud APIs   to   1.27 s and \\$0 locally", 14, INK,
  "bold", "center")
T(80, 3.6, "16× faster · offline · no parse-failure fallback", 11, MUTE, "normal", "center")

fig.savefig(OUT / "paper2_pipeline_figure.pdf")
fig.savefig(OUT / "paper2_pipeline_figure.png", dpi=300)
print("saved", OUT / "paper2_pipeline_figure.pdf")
