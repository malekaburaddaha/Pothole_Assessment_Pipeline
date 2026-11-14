"""
Build the Paper-2 human warning-evaluation kit (Paper-1 protocol, Tables 10-12).

Produces in human_eval_kit/:
  Warning_Evaluation_Template.xlsx      blank, REAL 50-scene stratified sample,
                                        blinded A/B (assignment hidden from raters)
  assignment_key_DO_NOT_SHARE.json      method<->label key (analysis only)
  EXAMPLE_Annotator_{1,2,3}.xlsx        TRAINING EXAMPLES ONLY — 5 fictional
                                        EX-scenes; cannot map onto real data
  INSTRUCTIONS.md                       rater instructions

Real ratings sheets are analyzed later by analyze_human_eval.py (to be run once
annotators return files). The EXAMPLE files are never analyzed.
"""
import json
import random
import os
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

P2 = Path(os.environ.get("POTHOLE_ROOT", "."), "Paper2_Improvements")

# WHICH SYSTEM IS BEING RATED - parameterised 2026-10-14.
#
# The completed round rated ...-FULL.json, which is the UNCONSTRAINED generator on
# the pre-final corpus. Paper 2 reports the CONSTRAINED system (...-FINAL.json,
# whose maneuver distribution matches results/advisory_safety.json exactly:
# reduce_speed 1089 / maintain_lane_and_speed 16 / brake_firmly 14). The published
# human numbers are therefore a lower bound, as stated in the Limitations section.
#
# Re-running against the reported system:
#     python build_eval_kit.py --source FINAL --out human_eval_kit_v2
# The default stays FULL so re-running this file cannot overwrite the completed
# round's artefacts by accident.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--source", default="FULL",
                 choices=["FULL", "FINAL", "CONSTRAINED"],
                 help="which warnings run to rate")
_ap.add_argument("--out", default="human_eval_kit",
                 help="output directory, relative to Paper2_Improvements")
_ap.add_argument("--seed", type=int, default=7,
                 help="sampling seed; CHANGE IT for a fresh round so the new "
                      "sample is not the same 50 scenes the first round saw")
_args = _ap.parse_args()

KIT = P2 / _args.out
KIT.mkdir(parents=True, exist_ok=True)
_SRC = P2 / f"results/warnings_local_qwen3vl8b-{_args.source}.json"
WARN = json.loads(_SRC.read_text())["scenes"]
print(f"rating system: {_args.source}  ({_SRC.name})")
print(f"output dir   : {KIT}")

random.seed(_args.seed)

# ---- stratified 50-scene sample by local urgency (proportional) ----
by_urg = {}
for sid, rec in WARN.items():
    by_urg.setdefault(rec["local"]["urgency_level"], []).append(sid)
QUOTA = {"critical": 9, "high": 14, "medium": 23, "low": 4}
sample = []
for u, q in QUOTA.items():
    sample += random.sample(sorted(by_urg[u]), q)
random.shuffle(sample)

HEAD = ["Scene_ID", "Scene_Image", "Warning_A", "Maneuver_A", "Warning_B",
        "Maneuver_B", "Clarity_A (1-5)", "Clarity_B (1-5)",
        "Safety_A (1-5)", "Safety_B (1-5)", "Preference (A/B/Tie)", "Notes"]

def style_sheet(ws, banner=None, banner_color="1F4E79"):
    # BUG FIX 2026-10-14: the header was written INTO row 1, which is a DATA row
    # because the caller appends rows starting at row 1. That silently destroyed
    # the first sampled scene - the kit advertised 50 scenes and shipped 49, and
    # round 1 of the human study ran on 49 as a result. Always insert a fresh row
    # for the header instead of overwriting.
    ws.insert_rows(1)
    if banner:
        ws.insert_rows(1)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEAD))
        c = ws.cell(row=1, column=1, value=banner)
        c.font = Font(bold=True, color="FFFFFF", size=13)
        c.fill = PatternFill("solid", fgColor=banner_color)
        c.alignment = Alignment(horizontal="center")
    hr = 2 if banner else 1
    for j, h in enumerate(HEAD, 1):
        c = ws.cell(row=hr, column=j, value=h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="D6E4F0")
    widths = [12, 30, 46, 46, 46, 46, 14, 14, 14, 14, 18, 30]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + j)].width = w
    for row in ws.iter_rows(min_row=hr + 1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

# ---- REAL blank template ----
wb = Workbook()
ws = wb.active
ws.title = "Ratings"
key = {}
for sid in sample:
    rec = WARN[sid]
    local = rec["local"]
    p1 = rec["paper1_gemini"] or {}
    if random.random() < 0.5:
        A = ("local", local)
        B = ("paper1", p1)
    else:
        A = ("paper1", p1)
        B = ("local", local)
    key[sid] = {"A": A[0], "B": B[0]}
    ws.append([sid, f"scenes/{sid}/{sid}.JPG",
               A[1].get("warning_message", ""), A[1].get("maneuver_suggestion", ""),
               B[1].get("warning_message", ""), B[1].get("maneuver_suggestion", ""),
               "", "", "", "", "", ""])
style_sheet(ws)
wsi = wb.create_sheet("Instructions", 0)
for line in [
    "WARNING EVALUATION — PAPER 2 (50 scenes)",
    "",
    "1. Open the scene image (path in column B; base: ENPAD-A_Extended_Dataset_with_Warnings/).",
    "2. Read Warning A and Warning B. You do NOT know which system produced which.",
    "3. Rate EACH warning:",
    "   - Clarity (1-5): clear, well structured, easy to understand?",
    "   - Safety appropriateness (1-5): suitable for safety-critical driving decisions?",
    "4. Preference: which warning would you prefer while driving? A, B, or Tie.",
    "5. Rate independently. Do not discuss with other annotators.",
    "6. Time estimate: 45-60 minutes. Save the file with your initials in the name.",
]:
    wsi.append([line])
wsi.column_dimensions["A"].width = 110
wb.save(KIT / "Warning_Evaluation_Template.xlsx")
(KIT / "assignment_key_DO_NOT_SHARE.json").write_text(json.dumps(
    {"note": "A/B -> method mapping. Analysis only. Never show to annotators.",
     "key": key}, indent=2))

# ---- 3 TRAINING EXAMPLE copies (fictional scenes, cannot touch real data) ----
EX = [
    ("EX-001", "Deep pothole ahead in your left wheel path. Move right and slow to 20 mph.",
     "Shift right within the lane; reduce to 20 mph before the defect.",
     "Pothole detected ahead. Drive carefully.", "Slow down.",
     (5, 2, 5, 2, "A", "A is specific (lane + speed); B is vague.")),
    ("EX-002", "Minor surface defect on the shoulder. No action needed.",
     "Maintain lane and speed.",
     "Minor defect near shoulder; maintain course with normal awareness.",
     "Maintain current lane position and speed.",
     (4, 4, 4, 4, "Tie", "Both appropriate for a low-risk defect.")),
    ("EX-003", "CRITICAL: large cavity across both wheel paths. Brake now!",
     "Emergency braking; stop if safe.",
     "Severe pothole spanning the lane ahead. Reduce speed to 10 mph and steer to the shoulder edge if clear.",
     "Slow to 10 mph; use shoulder-side clearance if available.",
     (3, 5, 3, 5, "B", "A over-alarms ('brake now!') without guidance; B is calm and actionable.")),
    ("EX-004", "Moderate pothole in right wheel path at medium distance.",
     "Reduce speed to 25 mph and shift slightly left.",
     "Pothole ahead on the right. Adjust position.",
     "Move left a little.",
     (5, 3, 4, 3, "A", "A quantifies distance and speed; B lacks specifics.")),
    ("EX-005", "Two shallow potholes ahead, outside the wheel paths.",
     "Maintain lane; no speed change required.",
     "Multiple road defects ahead. Proceed with caution and reduce speed.",
     "Reduce speed slightly.",
     (4, 3, 4, 2, "A", "B suggests unnecessary slowdown for out-of-path shallow defects (over-warning).")),
]
BANNER = ("TRAINING EXAMPLE — FICTIONAL SCENES (EX-###). SHOWS HOW TO FILL THE SHEET. "
          "NOT DATA. NEVER ANALYZED, NEVER IN THE PAPER.")
for k in (1, 2, 3):
    wb = Workbook(); ws = wb.active; ws.title = "Ratings"
    for (sid, wa, ma, wbg, mb, r) in EX:
        ca, cb, sa, sb, pref, note = r
        # small per-annotator variation so examples show honest disagreement
        ca2 = max(1, min(5, ca + random.choice([-1, 0, 0])))
        sb2 = max(1, min(5, sb + random.choice([0, 0, 1])))
        ws.append([sid, "(fictional example scene — no image)",
                   wa, ma, wbg, mb, ca2, cb, sa, sb2, pref, note])
    style_sheet(ws, banner=BANNER, banner_color="C00000")
    wb.save(KIT / f"EXAMPLE_Annotator_{k}_TRAINING_ONLY.xlsx")

(KIT / "INSTRUCTIONS.md").write_text("""# Human Warning Evaluation — Kit Contents

- `Warning_Evaluation_Template.xlsx` — the REAL sheet (50 scenes, blinded A/B).
  Give one copy to each annotator; collect filled copies for analysis.
- `EXAMPLE_Annotator_{1,2,3}_TRAINING_ONLY.xlsx` — filled TRAINING EXAMPLES on
  fictional EX-scenes. Show these to annotators so they see how to fill the sheet.
  They are marked in red, use no real scenes, and are never analyzed or reported.
- `assignment_key_DO_NOT_SHARE.json` — maps A/B to systems. Analysis only.

Protocol mirrors Paper 1 (blinded, independent, clarity + safety 1-5 + preference).
""")
print(f"sample: {len(sample)} scenes | quota: {QUOTA}")
print("kit written to", KIT)
