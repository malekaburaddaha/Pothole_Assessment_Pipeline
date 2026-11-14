"""
Paper 2 — automated LLM-judge comparison of Paper-1 (cloud) and Paper-2 (local)
warnings.

THIS IS NOT A HUMAN STUDY and is reported as such. It is an automated preference
evaluation with an independent judge model, used because a human study was not
available for this revision. Its known weaknesses (self-preference, verbosity bias,
position bias) are controlled where possible and disclosed where not:

  - blind: the judge never learns which system produced which warning;
  - position-bias controlled: every pair is judged twice with A/B order swapped,
    and only order-consistent verdicts are counted as decisive;
  - independent judge family: the judge is not the model that generated either
    warning where possible (Paper-2 warnings come from Qwen3-VL-8B; Paper-1
    warnings from Gemini 2.5 Pro; the judge is Gemini 2.5 Flash, which shares a
    family with one side - this residual conflict is disclosed, not hidden);
  - scene image supplied so that safety judgements are grounded in the scene.

Output: Paper2_Improvements/results/llm_judge_warnings.json
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
J = ROOT / "Main_Work/final_assessments/Json_and_Log_Files"
IMGS = ROOT / "ENPAD-A_Extended_Dataset_with_Warnings/scenes"
N_SCENES = 150
JUDGE = "gemini-2.5-flash"

RUBRIC = """You are evaluating two candidate driver warnings for the SAME road scene.
You are shown the scene image and two warnings, A and B. You do not know how either
was produced.

Rate EACH warning independently on two 1-5 scales:
- clarity: how clear, well structured and immediately understandable it is.
- safety: how appropriate the urgency and the recommended action are for what is
  actually visible in the scene. Penalise BOTH over-warning (treating a minor or
  out-of-path defect as an emergency) and under-warning (being casual about a deep
  defect in the wheel path). Penalise advice that would be unsafe or impractical.

Then state which single warning you would rather receive while driving: "A", "B",
or "tie" if they are genuinely equivalent.

Respond ONLY with JSON:
{"clarity_a":1-5,"safety_a":1-5,"clarity_b":1-5,"safety_b":1-5,
 "preference":"A|B|tie","justification":"one sentence"}
"""


def main():
    w1 = json.loads((J / "scene_warnings_multimodal_hybrid.json").read_text())
    import os as _o
    _f = _o.getenv("LOCAL_WARNINGS", "results/warnings_local_qwen3vl8b-FULL.json")
    w2 = json.loads((P2 / _f).read_text())["scenes"]
    common = sorted(set(w1) & set(w2))
    random.seed(23)
    scenes = random.sample(common, min(N_SCENES, len(common)))
    # JUDGE_SCENES: judge exactly the scenes of a human kit (its assignment key), so
    # judge and humans can be compared item by item rather than across two samples.
    if _o.getenv("JUDGE_SCENES"):
        scenes = sorted(json.loads((P2 / _o.getenv("JUDGE_SCENES")).read_text())["key"])

    import google.generativeai as genai
    from PIL import Image
    from dotenv import load_dotenv
    load_dotenv()  # GOOGLE_API_KEY from a local .env (never committed)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in
              ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    judge = genai.GenerativeModel(
        JUDGE, safety_settings=safety,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"})

    def ask(img, wa, wb):
        txt = (RUBRIC + f"\n\nWARNING A:\n{wa['warning_message']}\nManeuver: "
               f"{wa['maneuver_suggestion']}\n\nWARNING B:\n{wb['warning_message']}\n"
               f"Maneuver: {wb['maneuver_suggestion']}\n")
        r = judge.generate_content([txt, img])
        return json.loads(r.text)

    rows, fails = [], 0
    for i, sid in enumerate(scenes):
        ip = IMGS / sid / f"{sid}.JPG"
        if not ip.exists():
            continue
        im = Image.open(ip).convert("RGB")
        im.thumbnail((1024, 1024))
        a1, a2 = w1[sid]["warning"], w2[sid]["local"]
        try:
            fwd = ask(im, a1, a2)          # A = cloud,  B = local
            rev = ask(im, a2, a1)          # A = local,  B = cloud
        except Exception:
            fails += 1
            continue
        rows.append({
            "scene": sid,
            "cloud_clarity": [fwd["clarity_a"], rev["clarity_b"]],
            "local_clarity": [fwd["clarity_b"], rev["clarity_a"]],
            "cloud_safety": [fwd["safety_a"], rev["safety_b"]],
            "local_safety": [fwd["safety_b"], rev["safety_a"]],
            "pref_fwd": fwd["preference"], "pref_rev": rev["preference"],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(scenes)}")

    def mean(key):
        return float(np.mean([v for r in rows for v in r[key]]))

    # order-consistent preference: forward says X and reversed says the same system
    dec = Counter()
    for r in rows:
        f, v = r["pref_fwd"].lower(), r["pref_rev"].lower()
        fwd_sys = {"a": "cloud", "b": "local"}.get(f, "tie")
        rev_sys = {"a": "local", "b": "cloud"}.get(v, "tie")
        dec["consistent_" + fwd_sys if fwd_sys == rev_sys else "order_inconsistent"] += 1

    rng = np.random.default_rng(9)
    def boot_diff(k_local, k_cloud):
        d = np.array([np.mean(r[k_local]) - np.mean(r[k_cloud]) for r in rows])
        bs = [d[rng.choice(len(d), len(d), True)].mean() for _ in range(5000)]
        return round(float(d.mean()), 3), [round(float(np.percentile(bs, 2.5)), 3),
                                           round(float(np.percentile(bs, 97.5)), 3)]

    cl_d, cl_ci = boot_diff("local_clarity", "cloud_clarity")
    sf_d, sf_ci = boot_diff("local_safety", "cloud_safety")
    n_dec = dec["consistent_cloud"] + dec["consistent_local"]
    from scipy import stats as st
    binom = st.binomtest(dec["consistent_local"], n_dec, 0.5) if n_dec else None

    out = {
        "STUDY_TYPE": "AUTOMATED LLM-JUDGE EVALUATION - NOT A HUMAN STUDY",
        # local_source is recorded because omitting it is exactly how a stale
        # comparison survived review here once: two judge runs sat side by side in
        # results/ with no way to tell which corpus each had scored.
        "config": {"judge": JUDGE, "temperature": 0.0, "n_scenes": len(rows),
                   "local_source": _f, "api_failures": fails,
                   "position_bias_control": "each pair judged twice with A/B swapped",
                   "disclosed_conflict": "the judge shares a model family with the "
                                         "cloud system's generator (Gemini); this may "
                                         "bias in favour of the cloud warnings"},
        "mean_ratings": {
            "cloud_clarity": round(mean("cloud_clarity"), 3),
            "local_clarity": round(mean("local_clarity"), 3),
            "cloud_safety": round(mean("cloud_safety"), 3),
            "local_safety": round(mean("local_safety"), 3),
        },
        "differences_local_minus_cloud": {
            "clarity": {"mean": cl_d, "bootstrap_CI95": cl_ci},
            "safety": {"mean": sf_d, "bootstrap_CI95": sf_ci},
        },
        "preference": {
            "order_consistent_cloud": dec["consistent_cloud"],
            "order_consistent_local": dec["consistent_local"],
            "order_consistent_tie": dec["consistent_tie"],
            "order_inconsistent_discarded": dec["order_inconsistent"],
            "binomial_p_local_vs_cloud": float(binom.pvalue) if binom else None,
        },
        "rows": rows,
    }
    p = P2 / _o.getenv("JUDGE_OUT", "results/llm_judge_warnings.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("Saved:", p)


if __name__ == "__main__":
    main()
