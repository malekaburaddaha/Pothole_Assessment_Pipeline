"""
R2-M4 - incremental prompt ablation with Gemini 2.5 Pro, built from the verbatim
published prompts.

    python Paper1_Resubmission/m4_prompt_ablation_gemini.py --n 300 --workers 8
    python Paper1_Resubmission/m4_prompt_ablation_gemini.py --dry-run     # print arm prompts, no API

Arms (the same 300 scenes as the earlier local run: random.seed(5)):

  A1 image_only         the published image-only prompt, verbatim
  A2 image_rubric       the published hybrid prompt without the worked examples and
                        without the scene assessment (rubric and task only)
  A3 image_geometry     A2 + per-pothole size and position (no depth, no risk score)
  A4 image_depth        A2 + per-pothole depth and position (no size, no risk score)
  A5 hybrid_noexamples  the published hybrid prompt without the worked examples
  A6 hybrid_full        the published hybrid prompt, verbatim

Arms A2-A5 are made by deleting sections and per-pothole lines from the verbatim
hybrid prompt; nothing is reworded except the one task sentence of A2, which
otherwise refers to "the structured assessments above" that A2 does not contain.

Every arm uses one generation configuration: the configuration of the published
hybrid run (JSON schema with the urgency enum, API-default sampling), with safety
filters off so that road-damage images are not blocked. Failed calls are retried
and, if they still fail, recorded as failures and excluded; nothing is filled in.
Results are checkpointed, so the run can be resumed.

Output: Paper1_Resubmission/m4_ablation_gemini.json
"""
import argparse
import ast
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
RES = ROOT / "Paper1_Resubmission"
CODE = Path(__file__).resolve().parents[1] / "vlm"   # the published prompt scripts
ASSESS = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/final_assessments_with_risk.json"
IMAGES = ROOT / "Main_Work/image_patches"
OUT = RES / "m4_ablation_gemini.json"
CKPT = RES / "m4_ablation_gemini.checkpoint.json"

ARMS = ["image_only", "image_rubric", "image_geometry", "image_depth",
        "hybrid_noexamples", "hybrid_full"]


def load_function(path, name):
    """Load one function from a published script without running the script."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


hybrid_prompt = load_function(CODE / "generate_scene_warnings.py", "create_scene_context_prompt")
image_only_prompt = load_function(CODE / "Images_only_warning_improved.py", "create_image_only_prompt")

A2_TASK_OLD = ("**Examine the road scene image** and integrate it with the structured assessments "
               "above. Use both visual context and quantified metrics to:")
A2_TASK_NEW = "**Examine the road scene image** and use the visual context to:"


def build_prompt(arm, scene):
    if arm == "image_only":
        return image_only_prompt()
    full = hybrid_prompt(scene)
    if arm == "hybrid_full":
        return full
    i_ex, i_sc, i_task = (full.index("## EXAMPLES"), full.index("## CURRENT SCENE ASSESSMENT"),
                          full.index("## YOUR TASK"))
    head, scene_block, task = full[:i_ex], full[i_sc:i_task], full[i_task:]
    if arm == "image_rubric":
        assert A2_TASK_OLD in task
        return head + task.replace(A2_TASK_OLD, A2_TASK_NEW)
    drop = {"image_geometry": ("- Depth:", "- Risk Score:", "**Critical/High Risk Count:**"),
            "image_depth": ("- Size:", "- Risk Score:", "**Critical/High Risk Count:**"),
            "hybrid_noexamples": ()}[arm]
    kept = "\n".join(l for l in scene_block.split("\n") if not l.strip().startswith(drop))
    return head + kept + task


def rubric_urgency(max_risk):
    return ("critical" if max_risk >= 80 else "high" if max_risk >= 60
            else "medium" if max_risk >= 40 else "low")


def make_model(model_name):
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv(CODE / ".env")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    cfg = {"response_mime_type": "application/json",
           "response_schema": {"type": "object", "properties": {
               "warning_message": {"type": "string"},
               "maneuver_suggestion": {"type": "string"},
               "urgency_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
               "reasoning": {"type": "string"},
               "affected_lanes": {"type": "array", "items": {"type": "string"}},
               "recommended_speed_reduction": {"type": "string"}},
               "required": ["warning_message", "maneuver_suggestion", "urgency_level", "reasoning"]}}
    safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in
              ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    return genai.GenerativeModel(model_name, generation_config=cfg, safety_settings=safety)


def call(model, prompt, image_path, tries=5):
    from PIL import Image
    last = ""
    for k in range(tries):
        try:
            r = model.generate_content([prompt, Image.open(image_path)])
            out = json.loads(r.text)
            if out.get("urgency_level") in ("low", "medium", "high", "critical"):
                return out
            last = "bad urgency"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:160]
            if "depleted" in last or "billing" in last.lower():
                raise RuntimeError(last)
        time.sleep(min(60, 4 * 2 ** k))
    return {"failed": True, "error": last}


def analyse(results, scenes):
    ana, ref = {}, results.get("hybrid_full", {})
    for arm in ARMS:
        ok = {s: v for s, v in results.get(arm, {}).items() if not v.get("failed")}
        urg = Counter(v["urgency_level"] for v in ok.values())
        cons = [v["urgency_level"] == rubric_urgency(scenes[s]["max_risk_score"]) for s, v in ok.items()]
        agree = [v["urgency_level"] == ref[s]["urgency_level"] for s, v in ok.items()
                 if s in ref and not ref[s].get("failed")]
        n = max(1, sum(urg.values()))
        ana[arm] = {"n": len(ok), "failed": len(results.get(arm, {})) - len(ok),
                    "urgency_distribution": dict(urg),
                    "pct_high_or_critical": round(100 * (urg["high"] + urg["critical"]) / n, 2),
                    "rubric_consistency": round(100 * sum(cons) / max(1, len(cons)), 2),
                    "agreement_with_hybrid_full": round(100 * sum(agree) / max(1, len(agree)), 2)}
    return ana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    assess = json.loads(ASSESS.read_text())
    sids = sorted(assess)
    random.seed(5)
    sids = sorted(random.sample(sids, min(a.n, len(sids))))
    scenes = {s: assess[s] for s in sids}
    imgs = {s: IMAGES / s / f"{s}.JPG" for s in sids if (IMAGES / s / f"{s}.JPG").exists()}
    print(f"scenes={len(sids)} with images={len(imgs)}")

    if a.dry_run:
        s0 = sids[0]
        for arm in ARMS:
            p = build_prompt(arm, scenes[s0])
            print(f"\n===== {arm}: {len(p)} chars =====\n{p[:200]} ... {p[-120:]}")
        return

    results = json.loads(CKPT.read_text()) if CKPT.exists() else {arm: {} for arm in ARMS}
    todo = [(arm, s) for arm in ARMS for s in imgs if s not in results[arm]]
    print(f"calls to make: {len(todo)}")
    model = make_model(a.model)
    lock, done, t0 = threading.Lock(), 0, time.time()

    def job(arm, s):
        return arm, s, call(model, build_prompt(arm, scenes[s]), imgs[s])

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, arm, s) for arm, s in todo]
        for f in as_completed(futs):
            arm, s, out = f.result()
            with lock:
                results[arm][s] = out
                done += 1
                if done % 50 == 0 or done == len(todo):
                    CKPT.write_text(json.dumps(results))
                    print(f"  {done}/{len(todo)} calls, {time.time() - t0:.0f}s", flush=True)

    out = {"config": {"backend": "gemini", "model": a.model, "n_scenes": len(imgs),
                      "seed": 5, "arms": ARMS,
                      "generation": "published hybrid configuration for every arm; API-default sampling"},
           "analysis": analyse(results, scenes), "results": results}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["analysis"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
