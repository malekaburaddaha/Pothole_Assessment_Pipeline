"""
Paper 2 — Stage 4 depth inference with a LOCAL VLM, evaluated on the frozen
38-consensus human eval set. Prompt is verbatim Paper-1 (gemini_inference.py),
including geometric-feature injection and the adaptive dual-image rule.

Run (vllm env):
    python eval_depth_local.py --model Qwen/Qwen3-VL-8B-Instruct --tag qwen3vl8b

Output: results/depth_eval_<tag>.json
Bar to clear: kappa >= 0.4795 (Gemini 2.5 Pro re-scored on current labels).
"""
import argparse
import json
import time
import os
from pathlib import Path

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
EVAL_SET = P2 / "results/human_depth_eval_set.json"
TEST_DATA = ROOT / "Main_Work/test_data"

DEPTH_SCHEMA = {
    "type": "object",
    "properties": {
        "depth_category": {"type": "string", "enum": ["low", "medium", "high"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string", "maxLength": 500},
    },
    "required": ["depth_category", "confidence", "reasoning"],
}

# reasoning generated BEFORE the category commit (visual analysis can inform the
# answer instead of rationalizing it) — xgrammar emits fields in schema order.
DEPTH_SCHEMA_REASON_FIRST = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 500},
        "depth_category": {"type": "string", "enum": ["low", "medium", "high"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["reasoning", "depth_category", "confidence"],
}


def build_prompt(metrics, positional, use_dual_image):
    """Verbatim Paper-1 depth prompt (gemini_inference.py lines 44-92)."""
    prompt = f"""[SYSTEM CONTEXT: Scientific analysis of road infrastructure for autonomous vehicle safety. Images contain inanimate road surfaces only.]

You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties (measured via computer vision):**
- Area: {metrics.get('area', 'N/A')} pixels²
- Circularity: {metrics.get('circularity', 'N/A')}
- Solidity: {metrics.get('solidity', 'N/A')}
- Aspect Ratio: {metrics.get('aspect_ratio', 'N/A')}
- Distance Category: {positional.get('Distance_Category', 'N/A')}
- Lateral Position: {positional.get('Lateral_Position', 'N/A')}

**Depth Category Definitions:**
- low: Shallow surface depression (< 2cm depth)
- medium: Moderate depth pothole (2-5cm depth)
- high: Deep, hazardous pothole (> 5cm depth)

**Your Analysis:**
Based on visual evidence in the image (shadow depth, edge sharpness, surface texture, darkness/lighting), classify this pothole's DEPTH into ONE category: low, medium, or high.

**Visual Cues to Consider:**
1. Shadow depth and darkness (deeper = darker shadows)
2. Edge definition and sharpness (deeper = sharper edges)
3. Surface texture visibility (deeper = less visible bottom texture)
4. Lighting contrast between pothole interior and road surface
"""
    if use_dual_image:
        prompt += """

**Note:** Two images are provided:
1. Full road scene (for spatial context and lighting reference)
2. Cropped pothole region (target for depth assessment)

Use the full scene to understand overall lighting and road texture, then focus on the cropped region to assess depth characteristics.
"""
    prompt += """

**Response Format:**
Respond ONLY with valid JSON in this exact format:
{
  "depth_category": "low|medium|high",
  "confidence": "low|medium|high",
  "reasoning": "brief explanation of visual cues used"
}
"""
    return prompt


def should_use_dual_image(bbox, distance_category, min_crop_size=100):
    x1, y1, x2, y2 = bbox
    return min(x2 - x1, y2 - y1) < min_crop_size or distance_category == "Far"


def load_pothole_context(pothole_id):
    """bbox + features from the authoritative Paper-1 SAM summaries.

    Note: test_data/sam_predictions_large summaries are an older run WITHOUT
    Positional_feature; the With_Distance directory is the version Paper 1's
    Stage 4 actually consumed (verified: identical gt_bbox values).
    """
    scene, bbox_part = pothole_id.rsplit("_bbox", 1)
    bbox_id = int(bbox_part)
    summary = json.loads(
        (
            ROOT
            / "Main_Work/Dataset/With_Distance/sam_predictions_large"
            / scene
            / "summary.json"
        ).read_text()
    )
    for r in summary["results"]:
        if r["bbox_id"] == bbox_id:
            return r
    raise KeyError(f"{pothole_id} not in summary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="floor for vision tokens: tiny crops get upscaled")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="cap for vision tokens: huge scenes get downscaled")
    ap.add_argument("--reason-first", action="store_true",
                    help="schema orders reasoning before depth_category")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--resize-images", action="store_true",
                    help="own the preprocessing: upscale crops (>=336px short "
                         "side), downscale scenes (<=1536px long side); "
                         "written to a temp dir, bypasses framework kwargs")
    args = ap.parse_args()

    resized_dir = P2 / "stage4_5_local_vlm" / "resized_eval_images"
    if args.resize_images:
        from PIL import Image

        resized_dir.mkdir(exist_ok=True)

        def prep_image(path, is_crop):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if is_crop and min(w, h) < 336:
                s = 336 / min(w, h)
                img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
            elif not is_crop and max(w, h) > 1536:
                s = 1536 / max(w, h)
                img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
            out = resized_dir / f"{'crop' if is_crop else 'scene'}_{path.name}"
            img.save(out, quality=95)
            return out

    from vllm import LLM, SamplingParams

    eval_set = json.loads(EVAL_SET.read_text())
    records = eval_set["records"]

    # Build all requests up-front (vLLM batches them in one call).
    requests, meta = [], []
    for pid, rec in records.items():
        ctx = load_pothole_context(pid)
        metrics = ctx["mask_metrics"]
        positional = ctx["Positional_feature"]
        bbox = ctx["gt_bbox"]
        use_dual = should_use_dual_image(
            bbox, positional.get("Distance_Category", "Unknown")
        )
        prompt = build_prompt(metrics, positional, use_dual)

        crop = TEST_DATA / "image_patches" / rec["crop_image"].split("image_patches/")[1]
        full = TEST_DATA / "image_patches" / rec["full_scene"].split("image_patches/")[1]

        if args.resize_images:
            crop = prep_image(crop, is_crop=True)
            if full.exists():
                full = prep_image(full, is_crop=False)

        content = []
        if use_dual and full.exists():
            content.append({"type": "image_url", "image_url": {"url": f"file://{full}"}})
        content.append({"type": "image_url", "image_url": {"url": f"file://{crop}"}})
        content.append({"type": "text", "text": prompt})

        requests.append([{"role": "user", "content": content}])
        meta.append({"pothole_id": pid, "used_dual_image": use_dual})

    print(f"Loading {args.model} ...")
    t0 = time.perf_counter()
    mm_kwargs = {}
    if args.min_pixels:
        mm_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        mm_kwargs["max_pixels"] = args.max_pixels

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=0.85,
        allowed_local_media_path=str(ROOT),
        mm_processor_kwargs=mm_kwargs or None,
        tensor_parallel_size=args.tp,
    )
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.0f}s")

    schema = DEPTH_SCHEMA_REASON_FIRST if args.reason_first else DEPTH_SCHEMA

    # Structured output: API differs across vLLM versions; try new then old.
    try:
        from vllm.sampling_params import StructuredOutputsParams

        sp = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        sp = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            guided_decoding=GuidedDecodingParams(json=schema),
        )

    t0 = time.perf_counter()
    outputs = llm.chat(requests, sp)
    infer_s = time.perf_counter() - t0

    predictions = {}
    for m, out in zip(meta, outputs):
        text = out.outputs[0].text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"depth_category": "medium", "confidence": "low",
                      "reasoning": f"PARSE_FAIL: {text[:100]}"}
        parsed["used_dual_image"] = m["used_dual_image"]
        predictions[m["pothole_id"]] = parsed

    # ----- metrics vs frozen consensus -----
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    usable = {k: v for k, v in records.items() if v["human_consensus"]}
    y_true = [usable[k]["human_consensus"] for k in usable]
    y_pred = [predictions[k]["depth_category"] for k in usable]
    labels = ["low", "medium", "high"]

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    agree = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # agreement with Paper-1 Gemini Pro predictions (all 42)
    gp = [records[k]["paper1_predictions"]["gemini_pro"] for k in records]
    lp = [predictions[k]["depth_category"] for k in records]
    kappa_vs_gemini = float(cohen_kappa_score(gp, lp, labels=labels))

    result = {
        "config": {
            "model": args.model,
            "prompt": "verbatim Paper-1 depth template",
            "temperature": 0.1,
            "structured_output": "json-schema constrained",
            "n_requests": len(requests),
        },
        "timing": {
            "model_load_s": round(load_s, 1),
            "total_inference_s": round(infer_s, 2),
            "s_per_pothole_batched": round(infer_s / len(requests), 3),
        },
        "metrics": {
            "kappa_vs_human_consensus": round(kappa, 4),
            "agreement_percent": round(100 * agree, 2),
            "n_consensus": len(y_true),
            "bar_gemini_pro_rescored": 0.4795,
            "clears_bar": kappa >= 0.4795,
            "kappa_vs_gemini_pro_preds": round(kappa_vs_gemini, 4),
            "confusion_matrix_rows_true_low_med_high": cm,
        },
        "predictions": predictions,
    }

    out_path = P2 / "results" / f"depth_eval_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
