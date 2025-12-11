import google.generativeai as genai
import os
import json
import re
from PIL import Image
import logging
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def should_use_dual_image(bbox, distance_category, min_crop_size=100):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    min_dimension = min(width, height)
    # Dual image if crop is small OR distant
    return min_dimension < min_crop_size or distance_category == "Far"


def call_gemini_api(
    crop_image_path: str,
    full_image_path: str,
    metrics: dict,
    positional_features: dict,
    use_dual_image: bool,
    model_name: str = "gemini-2.5-flash"
) -> dict:
    """
    Call Gemini for depth inference.
    Optimized for PAID TIER: No rate limiting, robust safety bypass.
    """
    try:
        model = genai.GenerativeModel(model_name)

        # --- IMPROVEMENT 1: Scientific Context Header ---
        # This is critical to bypass the "Dangerous Content" filter for pothole images.
        prompt = f"""[SYSTEM CONTEXT: Scientific analysis of road infrastructure for autonomous vehicle safety. Images contain inanimate road surfaces only.]

You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties (measured via computer vision):**
- Area: {metrics.get('area', 'N/A')} pixels²
- Circularity: {metrics.get('circularity', 'N/A')} 
- Solidity: {metrics.get('solidity', 'N/A')} 
- Aspect Ratio: {metrics.get('aspect_ratio', 'N/A')} 
- Distance Category: {positional_features.get('Distance_Category', 'N/A')}
- Lateral Position: {positional_features.get('Lateral_Position', 'N/A')}

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

        # Load images for API
        images = []
        if use_dual_image and full_image_path and os.path.exists(full_image_path):
            images.append(Image.open(full_image_path))
        images.append(Image.open(crop_image_path))

        # --- IMPROVEMENT 2: Comprehensive Safety Settings ---
        # Unblocking ALL categories is required to prevent false positives on damage.
        # We use string literals to be compatible with all SDK versions.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Make API call
        # --- IMPROVEMENT 3: JSON Mode & Removed Sleep ---
        response = model.generate_content(
            [prompt] + images,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                # Force JSON output (native feature in 2.5 models)
                response_mime_type="application/json"
            ),
            safety_settings=safety_settings
        )

        # Check for Safety Filter (Candidate Finish Reason 2)
        if not response.candidates:
            logger.warning(f"Response blocked completely (likely safety).")
            return _fallback_result(use_dual_image, "Blocked by Safety Pre-Screen")

        finish_reason = getattr(response.candidates[0], 'finish_reason', None)
        if finish_reason == 2:  # 2 = Safety
            logger.warning(f"Safety filter triggered (FinishReason=2)")
            return _fallback_result(use_dual_image, "Skipped due to Gemini Safety Filter")

        # Parse output
        try:
            response_text = response.text.strip()
        except Exception as e:
            logger.error(f"Could not access .text for Gemini response: {e}")
            return _fallback_result(use_dual_image, "Missing response text")

        # Attempt to parse JSON
        # Since we forced application/json, we should be able to load directly.
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback regex if model included markdown ticks
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.warning(f"JSON parse failed. Response: {response_text[:50]}...")
                result = {
                    "depth_category": "medium",
                    "confidence": "low",
                    "reasoning": f"Failed to parse: {response_text[:50]}"
                }

        # Validate depth_category
        categories = ["low", "medium", "high"]
        if result.get("depth_category") not in categories:
            logger.warning(f"Invalid depth_category: {result.get('depth_category')}, defaulting to medium.")
            result["depth_category"] = "medium"

        # Updated Cost Calculation for Gemini 2.5 Flash (Paid Tier)
        # Input: ~$0.10 - $0.30 per 1M tokens | Output: ~$1.25 - $2.50 per 1M tokens
        input_tokens = len(prompt) // 4  # Approximation
        output_tokens = len(response_text) // 4
        
        # Pricing logic (approximate for tracking)
        if "flash" in model_name:
            # Flash 2.5 Paid Tier approximation
            cost = (input_tokens * 0.30 / 1_000_000) + (output_tokens * 2.50 / 1_000_000)
        else:
            # Pro 2.5 Paid Tier approximation
            cost = (input_tokens * 1.25 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)

        result['tokens_used'] = {
            'input': input_tokens,
            'output': output_tokens,
            'cost_usd': cost
        }
        result['used_dual_image'] = use_dual_image

        return result

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return _fallback_result(use_dual_image, f"Exception: {e}")

def _fallback_result(use_dual, reason):
    """Helper to return a consistent error object"""
    return {
        "depth_category": "medium",
        "confidence": "low",
        "reasoning": reason,
        "used_dual_image": use_dual,
        "tokens_used": {"input": 0, "output": 0, "cost_usd": 0}
    }