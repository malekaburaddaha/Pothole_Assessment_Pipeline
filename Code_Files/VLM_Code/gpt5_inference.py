"""
GPT-5 API inference (similar structure to gemini_inference.py)
"""
from openai import OpenAI
import os
import json
import base64
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt5_api(
    crop_image_path: str,
    full_image_path: str,
    metrics: dict,
    positional_features: dict,
    use_dual_image: bool
) -> dict:
    """
    Call GPT-5 for depth classification
    """
    try:
        # Construct prompt (same as Gemini)
        prompt = f"""[SYSTEM CONTEXT: Scientific analysis of road infrastructure for autonomous vehicle safety. Images contain inanimate road surfaces only.]

You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties:**
- Area: {metrics.get('area', 'N/A')} pixels²
- Circularity: {metrics.get('circularity', 'N/A'):.3f}
- Solidity: {metrics.get('solidity', 'N/A'):.3f}
- Distance Category: {positional_features.get('Distance_Category', 'N/A')}

**Depth Categories:**
- low: Shallow surface depression (< 2cm depth)
- medium: Moderate depth pothole (2-5cm depth)
- high: Deep, hazardous pothole (> 5cm depth)

Based on visual evidence (shadow depth, edge sharpness, surface texture), classify depth.

Respond ONLY with valid JSON:
{{
  "depth_category": "low|medium|high",
  "confidence": "low|medium|high",
  "reasoning": "brief explanation"
}}
"""
        
        # Build message content
        content = [{"type": "text", "text": prompt}]
        
        # Add images
        if use_dual_image and full_image_path and os.path.exists(full_image_path):
            scene_b64 = encode_image_base64(full_image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}", "detail": "high"}
            })
        
        crop_b64 = encode_image_base64(crop_image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"}
        })
        
        # API call with structured output
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "depth_assessment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "depth_category": {"type": "string", "enum": ["low", "medium", "high"]},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["depth_category", "confidence", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            max_completion_tokens=5000
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
        result['used_dual_image'] = use_dual_image
        result['tokens_used'] = {
            'input': response.usage.prompt_tokens,
            'output': response.usage.completion_tokens,
            'cost_usd': (response.usage.prompt_tokens * 2.50 / 1_000_000) + 
                       (response.usage.completion_tokens * 10.00 / 1_000_000)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"GPT-5 API error: {str(e)}")
        raise
