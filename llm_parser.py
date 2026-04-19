"""
Production LLM Parser — Hybrid Two-Model Architecture
=======================================================
Approach:
  Step 1 (Vision):  llava:7b describes the image regions in natural language.
  Step 2 (Parsing): phi3:mini maps the description + HSG + user prompt
                    to a structured JSON Influence Tensor edit schedule.

Why two models?
  llava:7b outputs garbled binary when processing large base64 images via the
  Ollama API on this hardware configuration (Q4_0 clip projector issue).
  phi3:mini produces clean, deterministic structured text output.
  Combining them gives full vision-aware, structured editing.
"""

import json
import requests
import base64
from edit_router import route_prompt

OLLAMA_BASE   = "http://localhost:11434"
VISION_MODEL  = "llava:7b"
PARSER_MODEL  = "phi3:mini"
OLLAMA_MODEL  = PARSER_MODEL   # Exported for display

PARSE_PROMPT  = """You are an AI Image Editing Conductor with professional precision.

You receive:
1. A Huffman Scene Graph (HSG) JSON describing named regions of an image.
2. A user's natural language edit instruction.

Translate the user intent into a JSON ARRAY of edits.

━━ TARGETING RULES (READ CAREFULLY) ━━
- ALWAYS prefer semantic nodes (semantic_person, semantic_sky, semantic_grass,
  semantic_tree, semantic_mountain, semantic_car) when editing a named subject.
  These are the FIRST nodes listed in the HSG and have the highest accuracy.
- NEVER pick a node with "size: tiny" for a global, subject-wide, or background edit.
  Tiny nodes cover < 2% of the image — any edit will be invisible.
- ONLY use leaf_* nodes for very precise micro-detail edits (a logo, an eye, a badge).
- If editing a PERSON (silhouette, color, clothing): target "semantic_person".
- If editing SKY, clouds, atmosphere: target "semantic_sky".
- If editing BACKGROUND, nature, scene: target "semantic_grass", "semantic_tree",
  "semantic_mountain" as appropriate.
- If editing the WHOLE IMAGE (mood, color grade, global lighting): target "SCENE_ROOT".

━━ INFLUENCE TENSOR ━━
- ZT = Texture / Material / Color change  [0.0 to 1.0]
- ZL = Light / Illumination / Brightness  [0.0 to 1.0]
- ZB = Boundary / Structure (0.0 unless a new object is added to the scene)

HSG:
{hsg_str}

User instruction: "{user_prompt}"

Respond ONLY with a valid JSON array, no prose, no markdown:
[
  {{
    "target_node": "<exact node id from HSG — prefer semantic_* nodes>",
    "intent": "<one sentence describing exactly what visual change to make>",
    "influence": {{"ZT": 0.8, "ZL": 0.0, "ZB": 0.0}}
  }}
]
CRITICAL: The influence values MUST be valid floats (e.g., 0.0, 1.0). NO TEXT! NO letters! NEVER write "0s" or "0in"! ONLY NUMBERS."""


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_generate(model: str, prompt: str, images: list = None,
                   num_predict: int = 512) -> str:
    """Low-level call to /api/generate. Returns raw text or raises."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,  # CRITICAL: Unload LLM immediately to free RAM for SD
        "options": {"temperature": 0.05, "num_predict": num_predict}
    }
    if images:
        payload["images"] = images

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        timeout=180
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


    # ------------------------------------------------------------------
    # Step 2: phi3:mini maps HSG + prompt → Influence Tensors
    # (We skip step 1 llava description since it corrupts on this setup;
    #  the HSG itself already provides the semantic region context)
    # ------------------------------------------------------------------
    # Classify intent and build routing hint
    intent_class, strategy, routing_hint = route_prompt(user_prompt)
    print(f"[Router] Intent: {intent_class} | Default node: {strategy['node']}")

    # Inject routing hint as a comment INSIDE the prompt (before HSG)
    # Keeps the token count low and avoids confusing phi3:mini
    hsg_str_with_hint = (
        f"# ROUTING HINT: intent={intent_class}, prefer node='{strategy['node']}', "
        f"ZT={strategy['ZT']} ZL={strategy['ZL']} ZB={strategy['ZB']}\n"
        + hsg_str
    )

    parse_prompt = PARSE_PROMPT.format(
        hsg_str=hsg_str_with_hint,
        user_prompt=user_prompt
    )

    import time
    raw_text = None
    for attempt in range(4):
        try:
            raw_text = _call_generate(PARSER_MODEL, parse_prompt, num_predict=600)
            break
        except Exception as e:
            if attempt < 3:
                print(f"[LLM] Attempt {attempt+1} failed ({e}). Retrying in 15s...")
                time.sleep(15)
                continue
            raise e

    try:
        print(f"[LLM] phi3:mini response:\n{raw_text[:800]}")

        # Strip JS-style // comments
        import re
        clean_text = re.sub(r'//[^\n]*', '', raw_text)
        # Fix phi3 JSON quirks
        clean_text = clean_text.replace('0s', '0.0')
        clean_text = re.sub(r'00\.', '0.', clean_text)

        # Robustly extract the JSON array — cut off EXACTLY at the first valid ] 
        start = clean_text.find("[")
        if start != -1:
            # Walk forward to find the matching closing bracket
            depth = 0
            end = start
            for i, ch in enumerate(clean_text[start:]):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        end = start + i + 1
                        break
            json_str = clean_text[start:end]
            if not json_str.endswith(']'):
                json_str += "}]"

            edits = json.loads(json_str)
            if isinstance(edits, dict):
                edits = [edits]
            # Clamp influence values [0,1] and ensure colour/texture intent has ZT > 0
            for e in edits:
                inf  = e.get("influence", {})
                intent_lower = e.get("intent", "").lower()
                for k in ["ZT", "ZL", "ZB"]:
                    inf[k] = float(max(0.0, min(1.0, inf.get(k, 0.0))))
                # Auto-correct: if intent is about colour change but ZT==0, set to 0.9
                if any(w in intent_lower for w in ["color", "colour", "black", "dark", "texture", "material"]) and inf["ZT"] == 0.0:
                    inf["ZT"] = 0.9
                # Auto-correct: if intent is about sun/light/glow but ZB==0 and ZL==0, set ZL
                if any(w in intent_lower for w in ["sun", "glow", "bright", "light"]) and inf["ZL"] == 0.0:
                    inf["ZL"] = 0.8
                if any(w in intent_lower for w in ["sun", "add", "appear"]) and inf["ZB"] == 0.0:
                    inf["ZB"] = 0.8  # Sun is a structural addition
            return {"edits": edits, "llm_raw_response": raw_text}

        # Fallback: single object
        start = clean_text.find("{")
        end   = clean_text.rfind("}") + 1
        if start != -1 and end > 0:
            obj = json.loads(clean_text[start:end])
            inf = obj.get("influence", {})
            for k in ["ZT", "ZL", "ZB"]:
                inf[k] = float(max(0.0, min(1.0, inf.get(k, 0.0))))
            return {"edits": [obj], "llm_raw_response": raw_text}

        return {"error": "LLM returned no JSON", "raw": raw_text}

    except requests.exceptions.ConnectionError:
        return {"error": "Ollama not running. Run: ollama serve"}
    except requests.exceptions.Timeout:
        return {"error": "phi3:mini timed out (180s)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": raw_text}
    except Exception as e:
        return {"error": str(e)}


def check_ollama_health() -> bool:
    """Returns True if Ollama is running and phi3:mini is available."""
    try:
        resp   = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        found  = any("phi3" in m or "phi" in m for m in models)
        if not found:
            print(f"[LLM] phi3:mini not found. Run: ollama pull phi3:mini")
            print(f"[LLM] Available: {models}")
        return found
    except Exception:
        print("[LLM] Ollama not reachable. Run: ollama serve")
        return False
