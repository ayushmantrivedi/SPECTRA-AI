"""
SPECTRA LLM Parser — Orchestration via phi3:mini
================================================
Maps the Spectral Semantic Graph (SSG) and user prompt to a structured 
JSON Influence Tensor edit schedule.
"""

import json
import requests
import base64
from edit_router import route_prompt

OLLAMA_BASE   = "http://localhost:11434"
PARSER_MODEL  = "phi3:mini"
OLLAMA_MODEL  = PARSER_MODEL

PARSE_PROMPT = """
You are a SPECTRA Surgical Scene Analyst.
A Spectral Semantic Graph (SSG) JSON describing named regions of an image is below:
{ssg_str}

User Prompt: "{user_prompt}"

Plan exactly ONE or TWO anatomical edits. Prefer nodes like 'semantic_hair', 'semantic_person', 'semantic_clothing', 'semantic_upper body', 'semantic_sky'. 
If the user wants to keep a person's identity identical to the source, DO NOT edit 'semantic_face'.

Influence Tensors:
- ZT (Texture/Color): [0.0 - 1.0]
- ZL (Lighting): [0.0 - 1.0]
- ZB (Boundary/Structure): [0.0 - 1.0]

Output a JSON ARRAY of edit objects:
[{{
  "target_node": "node_id",
  "intent": "Brief description of change",
  "influence": {{"ZT": 0.0, "ZL": 0.0, "ZB": 0.0}}
}}]
Respond ONLY with the JSON array.
"""

def call_ollama_parser(image_b64: str, ssg_dict: dict, user_prompt: str) -> dict:
    """
    Parses a user instruction against the SSG tree using phi3:mini.
    Returns structured edit schedule.
    """
    # 1. Classify intent and build routing hint
    intent_class, strategy, _ = route_prompt(user_prompt)
    
    # 2. Package SSG for LLM context
    ssg_str = json.dumps(ssg_dict, indent=1)
    if len(ssg_str) > 4000:
        ssg_str = ssg_str[:4000] + "\n... (truncated)"

    prompt = PARSE_PROMPT.format(
        ssg_str=ssg_str,
        user_prompt=user_prompt
    )

    payload = {
        "model": PARSER_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0, # Unload immediately to save RAM for Diffusion
        "options": {"temperature": 0.05}
    }

    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "").strip()

        # Extract JSON array from LLM response
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if start != -1 and end > 0:
            edits = json.loads(raw_text[start:end])
            if isinstance(edits, dict):
                edits = [edits]
            return {"edits": edits, "llm_raw_response": raw_text}
        
        return {"error": "LLM failed to produce valid JSON schedule", "raw": raw_text}

    except Exception as e:
        return {"error": str(e)}

def check_ollama_health() -> bool:
    """Returns True if Ollama is running."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except:
        return False
