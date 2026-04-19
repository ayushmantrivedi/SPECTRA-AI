"""
SPECTRA End-to-End Test Engine
===============================
Execution path: Image -> SSG Extraction -> phi3:mini Parsing -> Orchestration
"""

import argparse
import base64
import io
import json
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from main_model import ImageGenerationModel, device
from llm_parser import call_ollama_parser, check_ollama_health, OLLAMA_MODEL
from spectral_sync import SpectralSyncTracker
from dynamic_orchestrator import DynamicOrchestrator, _flatten_ssg

def load_image(path: str):
    pil = Image.open(path).convert("RGB").resize((512, 512))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return pil, b64

def print_ssg_tree(node, indent=0):
    pad = "  " * indent
    print(f"{pad}[D{node.get('depth', 0)}] {node['id']} | weight={node.get('weight', 0):.3f}")
    for c in node.get("children", []):
        print_ssg_tree(c, indent + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="real_woman.jpg")
    ap.add_argument("--prompt", type=str, default="Make her hair silver")
    ap.add_argument("--out", type=str, default="spectra_result.png")
    args = ap.parse_args()

    # 1. Init
    img_pil, img_b64 = load_image(args.image)
    model = ImageGenerationModel().to(device)
    tracker = SpectralSyncTracker()
    orchestrator = DynamicOrchestrator(model, tracker)

    # 2. Extract
    print("[+] Extracting Spectral Semantic Graph...")
    ssg_dict, masks_cache = model.extract_ssg(img_pil)
    print_ssg_tree(ssg_dict)

    # 3. Parse
    if not check_ollama_health():
        print("[!] Ollama not running. Using fallback.")
        edits = [{"target_node": "semantic_hair", "intent": "silver hair", "influence": {"ZT": 1.0, "ZL": 0, "ZB": 0}}]
    else:
        print("[+] Parsing intent via SPECTRA LLM...")
        res = call_ollama_parser(img_b64, ssg_dict, args.prompt)
        if "error" in res:
            print(f"[!] LLM Error: {res['error']}. Using fallback.")
            edits = [{"target_node": "semantic_hair", "intent": "silver hair", "influence": {"ZT": 1.0, "ZL": 0, "ZB": 0}}]
        else:
            edits = res["edits"]

    # 4. Execute
    print(f"[+] Executing {len(edits)} hops...")
    result_img, final_ssg, logs = orchestrator.execute_edit_schedule(img_pil, ssg_dict, edits, masks_cache)

    # 5. Save
    result_img.save(args.out)
    print(f"[SUCCESS] Result saved to {args.out}")

if __name__ == "__main__":
    main()
