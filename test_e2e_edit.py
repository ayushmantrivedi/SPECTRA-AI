"""
End-to-End Image Editing Pipeline
===================================
Full production pipeline:
  Image → HSGBuilder → Ollama (llava:7b) → DynamicOrchestrator → Result

Usage:
  python test_e2e_edit.py --image real_woman.jpg --prompt "change the women color to black and make the sun appear in sky"
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
from dynamic_orchestrator import DynamicOrchestrator, _flatten_hsg

IMG_SIZE = 512


# ── Helpers ────────────────────────────────────────────────────────────────

def load_image(path: str):
    """Load & preprocess. Returns (tensor [1,3,H,W], pil, base64_str)."""
    pil  = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(pil, dtype=np.float32) / 255.0
    t    = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    buf  = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    b64  = base64.b64encode(buf.getvalue()).decode("utf-8")
    return t, pil, b64


def tensor_to_pil(t):
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_side_by_side(orig_pil, edited_tensor, prompt, exec_log, out_path):
    # Scale up both images to 1024px width for a premium look
    DISPLAY_W = 1024
    DISPLAY_H = 512
    
    edited_pil = tensor_to_pil(edited_tensor).resize((DISPLAY_H, DISPLAY_H), Image.LANCZOS)
    orig_resized = orig_pil.resize((DISPLAY_H, DISPLAY_H), Image.LANCZOS)

    canvas_w = DISPLAY_W + 40
    canvas_h = DISPLAY_H + 120
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 15)) # Darker premium bg

    # Layout: [ [Original]  [Edited] ]
    canvas.paste(orig_resized, (10, 60))
    canvas.paste(edited_pil, (DISPLAY_H + 30, 60))

    draw = ImageDraw.Draw(canvas)
    # Premium typography simulation
    draw.text((20,  20), "SOURCE ASSET", fill=(180, 180, 180))
    draw.text((DISPLAY_H + 40, 20), "AI ENHANCED", fill=(0, 255, 128))
    draw.text((20, canvas_h - 40), f'Intent: "{prompt}"', fill=(150, 150, 255))
    canvas.save(out_path)
    print(f"[+] Saved -> {out_path}")


def print_tree(node, indent=0):
    pad   = "  " * indent
    light = node.get("attributes", {}).get(
        "light_intensity",
        node.get("attributes", {}).get("macro_light_intensity", "?")
    )
    sz = node.get("attributes", {}).get("size", "?")
    loc = node.get("attributes", {}).get("location", "?")
    print(f"{pad}[D{node.get('depth',0)}] {node['id']} | "
          f"weight={node.get('weight',0):.3f} | sz={sz} | loc={loc} | bbox={node.get('bbox',[])} | light={light}")
    for c in node.get("children", []):
        print_tree(c, indent + 1)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Lumina IKLD — full pipeline")
    ap.add_argument("--image",  type=str, default=None)
    ap.add_argument("--prompt", type=str,
                    default="Make the brightest region darker and add a glowing effect")
    ap.add_argument("--out",    type=str, default="e2e_result.png")
    args = ap.parse_args()

    # ── 1. Resolve image ─────────────────────────────────────────────────
    if args.image and Path(args.image).exists():
        image_path = args.image
        print(f"[+] Image: {image_path}")
    else:
        print("[!] No image. Generating synthetic test image.")
        import cv2
        img = np.ones((128, 128, 3), dtype=np.uint8) * 40
        cv2.rectangle(img, (15, 15), (80, 80), (220, 180, 50), -1)
        cv2.circle(img,   (100, 100), 15,       (80, 200, 255),  -1)
        image_path = "e2e_input.png"
        cv2.imwrite(image_path, img[:, :, ::-1])

    img_tensor, orig_pil, img_b64 = load_image(image_path)

    # ── 2. Model & SSG ───────────────────────────────────────────────────
    print("[+] Initialising model …")
    model = ImageGenerationModel().to(device)
    model.eval()

    # 2. Extract Stage: Initial SSG
    tracker = SpectralSyncTracker(decay_lambda=0.85)
    print("[+] Extracting Spectral Semantic Graph ...")
    ssg_adapted, masks_cache = model.extract_ssg(orig_pil)
    
    print("\n--- Spectral Semantic Graph -------------------------------------------")
    _print_ssg_tree(ssg_adapted)
    nodes = _flatten_ssg(ssg_adapted)
    print(f"\nTotal nodes: {len(nodes)} | "
          f"Leaf nodes: {sum(1 for n in nodes if not n.get('children'))}")

    # ── 3. LLM Parsing ───────────────────────────────────────────────────
    print(f"\n--- LLM Parser ({OLLAMA_MODEL}) -------------------------------------------")
    print(f"Prompt: \"{args.prompt}\"")

    import gc
    import time

    # Smart GC: reclaim memory, poll until GPU headroom is enough (max 8s)
    print("[+] Reclaiming memory for LLM call...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for _ in range(8):
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            if allocated_mb < 500:
                break
            time.sleep(1)

    if not check_ollama_health():
        print(f"[ERROR] Ollama / {OLLAMA_MODEL} not available. Start: ollama serve && ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    try:
        edit_plan = call_ollama_parser(img_b64, ssg_adapted, args.prompt)
        # Reclaim RAM: Force Ollama to unload the model immediately after parsing
        import requests
        try:
            requests.post("http://localhost:11434/api/generate", 
                          json={"model": OLLAMA_MODEL, "keep_alive": 0}, 
                          timeout=5)
            print("[TEST] Reclaimed RAM from Ollama.")
        except:
            pass
            
        if "error" in edit_plan:
             raise ValueError(edit_plan["error"])
        edits = edit_plan.get("edits", [])
    except Exception as e:
        print(f"[TEST] Ollama failed or timed out ({e}). Using Hardcoded Fallback Schedule...")
        edits = [
            {
                "target_node": "semantic_hair",
                "intent": "Change the woman's hair color to silver.",
                "influence": {"ZT": 0.8, "ZL": 0.0, "ZB": 0.0}
            },
            {
                "target_node": "semantic_person",
                "intent": "Change the woman's suit color to red.",
                "influence": {"ZT": 0.8, "ZL": 0.0, "ZB": 0.0}
            }
        ]
    print(f"\n--- Edit Schedule ({len(edits)} hops) --------------------------------------")
    for e in edits:
        print(f"  Node: {e['target_node']} | {e.get('intent','?')} "
              f"| ZT={e['influence']['ZT']} ZL={e['influence']['ZL']} ZB={e['influence']['ZB']}")

    # ── 4. Dynamic Orchestrator ─────────────────────────────────────────
    print("\n--- Executing via Dynamic DP Orchestrator -------------------------")
    orchestrator = DynamicOrchestrator(model, tracker)
    result_img, final_ssg, exec_log = orchestrator.execute_edit_schedule(
        orig_pil, ssg_adapted, edits, initial_masks_cache=masks_cache
    )

    print("\n--- Execution Log -------------------------------------------------")
    for log in exec_log:
        print(f"  {log}")

    # ── 5. Save ─────────────────────────────────────────────────────────
    save_side_by_side(orig_pil, result_img, args.prompt, exec_log, args.out)

    report = {
        "prompt":        args.prompt,
        "edit_plan":     edits,
        "execution_log": exec_log,
        "node_count":    len(_flatten_hsg(final_hsg)),
        "tracker_stats": tracker.get_stats()
    }
    with open("e2e_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE]  result -> {args.out}  |  report -> e2e_report.json")


if __name__ == "__main__":
    main()
