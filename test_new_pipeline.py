import argparse
import sys
import os
import json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_core.orchestrator_loop import OrchestratorLoop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="real_woman.jpg")
    ap.add_argument("--prompt", type=str, default="Make her hair red")
    ap.add_argument("--out", type=str, default="result_woman_red.png")
    args = ap.parse_args()

    print(f"Loading image: {args.image}")
    img_pil = Image.open(args.image).convert("RGB")
    
    # Resize down if it's too large to make local tests faster
    if max(img_pil.size) > 1024:
        img_pil.thumbnail((1024, 1024))

    print("Initializing OrchestratorLoop...")
    loop = OrchestratorLoop()

    print(f"Running edit for prompt: '{args.prompt}'")
    result = loop.run(img_pil, args.prompt)

    out_path = args.out
    print(f"Saving edited image to: {out_path}")
    
    result_img = result["edited_image"]
    result_img.save(out_path)

    print("\n--- AUDIT LOG ---")
    print(json.dumps(result["audit"], indent=2))
    
    print("\n--- VERIFICATION REPORT ---")
    print(json.dumps(result["verification_report"], indent=2))

if __name__ == "__main__":
    main()
