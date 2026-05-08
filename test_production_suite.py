"""
SPECTRA Production Test Suite
==============================
Runs a battery of real-world edit tests across multiple images,
logs per-stage latency, verification scores, and saves all outputs.

Usage:
    python test_production_suite.py

Output:
    - results/  (all edited images)
    - production_test_report.json  (full structured report)
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectra_core.orchestrator_loop import OrchestratorLoop

# --- OUTPUT DIRECTORY ---------------------------------------------------------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- TEST CASES ---------------------------------------------------------------
# Real-world surgical edit prompts tailored to each image's actual content.
TEST_CASES = [
    # -- LANDSCAPE: Wooden dock, alpine lake, mountains, forest, cloudy sky --
    {
        "image": "real_landscape.jpg",
        "id": "landscape_001",
        "prompt": "Make the sky stormy and dramatic with dark clouds",
        "description": "Atmospheric sky replacement - tests background cascade editing",
        "expected_changes": ["sky", "background"],
        "frozen_critical": ["dock", "lake", "mountains"],
    },
    {
        "image": "real_landscape.jpg",
        "id": "landscape_002",
        "prompt": "Change the lake water color to a deep emerald green",
        "description": "Targeted water color edit - tests precise region modification",
        "expected_changes": ["lake", "water"],
        "frozen_critical": ["dock", "mountains", "forest"],
    },
    {
        "image": "real_landscape.jpg",
        "id": "landscape_003",
        "prompt": "Make it golden hour - warm orange sunset lighting across the whole scene",
        "description": "Global lighting cascade edit - tests multi-node warming",
        "expected_changes": ["sky", "lighting", "reflections"],
        "frozen_critical": ["dock_structure", "mountain_silhouette"],
    },
    {
        "image": "real_landscape.jpg",
        "id": "landscape_004",
        "prompt": "Add a thick morning fog over the lake surface",
        "description": "Structural addition - tests add-region diffusion",
        "expected_changes": ["lake_surface"],
        "frozen_critical": ["dock", "trees", "mountains"],
    },

    # -- MOUNTAIN SKY: Snow peaks, sea of clouds, sunset sky, rocky foreground --
    {
        "image": "real_mountainsky.jpg",
        "id": "mountainsky_001",
        "prompt": "Change the sunset sky to a vivid purple and pink aurora borealis",
        "description": "Sky color transform - tests radical hue shift on large region",
        "expected_changes": ["sky", "clouds"],
        "frozen_critical": ["mountain_peaks", "rocky_foreground"],
    },
    {
        "image": "real_mountainsky.jpg",
        "id": "mountainsky_002",
        "prompt": "Make the mountain peaks glowing with bright white snow in moonlight",
        "description": "Targeted highlight boost - tests peak texture preservation",
        "expected_changes": ["mountain_peaks", "snow"],
        "frozen_critical": ["sky", "clouds", "foreground"],
    },
    {
        "image": "real_mountainsky.jpg",
        "id": "mountainsky_003",
        "prompt": "Turn the sea of clouds below the peaks into a fiery lava field",
        "description": "Extreme texture replacement - tests mid-region diffusion",
        "expected_changes": ["clouds", "valley"],
        "frozen_critical": ["mountain_peaks", "sky"],
    },
    {
        "image": "real_mountainsky.jpg",
        "id": "mountainsky_004",
        "prompt": "Add a full moon in the sky above the mountain peaks",
        "description": "Structural addition to sky region - tests inpainting with new object",
        "expected_changes": ["sky_upper"],
        "frozen_critical": ["mountain_peaks", "clouds", "foreground"],
    },
]

# --- METRICS TRACKING ---------------------------------------------------------
def compute_quality_metrics(test_result: dict) -> dict:
    """Derive final quality metrics from stage outputs."""
    audit = test_result.get("audit", {})
    verification = test_result.get("verification_report", {})
    stages = audit.get("stages", [])

    stage_map = {s["stage"]: s for s in stages}

    stage3 = stage_map.get("STAGE_3_EXECUTION", {})
    stage4 = stage_map.get("STAGE_4_VERIFICATION", {})

    return {
        "overall_score": verification.get("overall_score", 0.0),
        "verification_result": verification.get("verification_result", "UNKNOWN"),
        "checks_passed": stage4.get("checks_passed", 0),
        "checks_total": stage4.get("checks_total", 5),
        "total_time_s": audit.get("total_time_s", 0.0),
        "stage3_status": stage3.get("status", "N/A"),
        "execution_error": stage3.get("error"),
    }


def run_single_test(loop: OrchestratorLoop, test: dict) -> dict:
    """Run one test case and return a fully structured result dict."""
    print(f"\n{'='*70}")
    print(f"  TEST: [{test['id']}]")
    print(f"  Image:  {test['image']}")
    print(f"  Prompt: \"{test['prompt']}\"")
    print(f"  Goal:   {test['description']}")
    print(f"{'='*70}")

    status = "PASS"
    result = {}
    error_msg = None
    start = time.time()

    try:
        img_pil = Image.open(test["image"]).convert("RGB")

        # Normalise to 1024px for speed; production would use full-res
        if max(img_pil.size) > 1024:
            img_pil.thumbnail((1024, 1024), Image.LANCZOS)

        w, h = img_pil.size
        print(f"  Input size: {w}x{h}px")

        result = loop.run(img_pil, test["prompt"])

        # Save output image
        out_name = f"{test['id']}_result.png"
        out_path = RESULTS_DIR / out_name
        result["edited_image"].save(out_path)
        print(f"  [SAVED] -> {out_path}")

        # Compute quality metrics
        metrics = compute_quality_metrics(result)

        if metrics["verification_result"] == "FAIL":
            status = "FAIL"

        return {
            "id": test["id"],
            "image": test["image"],
            "prompt": test["prompt"],
            "description": test["description"],
            "status": status,
            "output_path": str(out_path),
            "wall_time_s": round(time.time() - start, 2),
            "metrics": metrics,
            "audit": result.get("audit", {}),
            "verification_report": result.get("verification_report", {}),
        }

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"  [EXCEPTION] {error_msg}")
        print(tb)
        return {
            "id": test["id"],
            "image": test["image"],
            "prompt": test["prompt"],
            "description": test["description"],
            "status": "ERROR",
            "output_path": None,
            "wall_time_s": round(time.time() - start, 2),
            "error": error_msg,
            "traceback": tb,
            "metrics": {},
            "audit": {},
            "verification_report": {},
        }


def print_summary(results: list):
    """Print a compact production-style summary table."""
    print(f"\n\n{'='*70}")
    print("  SPECTRA PRODUCTION TEST SUITE --- FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'TEST ID':<25} {'STATUS':<8} {'SCORE':<8} {'TIME':>8}  STAGE3")
    print(f"  {'-'*60}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    for r in results:
        m = r.get("metrics", {})
        score = m.get("overall_score", 0.0)
        s3 = m.get("stage3_status", "N/A")
        s3_err = f" | ERR: {m.get('execution_error', '')[:30]}" if m.get("execution_error") else ""
        status_tag = "[PASS]" if r["status"] == "PASS" else ("[FAIL]" if r["status"] == "FAIL" else "[ERR] ")
        print(f"  {status_tag} {r['id']:<23} {r['status']:<8} {score:<8.3f} {r['wall_time_s']:>7.1f}s  {s3}{s3_err}")

    print(f"  {'-'*60}")
    print(f"  Total: {total} | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")
    print(f"  Pass Rate: {(passed/total)*100:.1f}%")
    print(f"{'='*70}\n")


def main():
    print("\n[SPECTRA] Production Test Suite Starting...")
    print(f"   Running {len(TEST_CASES)} tests across 2 images\n")

    # Initialise the orchestrator loop ONCE (production singleton pattern)
    print("Initialising OrchestratorLoop (singleton)...")
    loop = OrchestratorLoop()

    suite_start = time.time()
    results = []

    for test in TEST_CASES:
        result = run_single_test(loop, test)
        results.append(result)

        # Brief pause between tests for memory hygiene
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_suite_time = round(time.time() - suite_start, 2)

    print_summary(results)

    # Write full structured report
    report = {
        "suite": "SPECTRA Production Test Suite v1.0",
        "total_tests": len(results),
        "total_time_s": total_suite_time,
        "pass_rate": round(
            sum(1 for r in results if r["status"] == "PASS") / len(results) * 100, 1
        ),
        "results": results,
    }

    report_path = "production_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[REPORT] Full report saved -> {report_path}")
    print(f"[OUTPUT] All output images saved -> {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
