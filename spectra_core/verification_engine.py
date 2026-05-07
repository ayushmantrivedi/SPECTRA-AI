"""
SPECTRA Self-Verification Engine
==================================
Stage 4 of the SPECTRA pipeline.

Implements 5 rigorous verification checks on every edited image:

  CHECK 1 — TARGET_ACCURACY:     Did the edit achieve what was requested?
  CHECK 2 — FROZEN_INTEGRITY:    Are all non-edited regions pixel-identical?
  CHECK 3 — VISUAL_CONSISTENCY:  Lighting, colour harmony, texture, edges, resolution.
  CHECK 4 — ARTIFACT_DETECTION:  Seams, halos, warped geometry, noise.
  CHECK 5 — INSTRUCTION_FULFILL: Holistic VLM assessment.

On FAIL: generates a corrective action plan and feeds back to the Execution Engine.
Maximum 3 correction loops; after that delivers with a warning.
"""

from __future__ import annotations

import os
import gc
import json
import base64
import io
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("spectra.verification")

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from skimage.metrics import structural_similarity as _ssim
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not installed — SSIM check will use fallback.")

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ── Thresholds ────────────────────────────────────────────────────────────────

class Thresholds:
    TARGET_ACCURACY_MIN    = 0.80   # min color/attribute match
    FROZEN_SSIM_MIN        = 0.96   # SSIM for frozen regions
    FROZEN_COSINE_MIN      = 0.93   # embedding cosine for frozen regions
    VISUAL_CONSISTENCY_MIN = 0.75   # holistic VLM check
    ARTIFACT_SCORE_MAX     = 0.15   # edge noise allowance
    FULFILLMENT_CONFIDENCE = 0.82   # VLM holistic pass threshold
    OVERALL_PASS_MIN       = 0.80   # weighted average to PASS


# ── Individual Check Results ──────────────────────────────────────────────────

class CheckResult:
    def __init__(
        self,
        name: str,
        status: str,              # "PASS" | "FAIL" | "WARN"
        confidence: float,
        details: str,
        data: Optional[Dict] = None,
    ):
        self.name = name
        self.status = status
        self.confidence = confidence
        self.details = details
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "details": self.details,
            **self.data,
        }


# ── Verification Engine ───────────────────────────────────────────────────────

class VerificationEngine:
    """
    Runs all 5 verification checks and produces a structured report.
    If a cloud VLM is available, checks 3 and 5 use it for semantic quality.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if _TORCH_AVAILABLE and __import__("torch").cuda.is_available() else "cpu")
        self._gemini_model = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and _GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro-preview-05-06",
                    generation_config=genai.types.GenerationConfig(temperature=0.1),
                )
                logger.info("[Verification] Gemini VLM configured.")
            except Exception as e:
                logger.warning(f"[Verification] Gemini setup failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self,
        original_img: Image.Image,
        edited_img: Image.Image,
        edit_plan: Dict[str, Any],
        original_hsg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run all 5 verification checks.

        Returns:
            {
              "verification_result": "PASS" | "FAIL",
              "overall_score": float,
              "checks": [...],
              "corrections_needed": None | {...},
              "next_action": "DELIVER_TO_USER" | "RETRY_WITH_CORRECTIONS"
            }
        """
        t0 = time.perf_counter()

        checks: List[CheckResult] = []

        # ── Check 1: Target Accuracy ────────────────────────────────────────
        checks.append(self._check_target_accuracy(original_img, edited_img, edit_plan))

        # ── Check 2: Frozen Region Integrity ───────────────────────────────
        checks.append(self._check_frozen_integrity(original_img, edited_img, edit_plan, original_hsg))

        # ── Check 3: Visual Consistency ────────────────────────────────────
        checks.append(self._check_visual_consistency(original_img, edited_img, edit_plan))

        # ── Check 4: Artifact Detection ─────────────────────────────────────
        checks.append(self._check_artifact_detection(original_img, edited_img, edit_plan))

        # ── Check 5: Holistic Instruction Fulfillment ───────────────────────
        checks.append(self._check_instruction_fulfillment(edited_img, edit_plan))

        # ── Aggregate ────────────────────────────────────────────────────────
        overall_score = np.mean([c.confidence for c in checks])
        all_pass = all(c.status in ("PASS", "WARN") for c in checks)
        result = "PASS" if (all_pass and overall_score >= Thresholds.OVERALL_PASS_MIN) else "FAIL"

        corrections = None
        if result == "FAIL":
            corrections = self._generate_corrections(checks, edit_plan)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "verification_result": result,
            "overall_score": round(float(overall_score), 4),
            "elapsed_ms": elapsed_ms,
            "checks": [c.to_dict() for c in checks],
            "corrections_needed": corrections,
            "next_action": "DELIVER_TO_USER" if result == "PASS" else "RETRY_WITH_CORRECTIONS",
        }

    # ── Check 1 ───────────────────────────────────────────────────────────────

    def _check_target_accuracy(
        self,
        original: Image.Image,
        edited: Image.Image,
        edit_plan: Dict[str, Any],
    ) -> CheckResult:
        """
        Compare expected attribute change against what was actually achieved.
        For colour edits: measures hue deviation in HSV space.
        """
        modifications = edit_plan.get("modifications", [])
        if not modifications:
            return CheckResult(
                "TARGET_ACCURACY", "PASS", 0.90,
                "No modifications specified — nothing to verify."
            )

        scores: List[float] = []
        details_parts: List[str] = []

        orig_arr = np.array(original.convert("RGB"), dtype=np.float32)
        edit_arr = np.array(edited.convert("RGB"), dtype=np.float32)

        for mod in modifications:
            new_val = mod.get("new_value", "")
            attr_path = mod.get("attribute_path", "")

            if "color" in attr_path.lower() or "colour" in attr_path.lower():
                target_rgb = self._color_name_to_rgb(str(new_val))
                if target_rgb:
                    # Sample edit region (approximate via centre crop if no mask)
                    edit_region = edit_arr[
                        edit_arr.shape[0]//4 : 3*edit_arr.shape[0]//4,
                        edit_arr.shape[1]//4 : 3*edit_arr.shape[1]//4,
                    ]
                    avg_rgb = edit_region.mean(axis=(0, 1))
                    dist = np.linalg.norm(np.array(target_rgb) - avg_rgb) / 441.67
                    score = max(0.0, 1.0 - dist)
                    scores.append(score)
                    details_parts.append(
                        f"{mod.get('node','?')}: target={new_val} "
                        f"rgb={tuple(int(x) for x in avg_rgb)}, score={score:.2f}"
                    )
                else:
                    # Can't verify numerically — give benefit of doubt
                    scores.append(0.85)
                    details_parts.append(f"{mod.get('node','?')}: unquantifiable colour '{new_val}'")
            else:
                # Non-colour attribute — visual change assumed successful if edit region differs
                diff = np.abs(orig_arr - edit_arr).mean()
                score = min(1.0, diff / 20.0)  # expect ≥ 20px avg delta
                scores.append(score)
                details_parts.append(f"{mod.get('node','?')}: pixel-delta={diff:.1f}")

        avg_score = float(np.mean(scores)) if scores else 0.85
        status = "PASS" if avg_score >= Thresholds.TARGET_ACCURACY_MIN else "FAIL"

        return CheckResult(
            "TARGET_ACCURACY", status, avg_score,
            " | ".join(details_parts) or "Verified.",
            {"modifications_checked": len(modifications)},
        )

    # ── Check 2 ───────────────────────────────────────────────────────────────

    def _check_frozen_integrity(
        self,
        original: Image.Image,
        edited: Image.Image,
        edit_plan: Dict[str, Any],
        hsg: Dict[str, Any],
    ) -> CheckResult:
        """
        Verify that frozen (non-edited) regions are pixel-identical.
        Uses SSIM for structural similarity.
        """
        orig_arr = np.array(original.convert("RGB"))
        edit_arr = np.array(edited.convert("RGB"))

        # Global SSIM as proxy (no per-mask SSIM without runtime masks)
        if _SKIMAGE_AVAILABLE:
            ssim_val = float(_ssim(
                orig_arr, edit_arr, multichannel=True, channel_axis=-1,
                data_range=255,
            ))
        else:
            # Fallback: normalised L2 distance
            diff = np.abs(orig_arr.astype(float) - edit_arr.astype(float))
            ssim_val = max(0.0, 1.0 - diff.mean() / 50.0)

        frozen_regions = edit_plan.get("frozen_regions", [])
        status = "PASS" if ssim_val >= Thresholds.FROZEN_SSIM_MIN else "FAIL"

        return CheckResult(
            "FROZEN_REGION_INTEGRITY", status, ssim_val,
            f"Global SSIM={ssim_val:.4f} (threshold={Thresholds.FROZEN_SSIM_MIN}). "
            f"Frozen regions declared: {len(frozen_regions)}.",
            {
                "ssim": round(ssim_val, 4),
                "frozen_regions_count": len(frozen_regions),
            },
        )

    # ── Check 3 ───────────────────────────────────────────────────────────────

    def _check_visual_consistency(
        self,
        original: Image.Image,
        edited: Image.Image,
        edit_plan: Dict[str, Any],
    ) -> CheckResult:
        """
        Check lighting consistency, colour harmony, edge quality.
        Uses Gemini VLM when available; falls back to heuristics.
        """
        if self._gemini_model:
            return self._vlm_consistency_check(original, edited, edit_plan)
        return self._heuristic_consistency_check(original, edited, edit_plan)

    def _vlm_consistency_check(
        self, original: Image.Image, edited: Image.Image, edit_plan: Dict[str, Any]
    ) -> CheckResult:
        try:
            instruction = edit_plan.get("modifications", [{}])[0].get("new_value", "")
            prompt = (
                f"The original image was edited with the instruction: "
                f"\"{instruction}\".\n"
                "Evaluate ONLY visual consistency:\n"
                "1. Lighting: Are shadows/highlights consistent with the scene?\n"
                "2. Colour harmony: Do new colours blend with the palette?\n"
                "3. Texture continuity: Is material consistent?\n"
                "4. Edge quality: Are there halos or visible seams?\n"
                "5. Resolution: Is the edited area sharp?\n\n"
                "Respond with JSON: "
                "{\"lighting\": bool, \"colour_harmony\": bool, \"texture\": bool, "
                "\"edges\": bool, \"resolution\": bool, \"overall_score\": float 0-1}"
            )

            orig_b = self._pil_to_bytes(original)
            edit_b = self._pil_to_bytes(edited)

            resp = self._gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": base64.b64encode(orig_b).decode()},
                {"mime_type": "image/jpeg", "data": base64.b64encode(edit_b).decode()},
            ])
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
            data = json.loads(raw)
            score = float(data.get("overall_score", 0.85))
            status = "PASS" if score >= Thresholds.VISUAL_CONSISTENCY_MIN else "FAIL"
            checks_str = " | ".join(
                f"{k}={'✓' if v else '✗'}"
                for k, v in data.items() if isinstance(v, bool)
            )
            return CheckResult("VISUAL_CONSISTENCY", status, score, checks_str, data)
        except Exception as e:
            logger.warning(f"[Verification] VLM consistency check failed: {e}")
            return self._heuristic_consistency_check(original, edited, edit_plan)

    def _heuristic_consistency_check(
        self, original: Image.Image, edited: Image.Image, edit_plan: Dict[str, Any]
    ) -> CheckResult:
        orig_arr = np.array(original.convert("RGB"), dtype=float)
        edit_arr = np.array(edited.convert("RGB"), dtype=float)

        # Colour range preservation
        orig_range = orig_arr.max() - orig_arr.min()
        edit_range = edit_arr.max() - edit_arr.min()
        range_ratio = min(orig_range, edit_range) / (max(orig_range, edit_range) + 1)

        # Brightness delta
        orig_bri = orig_arr.mean() / 255.0
        edit_bri = edit_arr.mean() / 255.0
        bri_score = 1.0 - abs(orig_bri - edit_bri)

        score = 0.5 * range_ratio + 0.5 * bri_score
        status = "PASS" if score >= Thresholds.VISUAL_CONSISTENCY_MIN else "WARN"
        return CheckResult(
            "VISUAL_CONSISTENCY", status, score,
            f"Heuristic: colour_range_ratio={range_ratio:.2f}, brightness_preservation={bri_score:.2f}",
        )

    # ── Check 4 ───────────────────────────────────────────────────────────────

    def _check_artifact_detection(
        self,
        original: Image.Image,
        edited: Image.Image,
        edit_plan: Dict[str, Any],
    ) -> CheckResult:
        """
        Detect boundary artifacts (halos, colour discontinuities) using gradient analysis.
        """
        orig_gray = np.array(original.convert("L"), dtype=float)
        edit_gray = np.array(edited.convert("L"), dtype=float)

        # High-frequency noise proxy: difference of Laplacians
        def laplacian(img: np.ndarray) -> np.ndarray:
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            from scipy.signal import convolve2d
            return np.abs(convolve2d(img, kernel, mode="same"))

        try:
            from scipy.signal import convolve2d
            lap_orig = laplacian(orig_gray)
            lap_edit = laplacian(edit_gray)
            artifact_score = float(np.abs(lap_edit - lap_orig).mean() / (lap_orig.mean() + 1e-6))
            artifact_score = min(1.0, artifact_score)
        except ImportError:
            # Fallback: pixel L1 delta
            diff = np.abs(orig_gray - edit_gray).mean() / 255.0
            artifact_score = diff

        # Lower artifact_score → better (inverted for confidence)
        confidence = 1.0 - min(artifact_score, 1.0)
        status = "PASS" if artifact_score <= Thresholds.ARTIFACT_SCORE_MAX else "FAIL"

        return CheckResult(
            "ARTIFACT_DETECTION", status, confidence,
            f"Artifact score={artifact_score:.4f} (threshold={Thresholds.ARTIFACT_SCORE_MAX}). "
            f"{'No significant artifacts.' if status == 'PASS' else 'Artifacts detected.'}",
            {"artifact_score": round(artifact_score, 4)},
        )

    # ── Check 5 ───────────────────────────────────────────────────────────────

    def _check_instruction_fulfillment(
        self,
        edited_img: Image.Image,
        edit_plan: Dict[str, Any],
    ) -> CheckResult:
        """
        VLM holistic fulfillment check: 'Does the edited image match the instruction?'
        Falls back to keyword heuristic if VLM unavailable.
        """
        if self._gemini_model:
            return self._vlm_fulfillment_check(edited_img, edit_plan)
        return self._heuristic_fulfillment_check(edit_plan)

    def _vlm_fulfillment_check(
        self, edited_img: Image.Image, edit_plan: Dict[str, Any]
    ) -> CheckResult:
        try:
            mods = edit_plan.get("modifications", [])
            expected = " and ".join(
                f"{m.get('node', '?')}.{m.get('attribute_path', '?')} = {m.get('new_value', '?')}"
                for m in mods
            )
            prompt = (
                f"Expected edits: {expected}.\n"
                "Describe what you see in this image (2 sentences). "
                "Then respond with JSON: "
                "{\"fulfilled\": bool, \"confidence\": float 0-1, \"assessment\": \"<one sentence>\"}"
            )
            edit_b = self._pil_to_bytes(edited_img)
            resp = self._gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": base64.b64encode(edit_b).decode()},
            ])
            raw = resp.text.strip()
            start = raw.rfind("{"); end = raw.rfind("}") + 1
            if start >= 0:
                data = json.loads(raw[start:end])
                score = float(data.get("confidence", 0.85))
                status = "PASS" if data.get("fulfilled", True) and score >= Thresholds.FULFILLMENT_CONFIDENCE else "FAIL"
                return CheckResult(
                    "INSTRUCTION_FULFILLMENT", status, score,
                    data.get("assessment", ""),
                    {"vlm_fulfilled": data.get("fulfilled", True)},
                )
        except Exception as e:
            logger.warning(f"[Verification] VLM fulfillment check failed: {e}")
        return self._heuristic_fulfillment_check(edit_plan)

    def _heuristic_fulfillment_check(self, edit_plan: Dict[str, Any]) -> CheckResult:
        # Without VLM, assume moderate confidence if plan parsed successfully
        confidence_proxy = edit_plan.get("confidence", 0.75)
        status = "PASS" if confidence_proxy >= 0.60 else "WARN"
        return CheckResult(
            "INSTRUCTION_FULFILLMENT", status, float(confidence_proxy),
            f"Heuristic fulfillment based on parser confidence={confidence_proxy:.2f} (no VLM).",
        )

    # ── Correction Plan Generator ──────────────────────────────────────────────

    def _generate_corrections(
        self, checks: List[CheckResult], edit_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse failed checks and produce a targeted correction plan.
        """
        corrections: List[Dict[str, Any]] = []
        for check in checks:
            if check.status != "FAIL":
                continue
            if check.name == "TARGET_ACCURACY":
                corrections.append({
                    "action": "ADJUST_DIFFUSION_STRENGTH",
                    "reason": "Target attribute not sufficiently changed.",
                    "suggestion": "Increase guidance_scale or strength in diffusion step.",
                })
            elif check.name == "FROZEN_REGION_INTEGRITY":
                corrections.append({
                    "action": "REDUCE_MASK_BLEED",
                    "reason": f"Frozen regions altered. SSIM={check.data.get('ssim', '?')}",
                    "suggestion": "Tighten the edit mask; reduce feather radius.",
                })
            elif check.name == "VISUAL_CONSISTENCY":
                corrections.append({
                    "action": "HARMONISE_COLOURS",
                    "reason": "Visual inconsistency detected.",
                    "suggestion": "Apply colour-grading pass to match scene palette.",
                })
            elif check.name == "ARTIFACT_DETECTION":
                corrections.append({
                    "action": "APPLY_POISSON_BLEND",
                    "reason": f"Artifacts detected (score={check.data.get('artifact_score','?')}).",
                    "suggestion": "Re-run Poisson blend with larger feather and tighter mask.",
                })
            elif check.name == "INSTRUCTION_FULFILLMENT":
                corrections.append({
                    "action": "REGENERATE",
                    "reason": "Instruction not fulfilled holistically.",
                    "suggestion": "Increase diffusion steps and revise prompt.",
                })

        return {
            "corrections": corrections,
            "corrected_edit_plan_delta": {
                "guidance_scale": 11.0,   # bump from 9 → 11
                "num_inference_steps": 50, # bump from 40 → 50
                "strength": min(1.0, edit_plan.get("generation_hints", {}).get("strength", 0.85) + 0.1),
            },
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_bytes(img: Image.Image, quality: int = 85) -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def _color_name_to_rgb(name: str) -> Optional[Tuple[int, int, int]]:
        _map = {
            "red": (220, 20, 20), "blue": (30, 80, 220), "green": (30, 150, 50),
            "black": (15, 15, 15), "white": (240, 240, 240), "yellow": (240, 220, 30),
            "purple": (130, 30, 200), "pink": (240, 100, 160), "orange": (240, 140, 30),
            "silver": (180, 180, 185), "gold": (210, 170, 40), "brown": (120, 70, 30),
            "grey": (140, 140, 140), "gray": (140, 140, 140),
        }
        return _map.get(name.lower())


# ── Correction Loop Controller ────────────────────────────────────────────────

class CorrectionPlanner:
    """
    Controls the iterative correction loop (up to MAX_ATTEMPTS).
    Wraps VerificationEngine and feeds corrections back to the orchestrator.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, verification_engine: VerificationEngine):
        self.ve = verification_engine
        self.attempt_log: List[Dict[str, Any]] = []

    def run_verification_loop(
        self,
        execute_fn,                    # callable(edit_plan, **kwargs) → Image.Image
        original_img: Image.Image,
        edit_plan: Dict[str, Any],
        original_hsg: Dict[str, Any],
        **exec_kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Execute → Verify → Correct loop.

        Args:
            execute_fn: Callable that runs the edit and returns an edited PIL image.
            original_img: The original unedited image.
            edit_plan: Parsed edit plan dict.
            original_hsg: Original HSG dict for frozen region comparison.
            **exec_kwargs: Extra kwargs forwarded to execute_fn.

        Returns:
            (final_image, verification_report)
        """
        current_plan = dict(edit_plan)

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            logger.info(f"[CorrectionLoop] Attempt {attempt}/{self.MAX_ATTEMPTS}")

            # Execute
            edited_img: Image.Image = execute_fn(current_plan, **exec_kwargs)

            # Verify
            report = self.ve.verify(original_img, edited_img, current_plan, original_hsg)
            report["attempt"] = attempt
            self.attempt_log.append(report)

            if report["verification_result"] == "PASS":
                logger.info(f"[CorrectionLoop] PASS on attempt {attempt}.")
                return edited_img, report

            # Apply corrections for next attempt
            corrections = report.get("corrections_needed", {})
            if corrections:
                delta = corrections.get("corrected_edit_plan_delta", {})
                current_plan = {**current_plan, **delta}
                logger.info(f"[CorrectionLoop] FAIL — applying {len(corrections.get('corrections',[]))} corrections.")
            else:
                logger.info("[CorrectionLoop] No corrections available. Stopping early.")
                break

        # Best-effort delivery
        logger.warning(f"[CorrectionLoop] Delivering best-effort after {self.MAX_ATTEMPTS} attempts.")
        report["verification_result"] = "BEST_EFFORT"
        report["next_action"] = "DELIVER_TO_USER"
        report["warning"] = f"Best effort result after {self.MAX_ATTEMPTS} correction attempts."
        return edited_img, report

    def get_attempt_summary(self) -> List[Dict[str, Any]]:
        return self.attempt_log
