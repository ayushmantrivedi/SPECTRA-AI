"""
POST /api/verify
Standalone verification: run the 5-check engine on any pair of images.
"""

from __future__ import annotations
import base64, io, logging
from fastapi import APIRouter, HTTPException
from PIL import Image

from api.schemas import VerifyRequest, VerifyResponse

router = APIRouter()
logger = logging.getLogger("spectra.api.verify")

_verification_engine = None


def _get_engine():
    global _verification_engine
    if _verification_engine is None:
        from spectra_core.verification_engine import VerificationEngine
        _verification_engine = VerificationEngine()
    return _verification_engine


def _decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


@router.post("/verify", response_model=VerifyResponse, summary="Run 5-check verification on image pair")
async def verify(request: VerifyRequest):
    """
    Run the full 5-check self-verification suite on an (original, edited) image pair.

    Checks:
      1. TARGET_ACCURACY        — Did the edit achieve the expected attribute change?
      2. FROZEN_REGION_INTEGRITY — Are non-edited regions preserved (SSIM > 0.96)?
      3. VISUAL_CONSISTENCY     — Lighting, colour harmony, texture, edge quality.
      4. ARTIFACT_DETECTION     — Halos, seams, noise, geometric distortion.
      5. INSTRUCTION_FULFILLMENT — Holistic VLM assessment.
    """
    try:
        original = _decode_image(request.original_image_base64)
        edited   = _decode_image(request.edited_image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    engine = _get_engine()
    try:
        report = engine.verify(
            original, edited,
            request.edit_plan,
            request.original_hsg,
        )
    except Exception as e:
        logger.exception("Verification engine failed")
        raise HTTPException(status_code=500, detail=str(e))

    return VerifyResponse(
        verification_result=report["verification_result"],
        overall_score=report["overall_score"],
        elapsed_ms=report.get("elapsed_ms", 0),
        checks=report["checks"],
        corrections_needed=report.get("corrections_needed"),
        next_action=report["next_action"],
    )
