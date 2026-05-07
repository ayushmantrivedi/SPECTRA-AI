"""
POST /api/analyze
Runs Stage 1 (VLM Orchestrator) and returns the Huffman Scene Graph.
"""

from __future__ import annotations
import base64, io, logging
from fastapi import APIRouter, HTTPException
from PIL import Image

from api.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()
logger = logging.getLogger("spectra.api.analyze")


def _decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze image → build HSG")
async def analyze(request: AnalyzeRequest):
    """
    Stage 1 only: analyze the image and return the Huffman Scene Graph.
    Use this to inspect what SPECTRA sees before committing to an edit.
    """
    try:
        image_pil = _decode_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    from spectra_core.orchestrator_vlm import OrchestratorVLM
    vlm = OrchestratorVLM()

    try:
        result = vlm.analyze(image_pil)
    except Exception as e:
        logger.exception("Orchestrator VLM failed")
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        status="ok",
        hsg=result["hsg"],
        metadata=result.get("metadata", {}),
        processing_time_ms=result.get("metadata", {}).get("processing_time_ms", 0),
    )
