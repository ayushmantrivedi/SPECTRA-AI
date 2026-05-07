"""
POST /api/edit
Full 5-stage SPECTRA pipeline: Analyze → Parse → Execute → Verify → Update HSG.
"""

from __future__ import annotations
import base64, io, logging
from fastapi import APIRouter, HTTPException
from PIL import Image

from api.schemas import EditRequest, EditResponse

router = APIRouter()
logger = logging.getLogger("spectra.api.edit")

# Singleton pipeline instance (initialised on first request)
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from spectra_core.orchestrator_loop import OrchestratorLoop
        _pipeline = OrchestratorLoop()
    return _pipeline


def _decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post("/edit", response_model=EditResponse, summary="Run full 5-stage SPECTRA edit pipeline")
async def edit(request: EditRequest):
    """
    End-to-end surgical edit.

    - Stage 1: VLM scene analysis → Huffman Scene Graph
    - Stage 2: NLP parser → structured Edit Plan
    - Stage 3: Kernel Diffusion → localized pixel edit
    - Stage 4: Self-verification → 5-check report
    - Stage 5: Adaptive HSG rebalancing
    """
    try:
        image_pil = _decode_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    pipeline = _get_pipeline()

    try:
        result = pipeline.run(
            image_pil,
            request.instruction,
            force_reanalyze=request.force_reanalyze,
        )
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(e))

    edited_img = result["edited_image"]
    if not isinstance(edited_img, Image.Image):
        # Tensor fallback
        import torch
        from torchvision.transforms.functional import to_pil_image
        if torch.is_tensor(edited_img):
            edited_img = to_pil_image(edited_img[0].clamp(0, 1).cpu())

    return EditResponse(
        status=result.get("status", "ok"),
        edit_id=result.get("edit_id", ""),
        result_image_b64=_encode_image(edited_img),
        hsg=result.get("hsg", {}),
        verification_report=result.get("verification_report", {}),
        edit_plans=result.get("edit_plans", []),
        audit=result.get("audit", {}),
    )
