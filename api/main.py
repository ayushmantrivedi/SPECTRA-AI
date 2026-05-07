"""
SPECTRA FastAPI Server — Production Entry Point
================================================
Replaces the legacy app.py with the full 5-stage pipeline.

Endpoints:
  GET  /                       Health check
  GET  /api/hsg/stats          HSG session statistics
  POST /api/analyze            Stage 1: image → HSG
  POST /api/edit               Stages 1-5: full surgical edit
  POST /api/verify             Stage 4 standalone: 5-check verification

Run:
  python api/main.py
  # or
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import logging

# Ensure repo root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Redirect HuggingFace cache to F: drive if it exists (4 GB space saver)
if os.path.exists("F:\\"):
    os.environ.setdefault("HF_HOME", r"F:\huggingface_cache")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyze import router as analyze_router
from api.routes.edit    import router as edit_router
from api.routes.verify  import router as verify_router

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spectra.api")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SPECTRA: Advanced Surgical Image Editing System",
    description=(
        "A self-verifying, VLM-driven image editing pipeline with "
        "Huffman Scene Graphs, kernel diffusion, and 5-check verification loops."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(analyze_router, prefix="/api", tags=["Analysis"])
app.include_router(edit_router,    prefix="/api", tags=["Editing"])
app.include_router(verify_router,  prefix="/api", tags=["Verification"])


# ── Health + Stats ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "SPECTRA Surgical AI API",
        "version": "2.0.0",
        "status": "online",
        "endpoints": ["/api/analyze", "/api/edit", "/api/verify"],
        "docs": "/docs",
    }


@app.get("/api/hsg/stats", tags=["Analysis"])
async def hsg_stats():
    """Return the adaptive HSG session statistics (edit frequency, weight rankings)."""
    from api.routes.edit import _get_pipeline
    try:
        pipeline = _get_pipeline()
        return pipeline.adaptive_rebalancer.get_stats()
    except Exception as e:
        return {"error": str(e), "note": "Pipeline not yet initialised."}


@app.get("/api/health", tags=["Health"])
async def health():
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }


# ── Legacy compatibility (keeps old /edit_agent working) ─────────────────────

from fastapi import UploadFile
from pydantic import BaseModel


class _LegacyRequest(BaseModel):
    image_base64: str
    prompt: str


@app.post("/edit_agent", tags=["Legacy"], deprecated=True,
          summary="[Deprecated] Use POST /api/edit instead")
async def legacy_edit_agent(request: _LegacyRequest):
    """Backwards-compatible shim: converts old-style request to new pipeline."""
    from api.schemas import EditRequest
    from api.routes.edit import edit as _edit
    new_req = EditRequest(
        image_base64=request.image_base64,
        instruction=request.prompt,
    )
    return await _edit(new_req)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
