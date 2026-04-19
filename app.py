import os
import io
import base64
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
import uvicorn

# SPECTRA Core Logic
from main_model import ImageGenerationModel, load_trained_model, device
from spectral_sync import SpectralSyncTracker
from llm_parser import call_ollama_parser, check_ollama_health
from dynamic_orchestrator import DynamicOrchestrator, _flatten_ssg

app = FastAPI(title="SPECTRA: Surgical AI Pipeline")

# Global instances
model = None
ssg_tracker = SpectralSyncTracker(decay_lambda=0.85)

# --- Pydantic Models ---
class EditAgentRequest(BaseModel):
    image_base64: str
    prompt: str

# --- Utilities ---
def decode_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    global model
    model = ImageGenerationModel().to(device)
    model.eval()

@app.get("/")
async def root():
    return {"message": "SPECTRA Surgical AI API Online"}

@app.get("/ssg_stats")
async def ssg_stats():
    return ssg_tracker.get_stats()

@app.post("/edit_agent")
async def edit_agent(request: EditAgentRequest):
    try:
        img_pil = decode_image(request.image_base64)
        
        # 1. SSG Extraction
        print("[API] Running SSG Extraction...")
        ssg_dict, masks_cache = model.extract_ssg(img_pil)

        # 2. Rebalancing
        ssg_adapted = ssg_tracker.update_weights(ssg_dict)

        # 3. LLM Parsing (Direct to Ollama)
        img_b64 = request.image_base64
        if check_ollama_health():
            res = call_ollama_parser(img_b64, ssg_adapted, request.prompt)
            edits = res.get("edits", [])
        else:
            # Simple fallback intent
            edits = [{"target_node": "semantic_hair", "intent": "change color", "influence": {"ZT": 0.8, "ZL": 0, "ZB": 0}}]

        # 4. Orchestration
        orchestrator = DynamicOrchestrator(model, ssg_tracker)
        result_pil, final_ssg, logs = orchestrator.execute_edit_schedule(
            img_pil, ssg_adapted, edits, initial_masks_cache=masks_cache
        )

        return {
            "status": "success",
            "execution_log": logs,
            "result_image_b64": encode_image(result_pil),
            "ssg_stats": ssg_tracker.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
