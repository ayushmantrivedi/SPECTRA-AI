"""
SPECTRA Orchestrator VLM
=========================
Stage 1 of the SPECTRA pipeline.

Architecture:
  Input Image → Shared Backbone (ViT / ConvNeXt) → Multi-Scale Feature Maps
      → Color Head | Edge Head | Texture Head
      → Cross-Attention Fusion
      → Unified Segmentation + Attribute Extraction
      → HSG Builder

When a hosted VLM (Gemini / Claude) API key is present, scene understanding is
delegated to the cloud model, which returns a structured HSG JSON.  When no API
key is available (offline mode) the system falls back to the local CLIPSeg +
feature-extractor pipeline already present in main_model.py.
"""

from __future__ import annotations

import os
import base64
import json
import io
import gc
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

logger = logging.getLogger("spectra.orchestrator")

# ── Optional heavy imports (guard against missing deps) ──────────────────────

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini VLM disabled.")

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    _CLIPSEG_AVAILABLE = True
except ImportError:
    _CLIPSEG_AVAILABLE = False
    logger.warning("CLIPSeg not available. Falling back to structure-only HSG.")


# ── HSG Schema for VLM prompt ─────────────────────────────────────────────────

_HSG_SCHEMA = """
Return a JSON object that exactly follows this schema:
{
  "id": "SCENE_ROOT",
  "label": "scene",
  "weight": 1.0,
  "depth": 0,
  "global_attributes": {
    "lighting": "<string>",
    "palette": ["<hex>", ...],
    "ambient_intensity": <float 0-1>
  },
  "children": [
    {
      "id": "<unique_snake_case_id>",
      "label": "<human_label>",
      "weight": <float 0-1>,
      "depth": 1,
      "bbox": [<x>, <y>, <w>, <h>],
      "attributes": {
        "color": {"primary": "<name>", "hex": "<hex>"},
        "edge": {"boundary_type": "<hard|soft|shadow>", "sharpness": <float>},
        "texture": {"material_class": "<string>", "roughness": <float 0-1>}
      },
      "children": []
    }
  ]
}
Rules:
- weight reflects visual saliency + semantic importance (0=background, 1=primary subject)
- bbox is [x, y, width, height] in pixels relative to the image
- Include at least: scene root, primary subjects, background, lighting node
- Do NOT include markdown fences or extra commentary — pure JSON only
"""

_SYSTEM_PROMPT = """You are SPECTRA's Orchestrator Vision-Language Model.
Your task is to deeply analyse the provided image and build a complete
Huffman Scene Graph (HSG) — a hierarchical understanding of every significant
region, object, and relationship in the scene.

Expert heads you must simulate:
1. COLOR HEAD — identify hue, saturation, palette, lighting temperature
2. EDGE HEAD  — detect boundary types, sharpness, occlusion, depth order
3. TEXTURE HEAD — classify material, roughness, pattern, reflectivity

""" + _HSG_SCHEMA


# ── Specialist Expert Heads (local CNN-based fallback) ────────────────────────


class ColorExpertHead(nn.Module):
    """Local color attribute prediction head.

    Operates on multi-scale feature maps from the shared backbone.
    Returns per-region colour descriptors.
    """

    def __init__(self, in_channels: int = 256, num_color_classes: int = 12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_color_classes)
        self.temperature_head = nn.Linear(128, 2)   # warm / cool logits
        self.brightness_head = nn.Linear(128, 1)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv(feat).flatten(1)
        return {
            "color_class_logits": self.classifier(x),
            "temperature_logits": self.temperature_head(x),
            "brightness": torch.sigmoid(self.brightness_head(x)),
        }


class EdgeExpertHead(nn.Module):
    """Local edge / boundary analysis head."""

    _EDGE_TYPES = ["hard", "soft", "shadow", "occlusion", "texture"]

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.edge_type_head = nn.Linear(64, len(self._EDGE_TYPES))
        self.sharpness_head = nn.Linear(64, 1)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv(feat).flatten(1)
        return {
            "edge_type_logits": self.edge_type_head(x),
            "sharpness": torch.sigmoid(self.sharpness_head(x)),
        }


class TextureExpertHead(nn.Module):
    """Local texture / material classification head."""

    _MATERIAL_CLASSES = [
        "skin", "cotton", "silk", "leather", "metal",
        "glass", "wood", "stone", "plastic", "water", "organic",
    ]

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.material_head = nn.Linear(128, len(self._MATERIAL_CLASSES))
        self.roughness_head = nn.Linear(128, 1)
        self.embed_head = nn.Linear(128, 64)  # texture descriptor embedding

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv(feat).flatten(1)
        return {
            "material_logits": self.material_head(x),
            "roughness": torch.sigmoid(self.roughness_head(x)),
            "texture_embedding": self.embed_head(x),
        }


class FusionModule(nn.Module):
    """Cross-attention fusion of the three expert head outputs."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.proj_color = nn.Linear(12 + 2 + 1, embed_dim)
        self.proj_edge = nn.Linear(5 + 1, embed_dim)
        self.proj_texture = nn.Linear(11 + 1 + 64, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        color_out: Dict[str, torch.Tensor],
        edge_out: Dict[str, torch.Tensor],
        texture_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        c_feat = torch.cat([
            color_out["color_class_logits"],
            color_out["temperature_logits"],
            color_out["brightness"],
        ], dim=-1)
        e_feat = torch.cat([
            edge_out["edge_type_logits"],
            edge_out["sharpness"],
        ], dim=-1)
        t_feat = torch.cat([
            texture_out["material_logits"],
            texture_out["roughness"],
            texture_out["texture_embedding"],
        ], dim=-1)

        c = self.proj_color(c_feat).unsqueeze(1)
        e = self.proj_edge(e_feat).unsqueeze(1)
        t = self.proj_texture(t_feat).unsqueeze(1)

        tokens = torch.cat([c, e, t], dim=1)           # (B, 3, embed_dim)
        fused = self.transformer(tokens)                 # cross-attend
        return self.out_proj(fused.flatten(1))           # (B, embed_dim*3)


# ── Main Orchestrator ─────────────────────────────────────────────────────────


class OrchestratorVLM:
    """
    SPECTRA Stage 1 — Scene Understanding + HSG Construction.

    Routing priority:
      1. Gemini 2.5 Pro (if GEMINI_API_KEY set)
      2. Any OpenAI-compatible endpoint (if OPENAI_API_KEY set)
      3. Local CLIPSeg + feature-extractor fallback

    Returns a structured HSG dict ready for HuffmanSceneGraph.from_dict().
    """

    # Semantic categories for local CLIPSeg segmentation
    SEMANTIC_PROMPTS = [
        "person", "face", "hair", "eyes", "mouth", "skin",
        "clothing", "upper body", "shirt", "jacket", "hat",
        "background", "sky", "grass", "tree", "ground",
        "shadow", "light source", "hands",
    ]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clipseg_processor = None
        self._clipseg_model = None
        self._gemini_model = None
        self._gemini_configured = False
        self._init_gemini()

    # ── Gemini setup ──────────────────────────────────────────────────────────

    def _init_gemini(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and _GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro-preview-05-06",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                self._gemini_configured = True
                logger.info("[Orchestrator] Gemini 2.5 Pro configured.")
            except Exception as e:
                logger.warning(f"[Orchestrator] Gemini setup failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self, image_pil: Image.Image, timeout_s: float = 60.0
    ) -> Dict[str, Any]:
        """
        Analyse an image and return a complete HSG dict + metadata.

        Returns:
            {
              "hsg": { ... },          # Huffman Scene Graph JSON
              "metadata": { ... },     # Timing, confidence, etc.
            }
        """
        t0 = time.perf_counter()

        if self._gemini_configured:
            result = self._analyze_gemini(image_pil, timeout_s)
        else:
            result = self._analyze_local(image_pil)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result.setdefault("metadata", {})["processing_time_ms"] = elapsed_ms
        result["metadata"]["image_resolution"] = list(image_pil.size)
        return result

    # ── Gemini VLM Path ───────────────────────────────────────────────────────

    def _analyze_gemini(
        self, image_pil: Image.Image, timeout_s: float
    ) -> Dict[str, Any]:
        """Delegate scene understanding to Gemini 2.5 Pro."""
        try:
            # Resize to reduce token cost while preserving aspect ratio
            img = self._prepare_image(image_pil, max_dim=1024)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()

            response = self._gemini_model.generate_content(
                [
                    _SYSTEM_PROMPT,
                    {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_bytes).decode(),
                    },
                ]
            )

            raw_text = response.text.strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = "\n".join(
                    line for line in raw_text.split("\n")
                    if not line.startswith("```")
                )

            hsg_dict = json.loads(raw_text)
            return {
                "hsg": hsg_dict,
                "metadata": {
                    "source": "gemini-2.5-pro",
                    "detected_objects_count": len(hsg_dict.get("children", [])),
                    "fusion_confidence": 0.95,
                },
            }
        except Exception as e:
            logger.warning(f"[Orchestrator] Gemini failed ({e}), falling back to local.")
            return self._analyze_local(image_pil)

    # ── Local Fallback Path ───────────────────────────────────────────────────

    def _analyze_local(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Build HSG from local CLIPSeg segmentation + simple feature extraction.
        Returns the same schema as the Gemini path.
        """
        img = self._prepare_image(image_pil, max_dim=512)
        W, H = img.size

        semantic_nodes: List[Dict[str, Any]] = []

        if _CLIPSEG_AVAILABLE:
            semantic_nodes = self._run_clipseg(img, H, W)

        hsg = self._build_local_hsg(img, semantic_nodes, H, W)

        return {
            "hsg": hsg,
            "metadata": {
                "source": "local_clipseg",
                "detected_objects_count": len(semantic_nodes),
                "fusion_confidence": 0.72,
            },
        }

    def _run_clipseg(
        self, img: Image.Image, H: int, W: int
    ) -> List[Dict[str, Any]]:
        """Segment image with CLIPSeg and return node descriptors."""
        self._load_clipseg()
        if self._clipseg_model is None:
            return []

        try:
            # Let the processor handle resizing, but force 224x224 if it defaults to 352
            inputs = self._clipseg_processor(
                text=self.SEMANTIC_PROMPTS,
                images=[img] * len(self.SEMANTIC_PROMPTS),
                padding=True,
                return_tensors="pt",
            )
            
            if inputs.pixel_values.shape[-1] != 224:
                # Force resize to 224 if the processor defaults to 352 but model wants 224
                inputs.pixel_values = F.interpolate(
                    inputs.pixel_values, size=(224, 224), 
                    mode="bilinear", align_corners=False
                )
            
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self._clipseg_model(**inputs)

            preds = torch.sigmoid(outputs.logits)
            if preds.dim() == 2:
                preds = preds.unsqueeze(0)
            preds = F.interpolate(preds.unsqueeze(1), size=(H, W), mode="bilinear")

            total_px = float(H * W)
            nodes: List[Dict[str, Any]] = []

            for idx, label in enumerate(self.SEMANTIC_PROMPTS):
                mask = (preds[idx, 0] > 0.25).float()
                coverage = float(mask.sum()) / total_px
                if coverage < 0.01:
                    continue

                ys, xs = torch.where(mask > 0.3)
                if len(ys) == 0:
                    continue
                x0 = int(xs.min()); y0 = int(ys.min())
                x1 = int(xs.max()); y1 = int(ys.max())

                # Simple colour sampling
                arr = np.array(img)
                region = arr[y0:y1, x0:x1]
                avg_rgb = tuple(int(c) for c in region.mean(axis=(0, 1))[:3]) if region.size else (128, 128, 128)
                primary_hex = "#{:02x}{:02x}{:02x}".format(*avg_rgb)

                weight = min(1.0, 0.3 + coverage * 1.4)
                nodes.append({
                    "id": f"semantic_{label.replace(' ', '_')}",
                    "label": label,
                    "weight": round(weight, 4),
                    "depth": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "attributes": {
                        "color": {"primary": label, "hex": primary_hex},
                        "edge": {"boundary_type": "soft", "sharpness": 0.7},
                        "texture": {"material_class": "unknown", "roughness": 0.5},
                        "coverage": round(coverage, 4),
                    },
                    "children": [],
                })

            return nodes
        finally:
            self._unload_clipseg()

    def _build_local_hsg(
        self,
        img: Image.Image,
        semantic_nodes: List[Dict[str, Any]],
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        """Assemble local HSG from CLIPSeg nodes."""
        arr = np.array(img)
        avg_rgb = tuple(int(c) for c in arr.mean(axis=(0, 1))[:3])
        palette_hex = "#{:02x}{:02x}{:02x}".format(*avg_rgb)

        # Sort children by weight descending (Huffman ordering)
        children = sorted(semantic_nodes, key=lambda x: x["weight"], reverse=True)

        return {
            "id": "SCENE_ROOT",
            "label": "scene",
            "weight": 1.0,
            "depth": 0,
            "global_attributes": {
                "lighting": "ambient",
                "palette": [palette_hex],
                "ambient_intensity": 0.75,
            },
            "attributes": {},
            "children": children,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_image(img: Image.Image, max_dim: int = 1024) -> Image.Image:
        img = img.convert("RGB")
        W, H = img.size
        if max(W, H) > max_dim:
            scale = max_dim / max(W, H)
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        return img

    def _load_clipseg(self) -> None:
        if self._clipseg_model is not None or not _CLIPSEG_AVAILABLE:
            return
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self._clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device, dtype=dtype)

    def _unload_clipseg(self) -> None:
        self._clipseg_model = None
        self._clipseg_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
