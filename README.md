# 💎 SPECTRA Surgical AI Pipeline

**Surgical Image Editing for Professional Workflows.**

SPECTRA is a high-fidelity image editing pipeline that moves beyond "Generative Fill" into **Surgical Feature Surgery**. It decomposes images into a **Spectral Semantic Graph (SSG)**, allowing bit-identical identity preservation while delivering dramatic aesthetic changes.

---

## 🦄 The SPECTRA Advantage

Standard AI editing (Generative Fill, Inpainting) often suffers from **Identity Shift**—where hair edits accidentally change the subject's eyes or nose. SPECTRA solves this via:

*   **Surgical Mask Algebra**: Automatically isolates target features (e.g., hair) while "hard-protecting" anatomical vitals (face, eyes).
*   **LBT Synchronization**: Synchronizes Light, Boundary, and Texture latents to ensure that a hair color change feels physically grounded in the scene's lighting.
*   **Spectral Integrity Sync**: Proprietary handling for 4GB VRAM cards using **Magic Upcast** (FP32 precision in RAM) to eliminate the "Blackout/NaN" bug common in low-resource environments.

## 🏗️ Architecture

SPECTRA uses a 6-stage semantic pipeline:
1.  **Extraction**: Recursive Spectral Tree generation + CLIPSeg semantic injection.
2.  **Intent Routing**: Classifies prompts (Lighting, Material, Structure) before parsing.
3.  **LLM Reasoning**: Phi-3 mini maps intent to the SSG.
4.  **Orchestration**: Surgical isolation and mask hygiene.
5.  **Diffusion Kernel**: Hybrid FP16/FP32 hardened inpainting.
6.  **Spectral Syncing**: Dynamic weight rebalancing based on session history.

> [!TIP]
> See [architecture.md](file:///c:/Users/ayush/OneDrive/Desktop/VBIP%20API/architecture.md) for detailed technical diagrams.

---

## 🚀 Quick Start

### 1. Requirements
*   **Python**: 3.10+
*   **Hardware**: 4GB VRAM (minimum) | 12GB RAM (minimum)
*   **Dependencies**: `diffusers`, `accelerate`, `transformers`, `torch`, `ollama`

### 2. Setup Ollama
```bash
ollama pull phi3:mini
```

### 3. Running a Test Edit
```bash
$env:HF_HOME="F:\huggingface_cache"
python test_e2e_edit.py --image "input.jpg" --prompt "Make her hair silver and her suit red"
```

## 🛠️ Hardware Hardening
SPECTRA was built to be **Hardware Agnostic**. On machines with < 8GB VRAM, the system automatically triggers:
*   **Model Offloading**: Sequential loading of UNet/VAE.
*   **Attention Slicing**: Reduced peak VRAM usage during the cross-attention pass.
*   **RAM Reclamation**: Aggressive unloading of the LLM before entering the Diffusion kernel.
