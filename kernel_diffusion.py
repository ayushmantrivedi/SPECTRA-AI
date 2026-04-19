"""
Kernel Latent Diffusion (IKLD) — Production Build
====================================================
Implements localized diffusion using Subspace Influence Tensors.

Editing path priority:
  1. SEMANTIC/SD PATH  (intent present + mask coverage >= 2%):
       CLIPSeg semantic mask → dilate → upsample to 512px → SD Inpainting → downsample → blend
  2. PIXEL-SPACE FALLBACK (no intent or mask too small):
       Deterministic ZT/ZL/ZB pixel transform inside mask region

Design notes:
  - SD is fed 512×512 (its native resolution) regardless of pipeline resolution.
  - Mask is morphologically dilated before SD so no tiny-mask invisible edits.
  - FP32 used for all PIL conversions to prevent FP16 NaN issues.
  - TurboQuant compression gated behind influence > 0.3 to prevent noise domination.
"""

from diffusers import StableDiffusionInpaintPipeline
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gc
import logging

from main_model import tq_tex, tq_light, tq_bound

SD_NATIVE_SIZE = 512   # Stable Diffusion's native resolution


class KernelDiffusionModule:
    def __init__(self, model, inpaint_model_id="runwayml/stable-diffusion-inpainting",
                 device=None):
        self.model = model
        from main_model import device as global_device
        self.device = device if device else global_device
        self.inpaint_model_id = inpaint_model_id
        self.steps = 10
        self.inpaint_pipeline = None

    # ------------------------------------------------------------------
    # SD PIPELINE LIFECYCLE
    # ------------------------------------------------------------------
    def load_pipeline(self):
        """Lazy load SD weights with 4GB VRAM optimisations."""
        if self.inpaint_pipeline is not None:
            return
        # Aggressive memory purge before loading ~4GB SD weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[KernelDiffusion] Loading {self.inpaint_model_id} in MAGIC UPCAST MODE (Full FP32 + Model Offload) ...")
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        is_cuda = "cuda" in str(self.device).lower()
        
        # 1. MAGIC UPCAST: Load the existing FP16 cache but into FP32 tensors
        # This prevents hardware NaNs and dtype crashes without downloading 4GB of weights.
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=torch.float32,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
        )

        if is_cuda:
            print("[KernelDiffusion] VRAM SAFETY: Using Model Offload + Attention Slicing + VAE Tiling")
            self.inpaint_pipeline.enable_model_cpu_offload()
            self.inpaint_pipeline.enable_attention_slicing()
            self.inpaint_pipeline.vae.enable_tiling()
        else:
            self.inpaint_pipeline.to(self.device)

    def unload_pipeline(self):
        """Purge SD weights to reclaim VRAM for CLIPSeg / Ollama."""
        print("[KernelDiffusion] Unloading SD ...")
        self.inpaint_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SUBSPACE DIFFUSION LOOP
    # ------------------------------------------------------------------
    def diffuse_subspace(self, z_base, target_feature, influence, tq_module,
                         step, total_steps):
        """
        Iterative TurboQuant-accelerated latent blending.
        TurboQuant compression only applied when influence > 0.3,
        preventing quantization noise from dominating subtle edits.
        """
        if influence <= 0.01:
            return z_base

        alpha_t      = step / float(total_steps)
        blend_factor = alpha_t * influence
        z_t = z_base * (1.0 - blend_factor) + target_feature * blend_factor

        # Gate TurboQuant — low-influence edits must not be corrupted by 4-bit noise
        if influence > 0.3:
            compressed = tq_module.compress(z_t, bits=4)
            return tq_module.decompress(compressed)
        return z_t

    # ------------------------------------------------------------------
    # MASK UTILITIES
    # ------------------------------------------------------------------
    @staticmethod
    def dilate_mask(mask_tensor, kernel_size=31):
        """
        Morphological dilation of a binary mask tensor [1,1,H,W].
        Ensures small semantic regions are covered with a generous inpainting area.
        """
        k = torch.ones(1, 1, kernel_size, kernel_size, device=mask_tensor.device)
        pad = kernel_size // 2
        dilated = torch.clamp(F.conv2d(mask_tensor.float(), k, padding=pad), 0, 1)
        return dilated

    @staticmethod
    def _safe_to_pil_mask(mask_tensor_1hw):
        """
        Safely convert a [1,H,W] or [H,W] mask tensor to a grayscale PIL Image.
        Uses float32 to avoid FP16 NaN issues.
        """
        m = mask_tensor_1hw.float().cpu().clamp(0, 1)
        if m.dim() == 3:
            m = m[0]
        return TF.to_pil_image((m * 255).to(torch.uint8))

    # ------------------------------------------------------------------
    # PIXEL-SPACE FALLBACK
    # ------------------------------------------------------------------
    def _apply_pixel_edit(self, full_img, mask_tensor, influence):
        """
        Deterministic pixel-space transform inside mask_tensor region.
        Used when SD is not triggered (no intent key or mask too small).
        """
        edited = full_img.clone()
        zt = influence.get("ZT", 0.0)
        zl = influence.get("ZL", 0.0)
        zb = influence.get("ZB", 0.0)

        # ZT: darken to black (silhouette / dark texture)
        if zt > 0.1:
            lum = (0.299 * edited[:, 0] +
                   0.587 * edited[:, 1] +
                   0.114 * edited[:, 2]).unsqueeze(1)
            dark   = lum * (1.0 - zt)
            target = dark.repeat(1, 3, 1, 1).to(edited.device)
            m3     = mask_tensor.repeat(1, 3, 1, 1).to(edited.device)
            edited = edited * (1 - m3) + target * m3

        # ZL: local brightness boost
        if zl > 0.1:
            lit    = torch.clamp(edited + zl * 0.5, 0, 1)
            m3     = mask_tensor.repeat(1, 3, 1, 1).to(edited.device)
            edited = edited * (1 - m3) + lit * m3

        # ZB: synthesise sun at the top of the mask region
        if zb > 0.1:
            B, C, H, W = full_img.shape
            y_idx, x_idx = torch.where(mask_tensor[0, 0] > 0.5)
            if len(y_idx) > 0:
                cy = int(y_idx.min().item()) + max(12, H // 10)
                cx = int(x_idx.float().mean().item())
                cy = max(14, min(H - 14, cy))
                cx = max(14, min(W - 14, cx))

                Y, X = torch.meshgrid(
                    torch.arange(H, device=full_img.device),
                    torch.arange(W, device=full_img.device),
                    indexing='ij'
                )
                dist     = torch.sqrt(((X - cx).float())**2 + ((Y - cy).float())**2)
                radius   = max(10.0, H * 0.09)
                glow_sig = max(16.0, H * 0.14)

                sun_core = (dist < radius).float()
                sun_glow = torch.exp(-dist / glow_sig)
                sun_r    = torch.clamp(sun_core + sun_glow * 0.8, 0, 1)
                sun_g    = torch.clamp(sun_core + sun_glow * 0.6, 0, 1)
                sun_b    = torch.clamp(sun_core * 0.8 + sun_glow * 0.1, 0, 1)
                sun_rgb  = torch.stack([sun_r, sun_g, sun_b], dim=0).unsqueeze(0)
                sun_a    = torch.clamp(sun_core + sun_glow * 0.5, 0, 1).unsqueeze(0).unsqueeze(0)
                edited   = edited * (1 - sun_a) + sun_rgb * sun_a

        return edited

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def run_diffusion_edit(self, full_img, base_features, target_node,
                           influence, mask_tensor):
        """
        Primary editing entry point.

        Routing:
          - If 'intent' in influence AND mask coverage >= 2%:
              → SD inpainting at 512×512 with dilated mask
          - Otherwise:
              → pixel-space deterministic edit (fallback, always visible)
        """
        zt_base, zl_base, zb_base = base_features

        # CTO Hardening: LBT Synchronization (Hand-in-Hand Editing)
        # If Texture (ZT) increases (e.g. silver/metallic), Light (ZL) should often scale too
        intent = influence.get("intent")
        zt_intent = influence.get("ZT", 0.0)
        zl_intent = influence.get("ZL", 0.0)
        zb_intent = influence.get("ZB", 0.0)
        
        # Heuristic: Silver/Light textures reflect more light
        if intent and "silver" in intent.lower() and zt_intent > 0.5:
            zl_intent = max(zl_intent, 0.4) # Automatic light boost for metallic hair
            print(f"[KernelDiffusion] LBT Sync: Boosted ZL to {zl_intent} for '{intent}'")

        # Subspace diffusion loop (shapes latent vectors)
        zt_target = zt_base * 0.05
        zl_target = zl_base * (1.0 + zl_intent)
        zb_target = zb_base

        zt_c, zl_c, zb_c = zt_base, zl_base, zb_base
        for step in range(1, self.steps + 1):
            zt_c = self.diffuse_subspace(
                zt_c, zt_target, zt_intent, tq_tex,   step, self.steps)
            zl_c = self.diffuse_subspace(
                zl_c, zl_target, zl_intent, tq_light, step, self.steps)
            zb_c = self.diffuse_subspace(
                zb_c, zb_target, zb_intent, tq_bound, step, self.steps)

        # Route decision
        has_intent = "intent" in influence
        mask_coverage = mask_tensor.mean().item()        # fraction of image covered
        use_sd = has_intent and mask_coverage >= 0.001   # >= 0.1% coverage min

        # ── SD INPAINTING PATH ─────────────────────────────────────────
        if use_sd:
            # CLEANSE: Binarize mask to prevent broad-spectrum background blackout from CLIPSeg noise
            mask_tensor = (mask_tensor > 0.5).float()
            
            self.load_pipeline()
            print(f"[KernelDiffusion] SD Inpainting | mask coverage: "
                  f"{mask_coverage*100:.1f}% | node: {target_node}")

            H_orig, W_orig = full_img.shape[2], full_img.shape[3]

            # 1. Dilate mask relatif to image size (e.g., 5% of largest dimension)
            # Ensures tiny HSG nodes become paintable without swallowing the whole background.
            target_dim = max(H_orig, W_orig)
            k_size = max(5, int(target_dim * 0.05))
            if k_size % 2 == 0: k_size += 1
            
            dilated = self.dilate_mask(mask_tensor, kernel_size=k_size)

            # 2. Minimum coverage pass (if still < 2% of image, dilate slightly more)
            if dilated.mean().item() < 0.02:
                k_size_extra = max(3, int(target_dim * 0.03))
                if k_size_extra % 2 == 0: k_size_extra += 1
                dilated = self.dilate_mask(dilated, kernel_size=k_size_extra)

            # 3. Upsample image + dilated mask to SD native 512×512
            img_512  = F.interpolate(full_img, size=(SD_NATIVE_SIZE, SD_NATIVE_SIZE),
                                     mode='bilinear', align_corners=False)
            mask_512 = F.interpolate(dilated,   size=(SD_NATIVE_SIZE, SD_NATIVE_SIZE),
                                     mode='nearest')

            # 4. Safe FP32 → uint8 conversion (prevents FP16 NaN)
            init_pil = TF.to_pil_image(
                (img_512[0].float().cpu().clamp(0, 1) * 255).to(torch.uint8))
            mask_pil = self._safe_to_pil_mask(mask_512[0])

            # Feature-Guided Prompting (Translating Latents to Tokens)
            feature_tokens = []
            if zt_intent > 0.6: feature_tokens.append("highly detailed texture")
            if zl_intent > 0.4: feature_tokens.append("specular highlights, reflective")
            if zb_intent > 0.4: feature_tokens.append("sharp edges, defined silhouette")
            f_guidance = ", ".join(feature_tokens)

            intent_text    = influence.get("intent", "high quality realistic edit")
            # Production Grade Prompt Augmentation
            prompt = (f"{intent_text}, {f_guidance}, masterpiece, photorealistic, "
                      f"professional studio lighting, 8k resolution")
            
            negative_prompt = ("low quality, bad anatomy, deformed, eyes closed, missing features, "
                               "worst quality, ugly, blurry, artifacts, distortion, "
                               "pixelated, cartoon, illustration, low resolution")

            # IDENTITY GUARD: If we are editing a person, lower the strength to preserve eyes/mouth/nose
            # instead of erasing them. 0.75 is the "Sweet Spot" for AI-remodelling.
            is_person = "person" in str(target_node).lower() or influence.get("ZT", 0) > 0.5
            strength = 0.75 if is_person else 0.95
            
            result_pil = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_pil,
                mask_image=mask_pil,
                strength=strength,
                num_inference_steps=40,
                guidance_scale=9.0,
                height=SD_NATIVE_SIZE,
                width=SD_NATIVE_SIZE,
            ).images[0]

            self.unload_pipeline()

            # 5. Downsample result back to original resolution
            result_t = TF.to_tensor(result_pil).unsqueeze(0)
            result_t = F.interpolate(result_t, size=(H_orig, W_orig),
                                     mode='bilinear', align_corners=False)
            result_t = result_t.to(full_img.device)

            # 6. Professional Smooth Blend (Soft Gaussian Feathering)
            # This prevents the "erased" cutout look at the edges
            blend_mask = F.interpolate(dilated, size=(H_orig, W_orig),
                                       mode='bilinear', align_corners=False)
            
            # Feather the edges significantly (25px radius for 512px context)
            feather_size = max(5, int(max(H_orig, W_orig) * 0.05))
            if feather_size % 2 == 0: feather_size += 1
            blend_mask = TF.gaussian_blur(blend_mask, kernel_size=[feather_size, feather_size])
            
            mv = blend_mask.max()
            if mv > 0.001:
                blend_mask = blend_mask / mv
            
            blend_mask3 = blend_mask.repeat(1, 3, 1, 1).to(full_img.device)
            result_t = result_t.to(full_img.device)

            print(f"[DEBUG Device] full_img: {full_img.device}, result_t: {result_t.device}, blend_mask3: {blend_mask3.device}")
            # Explicitly force everything to the same device again just in case
            full_img = full_img.to(full_img.device)
            blend_mask3 = blend_mask3.to(full_img.device)
            result_t = result_t.to(full_img.device)

            return full_img * (1.0 - blend_mask3) + result_t * blend_mask3

        # ── PIXEL-SPACE FALLBACK PATH ──────────────────────────────────
        print(f"[KernelDiffusion] Pixel-space edit | node: {target_node} | "
              f"{'no intent' if not has_intent else 'mask too small'}")
        edited = self._apply_pixel_edit(full_img, mask_tensor, influence)

        smooth = TF.gaussian_blur(mask_tensor, kernel_size=[9, 9])
        mv = smooth.max()
        if mv > 0.01:
            smooth = smooth / mv
        smooth3 = smooth.repeat(1, 3, 1, 1)

        return full_img * (1 - smooth3) + edited * smooth3
