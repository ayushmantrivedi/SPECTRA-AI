# ===============================================================
# Feature-Conditioned Image Generation & Inpainting Model
# Works with preprocessing model from prototype1.py
# Generates high-quality images using texture, light, and boundary features
# ===============================================================
#
# This model implements:
# 1. Image Generation: Generate new images from texture, light, and boundary features
# 2. Image Inpainting: Fill masked regions using feature-guided generation
#
# Usage:
# ------
# Training:
#   python "main model.py"
#
# Inference (after training):
#   from "main model" import ImageGenerationModel, load_trained_model
#   model = load_trained_model('checkpoints/final_model.pth', device)
#   
#   # Generate from reference image
#   generated, features = model.generate_from_image(reference_img)
#   
#   # Inpaint masked image
#   inpainted, _ = model.inpaint(masked_img, mask, reference_img)
#
# Features:
# - Uses preprocessing model (prototype1.py) to extract texture, light, boundary features
# - Conditional GAN architecture for high-quality generation
# - Perceptual loss for better visual quality
# - Supports inpainting with various mask types
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

# Redirect bulky model cache to F: drive to save C: space
os.environ["HF_HOME"] = r"F:\huggingface_cache"

# Centralized device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MainModel] Using device: {device}")

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except ImportError:
    CLIPSegProcessor, CLIPSegForImageSegmentation = None, None

# ==================== Semantic Segmenter (CLIPSeg) ====================
class SemanticSegmenter:
    """Uses CLIPSeg for zero-shot text-to-segmentation for high-quality masks"""
    def __init__(self, device=device):
        self.device = device
        self.processor = None
        self.model = None

    def load_model(self):
        """Lazy load CLIPSeg weights"""
        if self.model is not None:
            return
        if CLIPSegProcessor is not None:
            print(f"[SemanticSegmenter] Loading CLIPSeg model (JIT) onto {self.device}...")
            # Use float16 on CUDA to save VRAM
            dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
            self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model = self.model.to(self.device, dtype=dtype)

    def unload_model(self):
        """Purge CLIPSeg from RAM"""
        print("[SemanticSegmenter] Unloading CLIPSeg to free RAM...")
        self.model = None
        self.processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment(self, image, prompts):
        """Generate masks for a list of text prompts"""
        self.load_model()
        if self.model is None or self.processor is None:
            return None
        
        # Explicitly resize PIL images to 224x224 to avoid model mismatch
        resized_image = image.resize((224, 224))
        
        # Preprocess and segment - force size to 224
        inputs = self.processor(
            text=prompts, 
            images=[resized_image] * len(prompts), 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Absolute force for model compatibility
        if inputs.pixel_values.shape[-2:] != (224, 224):
            # Use dictionary assignment to ensure BatchEncoding update
            inputs["pixel_values"] = torch.nn.functional.interpolate(inputs.pixel_values, size=(224, 224), mode="bilinear")
        
        # Double check
        # print(f"[DEBUG] CLIPSeg Forward Shape: {inputs.pixel_values.shape}")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # outputs.logits shape: (len(prompts), H, W)
        # Resize to original image size
        preds = torch.sigmoid(outputs.logits)
        if preds.dim() == 2:
            preds = preds.unsqueeze(0)
            
        # Standardize to (len(prompts), 1, H, W)
        H, W = image.size[1], image.size[0]
        preds = F.interpolate(preds.unsqueeze(1), size=(H, W), mode="bilinear")
        return preds
import os

import prototype1
from prototype1 import preprocess, TextureEncoderConvNeXt, LightEncoderVGG, BoundaryEncoderResNet
import turboquant_utils
from turboquant_utils import TurboQuant, apply_turboquant
from ssg_builder import SSGBuilder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available() # Enable AMP if CUDA is available
scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)
# Equilibrium parameters
k_t = 0.0
gamma = 0.5
lambda_k = 1e-3

# Initialize TurboQuant for each embedding dimension
tq_tex = TurboQuant(128, device=device)
tq_light = TurboQuant(64, device=device)
tq_bound = TurboQuant(32, device=device)
# Ensure prototype1 uses the same device
prototype1.device = device
print("Device:", device)

# ==================== Best-in-Class Architectural Components ====================

class SelfAttention(nn.Module):
    """Self-attention layer for capturing global dependencies in texture/light"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(B, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class HybridResBlock(nn.Module):
    """Hybrid Block: ResNet short-circuit for stability + VGG-style bottleneck for detail"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.vgg_path = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = self.main_path(x) + self.shortcut(x)
        res = torch.relu(res)
        return self.vgg_path(res)

# ==================== Feature Extractor (Uses Preprocessing Model) ====================
class FeatureExtractor(nn.Module):
    """Extracts texture, light, and boundary features using preprocessing model"""
    def __init__(self, texture_emb_dim=128, light_emb_dim=64, boundary_emb_dim=32, 
                 load_pretrained_weights=None, freeze_encoders=False):
        super().__init__()
        self.enc_tex = TextureEncoderConvNeXt(emb_dim=texture_emb_dim)
        self.enc_light = LightEncoderVGG(emb_dim=light_emb_dim)
        self.enc_bound = BoundaryEncoderResNet(emb_dim=boundary_emb_dim)
        
        # Load pretrained weights if provided
        if load_pretrained_weights is not None:
            self.load_pretrained_weights(load_pretrained_weights)
            freeze_encoders = True # Auto-freeze if loaded
        
        # Freeze encoders if requested
        if freeze_encoders:
            for p in self.enc_tex.parameters(): p.requires_grad = False
            for p in self.enc_light.parameters(): p.requires_grad = False
            for p in self.enc_bound.parameters(): p.requires_grad = False
        
    def load_pretrained_weights(self, checkpoint_path):
        """Load weights from a trained preprocessing model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'enc_tex' in checkpoint:
                self.enc_tex.load_state_dict(checkpoint['enc_tex'])
            if 'enc_light' in checkpoint:
                self.enc_light.load_state_dict(checkpoint['enc_light'])
            if 'enc_bound' in checkpoint:
                self.enc_bound.load_state_dict(checkpoint['enc_bound'])
            print(f"Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        
    def forward(self, imgs):
        """Extract features from images"""
        tex, light, bound = preprocess(imgs)
        zt = self.enc_tex(tex)
        zl = self.enc_light(light)
        zb = self.enc_bound(bound)
        return zt, zl, zb, (tex, light, bound)

# ==================== Generator (Feature-Conditioned) ====================
class FeatureConditionedGenerator(nn.Module):
    """Generator that creates images from texture, light, and boundary features"""
    def __init__(self, texture_dim=128, light_dim=64, boundary_dim=32, noise_dim=100, 
                 base_channels=64, img_size=128):
        super().__init__()
        self.texture_dim = texture_dim
        self.light_dim = light_dim
        self.boundary_dim = boundary_dim
        self.noise_dim = noise_dim
        self.img_size = img_size
        self.base_channels = base_channels  # store for forward reshape
        
        # Project features to common dimension
        self.texture_proj = nn.Sequential(
            nn.Linear(texture_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.light_proj = nn.Sequential(
            nn.Linear(light_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.boundary_proj = nn.Sequential(
            nn.Linear(boundary_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Combine features
        combined_dim = 256 + 128 + 128 + noise_dim  # texture + light + boundary + noise
        # Feature fusion produces a (B, base_channels * (H/8) * (W/8)) vector
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, base_channels * (img_size // 8) * (img_size // 8))
        )
        
        # Stage 1: 16x16 -> 32x32
        self.deconv1 = nn.ConvTranspose2d(base_channels, base_channels*2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels*2)
        self.res1 = HybridResBlock(base_channels*2, base_channels*2)
        
        # Stage 2: 32x32 -> 64x64
        self.deconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.attn = SelfAttention(base_channels) # Global coherence at 64x64
        self.res2 = HybridResBlock(base_channels, base_channels)
        
        # Stage 3: 64x64 -> 128x128
        self.deconv3 = nn.ConvTranspose2d(base_channels, base_channels//2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels//2)
        
        # Final conv
        self.final_conv = nn.Conv2d(base_channels//2, 3, 3, 1, 1)
        
        # FiLM layers for light modulation
        self.light_film = nn.ModuleList([
            nn.Linear(light_dim, base_channels*4),  # for 16->32
            nn.Linear(light_dim, base_channels*2),  # for 32->64
            nn.Linear(light_dim, base_channels)     # for 64->128
        ])
        # Gating network to constrain light modulation within texture/boundary regions
        # Input: [B, 2, H, W] (1-ch texture, 1-ch boundary), Output: [B, 1, H, W] sigmoid gate
        self.gating_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z_texture, z_light, z_boundary, noise=None, texture_map=None, boundary_map=None):
        """
        Generate image from features
        Args:
            z_texture: texture features [B, texture_dim]
            z_light: light features [B, light_dim]
            z_boundary: boundary features [B, boundary_dim]
            noise: optional noise [B, noise_dim]
            texture_map: optional texture map [B, C, H, W]
            boundary_map: optional boundary map [B, 1, H, W]
        """
        B = z_texture.size(0)
        # Project features
        tex_proj = self.texture_proj(z_texture)
        light_proj = self.light_proj(z_light)
        bound_proj = self.boundary_proj(z_boundary)
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=z_texture.device)
        # Combine features and make initial feature map
        combined = torch.cat([tex_proj, light_proj, bound_proj, noise], dim=1)
        initial_size = self.img_size // 8
        features = self.feature_fusion(combined)
        # use stored base_channels instead of hard-coded 64
        features = features.view(B, self.base_channels, initial_size, initial_size)
        # Spatial gating preparation
        gating_map = None
        if texture_map is not None and boundary_map is not None:
            tex_1ch = texture_map.mean(dim=1, keepdim=True) if texture_map.size(1) > 1 else texture_map
            bound_1ch = boundary_map[:, :1, :, :]
            gating_map = self.gating_net(torch.cat([tex_1ch, bound_1ch], dim=1))  # [B,1,H,W]
        # Stage 1: 16->32 with FiLM and gating
        x = F.relu(self.bn1(self.deconv1(features)), inplace=True)
        x = self.res1(x) # Residual stability
        gamma_beta = self.light_film[0](z_light)
        C1 = x.size(1)
        gamma1, beta1 = gamma_beta[:, :C1], gamma_beta[:, C1:]
        x = x * (1 + gamma1.view(B, C1, 1, 1)) + beta1.view(B, C1, 1, 1)
        if gating_map is not None:
            g1 = F.interpolate(gating_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x * (0.5 + 0.5 * g1)
        # Stage 2: 32->64
        x = F.relu(self.bn2(self.deconv2(x)), inplace=True)
        x = self.attn(x) # Global attention
        x = self.res2(x) # Hybrid bottleneck
        gamma_beta = self.light_film[1](z_light)
        C2 = x.size(1)
        gamma2, beta2 = gamma_beta[:, :C2], gamma_beta[:, C2:]
        x = x * (1 + gamma2.view(B, C2, 1, 1)) + beta2.view(B, C2, 1, 1)
        if gating_map is not None:
            g2 = F.interpolate(gating_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x * (0.5 + 0.5 * g2)
        # Stage 3: 64->128
        x = F.relu(self.bn3(self.deconv3(x)), inplace=True)
        gamma_beta = self.light_film[2](z_light)
        C3 = x.size(1)
        gamma3, beta3 = gamma_beta[:, :C3], gamma_beta[:, C3:]
        x = x * (1 + gamma3.view(B, C3, 1, 1)) + beta3.view(B, C3, 1, 1)
        if gating_map is not None:
            g3 = F.interpolate(gating_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x * (0.5 + 0.5 * g3)
        # Final image
        img = self.final_conv(x)
        img = torch.tanh(img)
        img = (img + 1) / 2
        return img

# ==================== Inpainting Generator ====================
class InpaintingGenerator(nn.Module):
    """Generator for inpainting using features and masked images"""
    def __init__(self, texture_dim=128, light_dim=64, boundary_dim=32, base_channels=64, img_size=128):
        super().__init__()
        self.texture_dim = texture_dim
        self.light_dim = light_dim
        self.boundary_dim = boundary_dim
        
        # Feature projectors
        self.texture_proj = nn.Linear(texture_dim, 256)
        self.light_proj = nn.Linear(light_dim, 128)
        self.boundary_proj = nn.Linear(boundary_dim, 128)
        
        # Encoder for masked image
        self.encoder = nn.Sequential(
            nn.Conv2d(4, base_channels, 4, 2, 1),  # 4 channels: RGB + mask
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128 + base_channels*8 * (img_size//16)**2, base_channels*8 * (img_size//16)**2),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, masked_img, mask, z_texture, z_light, z_boundary):
        """
        Inpaint masked region using features
        Args:
            masked_img: image with masked region [B, 3, H, W]
            mask: binary mask [B, 1, H, W] (1 = masked, 0 = visible)
            z_texture: texture features [B, texture_dim]
            z_light: light features [B, light_dim]
            z_boundary: boundary features [B, boundary_dim]
        """
        B = masked_img.size(0)
        
        # Project features
        tex_proj = self.texture_proj(z_texture)
        light_proj = self.light_proj(z_light)
        bound_proj = self.boundary_proj(z_boundary)
        features = torch.cat([tex_proj, light_proj, bound_proj], dim=1)
        
        # Encode masked image
        masked_input = torch.cat([masked_img, mask], dim=1)
        encoded = self.encoder(masked_input)
        
        # Flatten and fuse
        encoded_flat = encoded.view(B, -1)
        fused = self.fusion(torch.cat([features, encoded_flat], dim=1))
        
        # Reshape and decode
        initial_size = masked_img.size(2) // 16
        fused = fused.view(B, -1, initial_size, initial_size)
        output = self.decoder(fused)
        
        # Convert from [-1, 1] to [0, 1]
        output = (output + 1) / 2
        
        # Combine with original image
        mask_3ch = mask.repeat(1, 3, 1, 1)
        inpainted = masked_img * (1 - mask_3ch) + output * mask_3ch
        
        return inpainted, output

# ==================== Discriminator ====================
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def conv_block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(in_channels, base_channels, norm=False),
            *conv_block(base_channels, base_channels*2),
            *conv_block(base_channels*2, base_channels*4),
            *conv_block(base_channels*4, base_channels*8, stride=1),
            nn.Conv2d(base_channels*8, 1, 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.model(img)

# ==================== Perceptual Loss ====================
class PerceptualLoss(nn.Module):
    """Perceptual loss using a lighter VGG backbone; supports CPU execution to save GPU memory"""
    def __init__(self, use_cpu=True, backbone='vgg11', max_layers=18):
        super().__init__()
        if backbone == 'vgg19':
            vgg = models.vgg19(weights="IMAGENET1K_V1").features
        else:
            vgg = models.vgg11(weights="IMAGENET1K_V1").features
        # Use fewer layers to reduce memory
        self.feature_layers = nn.Sequential(*list(vgg.children())[:max_layers])
        for p in self.feature_layers.parameters():
            p.requires_grad = False
        # Mean/std buffers on CPU to avoid GPU RAM usage if use_cpu=True
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.on_cpu = use_cpu
        if self.on_cpu:
            self.feature_layers = self.feature_layers.cpu()
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
        
    def normalize(self, x):
        """Normalize images for VGG (ImageNet stats)"""
        return (x - self.mean) / self.std
        
    def forward(self, pred, target):
        if self.on_cpu:
            # Compute perceptual loss on CPU to save GPU memory
            with torch.no_grad():
                pred_cpu = pred.detach().cpu()
                target_cpu = target.detach().cpu()
            pred_norm = self.normalize(pred_cpu)
            target_norm = self.normalize(target_cpu)
            pred_features = self.feature_layers(pred_norm)
            target_features = self.feature_layers(target_norm)
        else:
            pred_norm = self.normalize(pred)
            target_norm = self.normalize(target)
            pred_features = self.feature_layers(pred_norm)
            target_features = self.feature_layers(target_norm)
        return F.mse_loss(pred_features, target_features)

# ==================== Full Model ====================
class ImageGenerationModel(nn.Module):
    """Complete model for image generation and inpainting
    
    This model works with the preprocessing model (prototype1.py) to:
    1. Extract texture, light, and boundary features from images
    2. Generate new images conditioned on these features
    3. Perform inpainting using features to fill masked regions
    
    Architecture:
    - FeatureExtractor: Uses preprocessing model encoders to extract features
    - Generator: Creates images from texture/light/boundary features + noise
    - InpaintingGenerator: Fills masked regions using features
    """
    def __init__(self, texture_dim=128, light_dim=64, boundary_dim=32, noise_dim=100,
                 pretrained_weights=None):
        super().__init__()
        self.texture_dim = texture_dim
        self.light_dim = light_dim
        self.boundary_dim = boundary_dim
        self.noise_dim = noise_dim
        self.pretrained_weights = pretrained_weights
        
        self.ssg_builder = SSGBuilder()
        self.segmenter = SemanticSegmenter(device=device if 'device' in locals() else "cpu")
        self.feature_extractor = None
        
    def _load_features(self):
        """Lazy load the feature extractor and move to device"""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.texture_dim, self.light_dim, self.boundary_dim, 
                load_pretrained_weights=self.pretrained_weights
            ).to(device)

    def purge_features(self):
        """Clean up the base encoders from RAM"""
        self.feature_extractor = None
        import gc
        gc.collect()

    def extract_ssg(self, image_pil):
        """
        Extract Spectral Semantic Graph (SSG) JSON and binary masks directly from image.
        Uses CLIPSeg for semantic enrichment.
        """
        self._load_features()
        
        # 1. Image preprocessing for feature extractor
        if not torch.is_tensor(image_pil):
            from torchvision import transforms as T
            transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
        else:
            input_tensor = image_pil

        # 2. Extract Latent Features
        zt, zl, zb, (tex_map, light_map, bound_map) = self.feature_extractor(input_tensor)
        
        # 3. Base Structural Construction via SSG Builder
        # (Assuming build_graph exists as per prev refactor)
        res_ssg_list, res_masks_list = self.ssg_builder.build_graph(tex_map, light_map, bound_map)
        ssg_root = res_ssg_list[0]
        masks = res_masks_list[0]

        # 4. Semantic Enrichment with CLIPSeg
        self.purge_features() # Free RAM
        
        semantic_prompts = ["person", "hair", "face", "clothing", "upper body", "eyes", "mouth", "background", "sky", "grass"]
        
        if self.segmenter is not None:
            # Convert to PIL for segmenter
            if torch.is_tensor(input_tensor):
                from torchvision import transforms as T2
                pil_for_seg = T2.ToPILImage()(input_tensor[0].cpu())
            else:
                pil_for_seg = image_pil
                
            semantic_masks = self.segmenter.segment(pil_for_seg, semantic_prompts)
            
            if semantic_masks is not None:
                H_img, W_img = input_tensor.shape[2], input_tensor.shape[3]
                total_px = float(H_img * W_img)
                semantic_nodes = []

                for i, label in enumerate(semantic_prompts):
                    mask_val = (semantic_masks[i, 0] > 0.25).float()
                    coverage = float(mask_val.sum()) / total_px
                    
                    if coverage > 0.01: # 1% threshold
                        s_key = f"semantic_{label}"
                        masks[s_key] = mask_val
                        
                        # BBox extraction
                        ys, xs = torch.where(mask_val > 0.3)
                        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                        
                        semantic_nodes.append({
                            "id": s_key,
                            "weight": round(coverage, 4),
                            "depth": 1,
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "attributes": {"label": label, "semantic": True},
                            "children": []
                        })
                
                # Inject into SSG Root
                ssg_root.setdefault("children", [])
                ssg_root["children"] = semantic_nodes + ssg_root["children"]
        
        # Ensure SCENE_ROOT exists for global edits
        masks["SCENE_ROOT"] = torch.ones(input_tensor.shape[2], input_tensor.shape[3], device=input_tensor.device)
        
        return ssg_root, masks
        
    def generate_from_features(self, z_texture, z_light, z_boundary, noise=None, use_turboquant=True):
        """Generate image from extracted features"""
        if use_turboquant:
            z_texture = tq_tex.decompress(tq_tex.compress(z_texture))
            z_light = tq_light.decompress(tq_light.compress(z_light))
            z_boundary = tq_bound.decompress(tq_bound.compress(z_boundary))
        return self.generator(z_texture, z_light, z_boundary, noise)
    
    def generate_from_image(self, reference_img, noise=None, use_turboquant=True):
        """Extract features from reference image and generate new image"""
        zt, zl, zb, (tex_map, light_map, bound_map) = self.feature_extractor(reference_img)
        
        if use_turboquant:
            zt = tq_tex.decompress(tq_tex.compress(zt))
            zl = tq_light.decompress(tq_light.compress(zl))
            zb = tq_bound.decompress(tq_bound.compress(zb))
            
        return self.generator(zt, zl, zb, noise, texture_map=tex_map, boundary_map=bound_map), (zt, zl, zb)
    
    def inpaint(self, masked_img, mask, reference_img=None, features=None, use_turboquant=True):
        """Inpaint masked region"""
        if features is None:
            if reference_img is None:
                raise ValueError("Either reference_img or features must be provided")
            zt, zl, zb, _ = self.feature_extractor(reference_img)
        else:
            zt, zl, zb = features
            
        if use_turboquant:
            zt = tq_tex.decompress(tq_tex.compress(zt))
            zl = tq_light.decompress(tq_light.compress(zl))
            zb = tq_bound.decompress(tq_bound.compress(zb))
        
        return self.inpainting_gen(masked_img, mask, zt, zl, zb)

# ==================== Training Functions ====================
def create_mask(batch_size, img_size=128, mask_type='random'):
    """Create masks for inpainting"""
    masks = torch.zeros(batch_size, 1, img_size, img_size, device=device)
    
    if mask_type == 'random':
        # Random rectangular masks
        for i in range(batch_size):
            h_start = torch.randint(0, img_size // 2, (1,)).item()
            w_start = torch.randint(0, img_size // 2, (1,)).item()
            h_size = torch.randint(img_size // 4, img_size // 2, (1,)).item()
            w_size = torch.randint(img_size // 4, img_size // 2, (1,)).item()
            masks[i, 0, h_start:h_start+h_size, w_start:w_start+w_size] = 1.0
    elif mask_type == 'center':
        # Center mask
        center = img_size // 2
        mask_size = img_size // 4
        masks[:, 0, center-mask_size:center+mask_size, center-mask_size:center+mask_size] = 1.0
    elif mask_type == 'bottom':
        # Bottom half mask
        masks[:, 0, img_size//2:, :] = 1.0
    
    return masks

def train_generation_model(model, discriminator, dataloader, epochs=50, lr_g=2e-4, lr_d=2e-4):
    """Train the generation model"""
    device = next(model.parameters()).device
    
    # Optimizers: train generator and feature extractor together if not frozen
    g_params = list(model.generator.parameters())
    if next(model.feature_extractor.parameters()).requires_grad:
        g_params += list(model.feature_extractor.parameters())
    opt_g = torch.optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_adv = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    # Use lighter perceptual loss on CPU to avoid GPU OOM
    criterion_perceptual = PerceptualLoss(use_cpu=True, backbone='vgg11', max_layers=18)
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    model.train()
    discriminator.train()
    
    for epoch in range(epochs):
        total_loss_g = 0
        total_loss_d = 0
        for batch_idx, (imgs, _) in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            # Extract features and real maps
            zt, zl, zb, (tex_real, light_real, bound_real) = model.feature_extractor(imgs)
            # Generate with spatial conditioning
            noise = torch.randn(batch_size, 100, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_imgs = model.generator(zt, zl, zb, noise, texture_map=tex_real, boundary_map=bound_real)
            # ========== Train Discriminator ==========
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                real_pred = discriminator(imgs)
                real_labels = torch.full_like(real_pred, 1.0)
                loss_d_real = criterion_adv(real_pred, real_labels)
                fake_pred = discriminator(fake_imgs.detach())
                fake_labels = torch.full_like(fake_pred, 0.0)
                loss_d_fake = criterion_adv(fake_pred, fake_labels)
                
                # Standard GAN discriminator loss
                loss_d = (loss_d_real + loss_d_fake) / 2.0
            
            # GATL: Check for vanishing gradients in Discriminator
            scaler_d.scale(loss_d).backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            if grad_norm_d > 1e-6:
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                # Vanishing gradients detected - skipping update
                opt_d.zero_grad(set_to_none=True)
                scaler_d.update() # Keep scaler in sync
            
            # ========== Train Generator ==========
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_pred = discriminator(fake_imgs)
                real_labels = torch.full_like(fake_pred, 1.0)
                loss_g_adv = criterion_adv(fake_pred, real_labels)  # adversarial component
                # Reconstruction components
                loss_l1 = criterion_l1(fake_imgs, imgs)
            # Perceptual component on CPU (outside autocast)
            loss_perc = criterion_perceptual(fake_imgs, imgs)
            # Final generator loss
            alpha_l1 = 100.0
            alpha_perc = 1.0
            loss_g = loss_g_adv + alpha_l1 * loss_l1 + alpha_perc * loss_perc
            
            # GATL: Check for vanishing gradients in Generator
            scaler_g.scale(loss_g).backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(g_params, max_norm=1.0)
            if grad_norm_g > 1e-6:
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                # Vanishing gradients detected - skipping update
                opt_g.zero_grad(set_to_none=True)
                scaler_g.update()
                
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            
        print(f"Epoch {epoch+1}/{epochs} - G Loss: {total_loss_g/len(dataloader):.4f}, D Loss: {total_loss_d/len(dataloader):.4f}")
    return model, discriminator

def train_inpainting_model(model, inpainting_disc, dataloader, epochs=30, lr_g=2e-4, lr_d=2e-4):
    """Train the inpainting model"""
    device = next(model.parameters()).device
    
    # Optimizers
    g_params = list(model.inpainting_gen.parameters())
    if next(model.feature_extractor.parameters()).requires_grad:
        g_params += list(model.feature_extractor.parameters())
    opt_g = torch.optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(inpainting_disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_adv = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    
    real_label = 1.0
    fake_label = 0.0
    
    model.train()
    inpainting_disc.train()
    
    for epoch in range(epochs):
        total_loss_g = 0
        total_loss_d = 0
        for batch_idx, (imgs, _) in enumerate(tqdm(dataloader, leave=False, desc=f"Inpaint Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            masks = create_mask(batch_size, imgs.size(2), mask_type='random')
            masked_imgs = imgs * (1 - masks.repeat(1, 3, 1, 1))
            # Extract features and real maps
            zt, zl, zb, (tex_real, light_real, bound_real) = model.feature_extractor(imgs)
            # Inpaint
            inpainted, generated = model.inpaint(masked_imgs, masks, features=(zt, zl, zb))
            # ========== Train Discriminator ==========
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                real_pred = inpainting_disc(imgs)
                real_labels = torch.full_like(real_pred, 1.0)
                loss_d_real = criterion_adv(real_pred, real_labels)
                fake_pred = inpainting_disc(inpainted.detach())
                fake_labels = torch.full_like(fake_pred, 0.0)
                loss_d_fake = criterion_adv(fake_pred, fake_labels)
                
                # Standard GAN discriminator loss
                loss_d = (loss_d_real + loss_d_fake) / 2.0
            
            # GATL
            scaler_d.scale(loss_d).backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(inpainting_disc.parameters(), max_norm=1.0)
            if grad_norm_d > 1e-6:
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                opt_d.zero_grad(set_to_none=True)
                scaler_d.update()
            
            # ========== Train Generator ==========
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_pred = inpainting_disc(inpainted)
                real_labels = torch.full_like(fake_pred, 1.0)
                loss_g_adv = criterion_adv(fake_pred, real_labels)
                loss_g_l1 = criterion_l1(inpainted, imgs)
                loss_g_perceptual = criterion_perceptual(inpainted, imgs)
                
                # Light consistency (mask-aware)
                tex_fake, light_fake, bound_fake = preprocess(inpainted)
                bound_mask = bound_real.mean(dim=1, keepdim=True)
                light_fake_1ch = light_fake.mean(dim=1, keepdim=True)
                light_real_1ch = light_real.mean(dim=1, keepdim=True)
                
                loss_light_outside = F.l1_loss(light_fake_1ch * (1 - bound_mask), torch.zeros_like(light_fake_1ch))
                loss_light_inside = F.l1_loss(light_fake_1ch * bound_mask, light_real_1ch * bound_mask)
                
                loss_g = (
                    loss_g_adv
                    + 100 * loss_g_l1
                    + 10 * loss_g_perceptual
                    + 5.0 * loss_light_outside
                    + 1.0 * loss_light_inside
                )
            
            # GATL
            scaler_g.scale(loss_g).backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(g_params, max_norm=1.0)
            if grad_norm_g > 1e-6:
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                opt_g.zero_grad(set_to_none=True)
                scaler_g.update()
                
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            
        print(f"Epoch {epoch+1}/{epochs} - G Loss: {total_loss_g/len(dataloader):.4f}, D Loss: {total_loss_d/len(dataloader):.4f}")
    return model, inpainting_disc

# ==================== Utility Functions ====================
def load_trained_model(checkpoint_path, device=None):
    """Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ImageGenerationModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def generate_from_reference(model, reference_img, noise=None):
    """Generate image from reference image
    
    Args:
        model: Trained ImageGenerationModel
        reference_img: Reference image tensor [1, 3, H, W]
        noise: Optional noise vector [1, 100]
    
    Returns:
        Generated image tensor
    """
    model.eval()
    with torch.no_grad():
        generated, features = model.generate_from_image(reference_img, noise)
    return generated, features

def inpaint_image(model, image, mask, reference_img=None):
    """Inpaint masked region in image
    
    Args:
        model: Trained ImageGenerationModel
        image: Image with masked region [1, 3, H, W]
        mask: Binary mask [1, 1, H, W] (1 = masked, 0 = visible)
        reference_img: Optional reference image for features
    
    Returns:
        Inpainted image tensor
    """
    model.eval()
    with torch.no_grad():
        inpainted, generated = model.inpaint(image, mask, reference_img)
    return inpainted, generated

# ==================== Visualization ====================
def visualize_results(model, test_imgs, num_samples=4):
    """Visualize generation and inpainting results"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        test_imgs = test_imgs[:num_samples].to(device)
        zt, zl, zb, (tex_map, light_map, bound_map) = model.feature_extractor(test_imgs)
        noise = torch.randn(num_samples, 100, device=device)
        generated = model.generator(zt, zl, zb, noise, texture_map=tex_map, boundary_map=bound_map)
        
        # Inpainting
        masks = create_mask(num_samples, test_imgs.size(2), mask_type='center')
        masked_imgs = test_imgs * (1 - masks.repeat(1, 3, 1, 1))
        inpainted, _ = model.inpaint(masked_imgs, masks, features=(zt, zl, zb))
        
        # Plot
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
        for i in range(num_samples):
            axes[i, 0].imshow(test_imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(generated[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title('Generated')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(masked_imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 2].set_title('Masked')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(inpainted[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 3].set_title('Inpainted')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.show()

# ==================== Main Execution ====================
if __name__ == '__main__':
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=False, num_workers=0
    )
    
    # Configuration
    PRETRAINED_PREPROCESSING_WEIGHTS = None  # Set to path if you have trained preprocessing model weights
    LOAD_CHECKPOINT = None  # Set to path to resume training from checkpoint
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    model = ImageGenerationModel(pretrained_weights=PRETRAINED_PREPROCESSING_WEIGHTS).to(device)
    discriminator = PatchDiscriminator().to(device)
    inpainting_disc = PatchDiscriminator().to(device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
        print(f"Loading checkpoint from {LOAD_CHECKPOINT}...")
        checkpoint = torch.load(LOAD_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'inpainting_disc_state_dict' in checkpoint:
            inpainting_disc.load_state_dict(checkpoint['inpainting_disc_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    print("=" * 60)
    print("Training Image Generation Model")
    print("=" * 60)
    
    # Train generation model
    model, discriminator = train_generation_model(
        model, discriminator, train_loader, epochs=20, lr_g=2e-4, lr_d=2e-4
    )
    
    # Save checkpoint after generation training
    checkpoint_path = os.path.join(SAVE_DIR, 'generation_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': 20
    }, checkpoint_path)
    print(f"Generation checkpoint saved to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Inpainting Model")
    print("=" * 60)
    
    # Train inpainting model
    model, inpainting_disc = train_inpainting_model(
        model, inpainting_disc, train_loader, epochs=15, lr_g=2e-4, lr_d=2e-4
    )
    # BEGAN-style equilibrium parameters to balance G and D
    k_t = 0.0        # start with no discount on fake loss
    gamma = 0.5      # target balance (adjust 0.3-0.7 as needed)
    lambda_k = 1e-3  # update rate (try 5e-4 to 2e-3)
    
    # Save final models
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'inpainting_disc_state_dict': inpainting_disc.state_dict(),
        'epoch': 35
    }
    
    torch.save(final_checkpoint, os.path.join(SAVE_DIR, 'final_model.pth'))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'generation_model.pth'))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator.pth'))
    torch.save(inpainting_disc.state_dict(), os.path.join(SAVE_DIR, 'inpainting_discriminator.pth'))
    print(f"\nModels saved to {SAVE_DIR}!")
    
    # Visualize results
    print("\nGenerating visualization...")
    test_imgs, _ = next(iter(test_loader))
    visualize_results(model, test_imgs, num_samples=4)
    
    print("\nTraining complete!")
    print(f"To load the model later, use:")
    print(f"  model = ImageGenerationModel().to(device)")
    print(f"  model.load_state_dict(torch.load('{os.path.join(SAVE_DIR, 'generation_model.pth')}'))")

