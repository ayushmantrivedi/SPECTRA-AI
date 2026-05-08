import pytest
import numpy as np
import torch
from kernel_diffusion import KernelDiffusionModule

def test_kernel_diffusion_initialization():
    module = KernelDiffusionModule()
    assert module is not None

def test_safe_to_pil_mask():
    # Test tensor to PIL mask conversion
    module = KernelDiffusionModule()
    mask_tensor = torch.zeros(1, 1, 64, 64)
    mask_tensor[0, 0, 10:20, 10:20] = 1.0
    
    pil_mask = module._safe_to_pil_mask(mask_tensor)
    assert pil_mask is not None
    assert pil_mask.size == (64, 64)

def test_dilate_mask():
    module = KernelDiffusionModule()
    mask_tensor = torch.zeros(1, 1, 64, 64)
    mask_tensor[0, 0, 32, 32] = 1.0
    
    dilated = module.dilate_mask(mask_tensor, kernel_size=5)
    assert dilated is not None
    assert dilated.shape == (1, 1, 64, 64)
    # The center pixel and its neighbors should be 1.0
    assert dilated[0, 0, 32, 32] == 1.0
    assert dilated[0, 0, 30, 30] == 1.0
