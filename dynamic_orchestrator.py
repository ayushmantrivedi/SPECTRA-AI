"""
SPECTRA Dynamic Orchestrator
============================
Tree-based multi-hop edit scheduler with live SSG re-indexing.
"""

import gc
import torch
import numpy as np
from kernel_diffusion import KernelDiffusionModule
from spectral_sync import SpectralSyncTracker

def _bbox_iou(b1, b2):
    """Intersection over Union for two [x,y,w,h] bboxes."""
    if not b1 or not b2:
        return 0.0
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = b1[2]*b1[3];  area2 = b2[2]*b2[3]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def _flatten_ssg(node, out=None):
    """Recursively flattens the SSG tree into a list of nodes."""
    if out is None:
        out = []
    out.append(node)
    for c in node.get("children", []):
        _flatten_ssg(c, out)
    return out

def _find_closest_node(target_id, target_bbox, masks_cache, ssg_flat):
    """
    Resolves a requested node ID to a physical mask.
    Matches by exact ID, semantic prefix, or spatial IoU.
    """
    if target_id in masks_cache:
        return target_id, masks_cache[target_id]

    # Semantic prefix match
    for key in masks_cache:
        if key.startswith("semantic_") and (target_id in key or key.replace("semantic_", "") in target_id):
            return key, masks_cache[key]

    # Spatial fallback
    if not target_bbox:
        return None, None

    best_id, best_iou = None, 0.0
    for node in ssg_flat:
        nid = node.get("id", "")
        bbox = node.get("bbox", [])
        if nid in masks_cache:
            iou = _bbox_iou(target_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = nid

    if best_id and best_iou > 0.01:
        return best_id, masks_cache[best_id]

    return None, None

class DynamicOrchestrator:
    def __init__(self, model, ssg_tracker: SpectralSyncTracker):
        self.model = model
        self.ssg_tracker = ssg_tracker
        self.diffusion_module = KernelDiffusionModule(model)

    def execute_edit_schedule(self, image_pil, initial_ssg, edits, initial_masks_cache=None):
        """
        Executes a sequence of edits with live SSG re-extraction.
        """
        current_img = image_pil
        masks_cache = initial_masks_cache
        
        # Initial state extraction
        with torch.no_grad():
            current_ssg_dict, masks_cache_new, base_features = self.model.extract_ssg(current_img)
            # Merge caches if needed
            if masks_cache is None: 
                masks_cache = masks_cache_new

        session_node_info = {n["id"]: n.get("bbox") for n in _flatten_ssg(initial_ssg)}
        exec_log = []

        for step, edit in enumerate(edits):
            raw_target = edit["target_node"]
            influence = dict(edit["influence"])
            intent = edit.get("intent", "")
            
            target_bbox = session_node_info.get(raw_target)
            ssg_flat = _flatten_ssg(current_ssg_dict)

            resolved_id, mask_tensor = _find_closest_node(raw_target, target_bbox, masks_cache, ssg_flat)

            if resolved_id is None:
                print(f"[ORCH] Skip '{raw_target}' - not found.")
                continue

            # SURGICAL MASK ALGEBRA
            if any(k in resolved_id for k in ["hair", "clothing", "upper body"]):
                for protect in ["semantic_face", "semantic_eyes"]:
                    p_mask = masks_cache.get(protect)
                    if p_mask is not None:
                        mask_tensor = torch.clamp(mask_tensor - p_mask.to(mask_tensor.device), 0, 1)
                mask_tensor = (mask_tensor > 0.5).float()

            self.ssg_tracker.record_edit(resolved_id)
            
            # Execute Diffusion with active base features
            current_img = self.diffusion_module.run_diffusion_edit(
                current_img, base_features, resolved_id, influence, mask_tensor
            )
            
            exec_log.append(f"Hop {step+1}: {resolved_id} edited.")
            
            # Re-extract SSG AFTER the edit so subsequent hops see updated image
            with torch.no_grad():
                current_ssg_dict, masks_cache, base_features = self.model.extract_ssg(current_img)

        final_ssg = self.ssg_tracker.update_weights(current_ssg_dict)
        return current_img, final_ssg, exec_log
