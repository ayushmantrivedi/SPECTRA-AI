"""
Dynamic Orchestrator
====================
Tree-DP multi-node edit scheduler with live HSG re-indexing between hops.

Key design:
- Accepts the LLM-session masks_cache as authoritative source (no re-extraction
  before first edit, eliminating the ID race condition).
- Spatial-match fallback with IoU > 0.01 (was 0.1) for sub-pixel drift tolerance.
- Direct semantic key lookup before IoU search (semantic_person etc.).
- Re-extraction happens ONLY after a successful edit hop, to track visual changes.
"""
import gc
import torch
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


def _find_closest_node(target_id, target_bbox, masks_cache, hsg_flat):
    """
    Resolve a node ID to a mask tensor.
    Priority order:
      1. Exact ID match in masks_cache (O(1))
      2. Direct semantic prefix match (semantic_person etc.)
      3. Spatial IoU fallback (threshold lowered to 0.01 for sub-pixel drift)
    Returns (node_id, mask_tensor) or (None, None).
    """
    # 1. Exact match
    if target_id in masks_cache:
        return target_id, masks_cache[target_id]

    # 2. Semantic prefix match — e.g. "semantic_person" or partial "person"
    for key in masks_cache:
        if key.startswith("semantic_") and (
            key == target_id or
            target_id in key or
            key.replace("semantic_", "") in target_id
        ):
            print(f"[ORCH] Semantic match: '{target_id}' → '{key}'")
            return key, masks_cache[key]

    # 3. Spatial IoU fallback
    if not target_bbox:
        return None, None

    best_id, best_iou = None, 0.0
    for node in hsg_flat:
        nid  = node.get("id", "")
        bbox = node.get("bbox", [])
        if nid in masks_cache:
            iou = _bbox_iou(target_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_id  = nid

    # Lowered from 0.1 → 0.01 to handle sub-pixel contour drift
    if best_id and best_iou > 0.01:
        return best_id, masks_cache[best_id]

    return None, None


def _flatten_hsg(node, out=None):
    if out is None:
        out = []
    out.append(node)
    for c in node.get("children", []):
        _flatten_ssg(c, out)
    return out


class DynamicOrchestrator:
    def __init__(self, model, ssg_tracker: SpectralSyncTracker):
        self.model   = model
        self.ssg_tracker = ssg_tracker
        self.diffusion_module = KernelDiffusionModule(model)

    def _dp_schedule(self, edits):
        """
        DP sort: semantic edits first, then by max influence.
        Semantic edits always take priority as they cover larger regions.
        """
        def sort_key(e):
            is_semantic = e["target_node"].startswith("semantic_")
            max_inf = max(e["influence"].values()) if e["influence"] else 0
            return (int(is_semantic), max_inf)
        return sorted(edits, key=sort_key, reverse=True)

    def execute_edit_schedule(self, full_img, initial_ssg, edits,
                              initial_masks_cache=None):
        """
        Executes a sequence of edits with live mid-hop SSG re-extraction.

        Args:
            full_img: Input image tensor [1,3,H,W]
            initial_ssg: SSG dict from the LLM session (authoritative node IDs)
            edits: List of edit dicts from LLM parser
            initial_masks_cache: masks_cache from the LLM-session extraction.
                                 If provided, used directly without re-extraction.
                                 This eliminates the ID race condition (BUG #2).

        Returns: (final_image, final_adapted_ssg, execution_log)
        """
        current_img = full_img

        if initial_masks_cache is not None:
            # Use the authoritative LLM-session masks — no re-extraction!
            masks_cache = initial_masks_cache
            print(f"[ORCH] Using {len(masks_cache)} authoritative masks from session.")
            with torch.no_grad():
                current_ssg_dict, _, base_features = self.model.extract_ssg(current_img)
        else:
            # Backward-compatible path
            with torch.no_grad():
                current_ssg_dict, masks_cache, base_features = self.model.extract_ssg(current_img)

        # Build session_node_info from the LLM-facing SSG for spatial fallback
        session_node_info = {n["id"]: n.get("bbox") for n in _flatten_ssg(initial_ssg)}

        schedule = self._dp_schedule(edits)
        exec_log = []

        for step, edit in enumerate(schedule):
            raw_target = edit["target_node"]
            influence  = dict(edit["influence"])  # copy so mutation is local
            intent     = edit.get("intent", "")
            if intent:
                influence["intent"] = intent

            target_bbox = session_node_info.get(raw_target)
            ssg_flat    = _flatten_ssg(current_ssg_dict)

            resolved_id, mask_tensor = _find_closest_node(
                raw_target, target_bbox, masks_cache, ssg_flat
            )

            if resolved_id is None:
                msg = f"[HOP {step+1}] SKIP — node '{raw_target}' not resolvable"
                print(f"  {msg}")
                exec_log.append(msg)
                continue

            if resolved_id != raw_target:
                exec_log.append(
                    f"[HOP {step+1}] Matched '{raw_target}' → '{resolved_id}'"
                )

            # SURGICAL MASK ALGEBRA: Protect the face!
            # If target is hair or clothing, subtract face/eyes to ensure identity preservation.
            if "semantic_hair" in resolved_id or "semantic_clothing" in resolved_id or "semantic_upper body" in resolved_id:
                face_m = masks_cache.get("semantic_face")
                eyes_m = masks_cache.get("semantic_eyes")
                mouth_m = masks_cache.get("semantic_mouth")
                
                orig_sum = mask_tensor.sum().item()
                
                # Subtract face regions
                for protect_key in ["semantic_face", "semantic_eyes", "semantic_mouth"]:
                    p_mask = masks_cache.get(protect_key)
                    if p_mask is not None:
                        # Force same device and shape
                        p_mask = p_mask.to(mask_tensor.device)
                        if p_mask.shape != mask_tensor.shape:
                            if p_mask.dim() == 2: p_mask = p_mask.unsqueeze(0).unsqueeze(0)
                        
                        mask_tensor = torch.clamp(mask_tensor - p_mask, 0, 1)
                
                new_sum = mask_tensor.sum().item()
                if new_sum < orig_sum:
                    print(f"[ORCH] Surgical Isolation: Protected face region removed from {resolved_id} mask.")
                
                # STRICT HYGIENE: Force clean binary mask for FP32 VAE stability
                mask_tensor = (mask_tensor > 0.5).float()

            # Normalise mask to [1,1,H,W]
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)

            mask_sum = mask_tensor.sum().item()
            print(f"[ORCH] Executing '{resolved_id}' | Mask coverage: "
                  f"{mask_sum:.0f}px ({100*mask_sum/(mask_tensor.numel()/mask_tensor.shape[1]):.1f}%)")

            self.ssg_tracker.record_edit(resolved_id)

            current_img = self.diffusion_module.run_diffusion_edit(
                current_img, base_features, resolved_id, influence, mask_tensor
            )

            exec_log.append(
                f"[HOP {step+1}] '{resolved_id}' | "
                f"ZT={influence.get('ZT',0)} ZL={influence.get('ZL',0)} "
                f"ZB={influence.get('ZB',0)} | {intent}"
            )

            # Reclaim memory between hops
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-extract SSG AFTER the edit so subsequent hops see updated image
            with torch.no_grad():
                current_ssg_dict, masks_cache, base_features = self.model.extract_ssg(current_img)

        adapted_ssg = self.ssg_tracker.update_weights(current_ssg_dict)
        return current_img, adapted_ssg, exec_log
