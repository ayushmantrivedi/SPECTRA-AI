import torch
import numpy as np
import cv2
import heapq


class SpectralSceneNode:
    """
    Represents a node in the Spectral Semantic Graph (SSG).
    Each node encapsulates a semantic region, its latent influence weights, 
    and its spatial hierarchy.
    """
    def __init__(self, node_id, weight=0.0, bbox=None):
        self.id = node_id
        self.weight = weight
        self.depth = 0
        self.bbox = bbox  # [x, y, w, h]
        self.children = []
        self.attributes = {}
        self._tie_breaker = np.random.rand()

    def __lt__(self, other):
        if self.weight == other.weight:
            return self._tie_breaker < other._tie_breaker
        return self.weight < other.weight

    def to_dict(self):
        """Recursively serialize to JSON format"""
        clean_attrs = {}
        for k, v in self.attributes.items():
            if isinstance(v, list):
                clean_attrs[k] = [round(float(x), 4) for x in v]
            elif isinstance(v, float):
                clean_attrs[k] = round(v, 4)
            else:
                clean_attrs[k] = v
        return {
            "id": self.id,
            "weight": round(float(self.weight), 4),
            "depth": self.depth,
            "bbox": self.bbox,
            "attributes": clean_attrs,
            **( {"children": [c.to_dict() for c in self.children]} if self.children else {} )
        }


def union_bbox(b1, b2):
    """Compute the bounding box that encompasses two bounding boxes"""
    if not b1: return b2
    if not b2: return b1
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[0] + b1[2], b2[0] + b2[2])
    y2 = max(b1[1] + b1[3], b2[1] + b2[3])
    return [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]


class SSGBuilder:
    """Constructs the Spectral Semantic Graph structure bottom-up."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _assign_depth(self, node, depth=0):
        node.depth = depth
        for child in node.children:
            self._assign_depth(child, depth + 1)

    def _sort_tree(self, node):
        node.children.sort(key=lambda x: x.weight, reverse=True)
        for child in node.children:
            self._sort_tree(child)

    def build_graph(self, tex_map, light_map, bound_map):
        """
        Builds the adaptive Spectral hierarchy using spatial coverage and saliency priorities.
        Node IDs are spatially stable: leaf_<x>_<y> so they survive pixel edits.
        """
        B = bound_map.size(0)
        hsgs = []
        masks_cache = []

        for i in range(B):
            bm_np = bound_map[i, 0].detach().cpu().numpy()
            img_h, img_w = bm_np.shape
            total_area = img_h * img_w

            binary_mask = (bm_np > self.threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            leaf_nodes = []
            batch_masks = {}

            # Step A: Leaf extraction with STABLE spatial IDs
            for contour in contours:
                node_mask = np.zeros_like(binary_mask)
                cv2.drawContours(node_mask, [contour], -1, 1, thickness=cv2.FILLED)

                area = node_mask.sum()
                if area < 10:
                    continue

                node_mask_tensor = torch.from_numpy(node_mask).float().to(tex_map.device)

                tex_i   = tex_map[i]   * node_mask_tensor.unsqueeze(0)
                light_i = light_map[i] * node_mask_tensor.unsqueeze(0)

                avg_tex   = tex_i.sum(dim=(1, 2))   / area
                avg_light = light_i.sum(dim=(1, 2)) / area

                x, y, w, h = cv2.boundingRect(contour)

                # Basic spatial location tag
                loc_y = "top" if y < img_h*0.33 else ("bottom" if y + h > img_h*0.66 else "center")
                loc_x = "left" if x < img_w*0.33 else ("right" if x + w > img_w*0.66 else "center")
                
                # Size tag
                area_ratio = float(area / total_area)
                node_size = "massive" if area_ratio > 0.25 else ("large" if area_ratio > 0.1 else ("medium" if area_ratio > 0.02 else "tiny"))

                # Saliency: area + central proximity
                center_x, center_y = img_w / 2, img_h / 2
                obj_cx, obj_cy     = x + w / 2, y + h / 2
                dist_to_center     = np.sqrt((obj_cx - center_x) ** 2 + (obj_cy - center_y) ** 2)
                max_dist           = np.sqrt(center_x ** 2 + center_y ** 2)
                proximity_score    = 1.0 - (dist_to_center / (max_dist + 1e-5))

                area_weight    = float(area / total_area)
                priority_weight = (0.7 * area_weight) + (0.3 * proximity_score)

                # === STABLE SPATIAL ID ===
                node_id = f"leaf_{int(x)}_{int(y)}"
                # Deduplicate in the rare case two contours share the same top-left
                while node_id in batch_masks:
                    node_id = node_id + "_"

                node = SpectralSceneNode(node_id, weight=priority_weight,
                                        bbox=[int(x), int(y), int(w), int(h)])
                node.attributes = {
                    "texture_vector_sample": avg_tex[:5].detach().cpu().tolist(),
                    "light_intensity": float(avg_light.mean().item()),
                    "size": node_size,
                    "location": f"{loc_y}_{loc_x}"
                }

                leaf_nodes.append(node)
                batch_masks[node_id] = node_mask_tensor

            # Step B: Spectral Merge Algorithm (Bottom-Up)
            if not leaf_nodes:
                # Full-image fallback
                full_mask = torch.ones(img_h, img_w, device=tex_map.device)
                batch_masks["SCENE_ROOT"] = full_mask
                hsgs.append({
                    "id": "SCENE_ROOT", "depth": 0,
                    "bbox": [0, 0, img_w, img_h],
                    "weight": 1.0, "attributes": {}
                })
                masks_cache.append(batch_masks)
                continue

            heapq.heapify(leaf_nodes)
            merge_idx = 0

            while len(leaf_nodes) > 1:
                child1 = heapq.heappop(leaf_nodes)
                child2 = heapq.heappop(leaf_nodes)

                parent = SpectralSceneNode(
                    node_id=f"macro_{int(child1.bbox[0])}_{int(child1.bbox[1])}",
                    weight=child1.weight + child2.weight
                )
                parent.bbox = union_bbox(child1.bbox, child2.bbox)
                parent.children = [child1, child2]

                # Parent tags
                p_x, p_y, p_w, p_h = parent.bbox
                p_area_ratio = (p_w * p_h) / total_area
                p_size = "massive" if p_area_ratio > 0.3 else ("large" if p_area_ratio > 0.1 else "medium")
                p_loc = "center" # macro nodes tend to be general
                
                c1_light = child1.attributes.get("light_intensity", 0)
                c2_light = child2.attributes.get("light_intensity", 0)
                parent.attributes = {
                    "macro_light_intensity": float((c1_light + c2_light) / 2.0),
                    "size": p_size,
                    "location": p_loc
                }

                mask1 = batch_masks[child1.id]
                mask2 = batch_masks[child2.id]
                batch_masks[parent.id] = torch.clamp(mask1 + mask2, 0, 1)

                heapq.heappush(leaf_nodes, parent)
                merge_idx += 1

            # Step C: Top-Down Restructuring
            root = heapq.heappop(leaf_nodes)
            batch_masks["SCENE_ROOT"] = batch_masks.pop(root.id)
            root.id = "SCENE_ROOT"

            self._assign_depth(root)
            self._sort_tree(root)

            hsgs.append(root.to_dict())
            masks_cache.append(batch_masks)

        return hsgs, masks_cache
