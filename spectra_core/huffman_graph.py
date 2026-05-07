"""
SPECTRA Huffman Scene Graph (HSG)
===================================
Hierarchical tree structure inspired by Huffman coding.
Higher-weight nodes live closer to the root (fast access).
Lower-weight leaf nodes are lazily loaded / compressed.

Weight = f(visual_saliency, semantic_importance, edit_frequency, spatial_coverage)

Graph Edges provide dependency tracking (separate overlay on top of the tree structure).
"""

from __future__ import annotations

import json
import heapq
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ── Node ─────────────────────────────────────────────────────────────────────


@dataclass
class HSGNode:
    """A single node in the Huffman Scene Graph."""

    id: str
    label: str = ""
    weight: float = 0.5
    depth: int = 0
    bbox: Optional[List[int]] = None          # [x, y, w, h]
    mask_key: Optional[str] = None            # key in masks_cache dict
    identity_vector: Optional[List[float]] = None   # face / clip embedding
    feature_vector: Optional[List[float]] = None    # CLIP / DINO embedding
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["HSGNode"] = field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Adaptive learning state
    edit_count: int = 0
    _tie_breaker: float = field(default_factory=np.random.rand)

    # ── Heap ordering (min-heap by weight for Huffman merge) ─────────────────
    def __lt__(self, other: "HSGNode") -> bool:
        if self.weight == other.weight:
            return self._tie_breaker < other._tie_breaker
        return self.weight < other.weight

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self, depth_limit: Optional[int] = None) -> Dict[str, Any]:
        """Recursively serialise to a JSON-compatible dict.

        Args:
            depth_limit: If set, nodes beyond this depth are represented as
                         seed-only stubs (Huffman leaf compression).
        """
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "weight": round(float(self.weight), 4),
            "depth": self.depth,
            "bbox": self.bbox,
            "edit_count": self.edit_count,
        }

        # Selective attribute inclusion by depth (lazy-load simulation)
        if depth_limit is None or self.depth <= depth_limit:
            d["attributes"] = self._clean_attributes()
            if self.identity_vector is not None:
                d["identity_vector"] = [round(float(v), 6) for v in self.identity_vector[:8]]
            if self.feature_vector is not None:
                d["feature_vector"] = [round(float(v), 6) for v in self.feature_vector[:8]]
        else:
            # Seed-only stub for deep nodes
            d["attributes"] = {"stub": True, "label": self.label}

        if self.graph_edges:
            d["graph_edges"] = self.graph_edges

        if self.children:
            d["children"] = [
                c.to_dict(depth_limit=depth_limit) for c in self.children
            ]

        return d

    def _clean_attributes(self) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in self.attributes.items():
            if isinstance(v, list):
                clean[k] = [round(float(x), 4) if isinstance(x, float) else x for x in v]
            elif isinstance(v, float):
                clean[k] = round(v, 4)
            else:
                clean[k] = v
        return clean

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HSGNode":
        """Deserialise a node (and its subtree) from a dict."""
        node = cls(
            id=d.get("id", "unknown"),
            label=d.get("label", d.get("id", "unknown")),
            weight=d.get("weight", 0.5),
            depth=d.get("depth", 0),
            bbox=d.get("bbox"),
            attributes=d.get("attributes", {}),
            graph_edges=d.get("graph_edges", []),
            identity_vector=d.get("identity_vector"),
            feature_vector=d.get("feature_vector"),
            edit_count=d.get("edit_count", 0),
        )
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node


# ── Huffman Scene Graph ───────────────────────────────────────────────────────


class HuffmanSceneGraph:
    """
    Full Huffman Scene Graph with:
    - Priority-based tree structure (weight-sorted)
    - Graph edge overlay for dependency tracking
    - Adaptive rebalancing (edit-frequency learning)
    - Depth-based compression (leaf nodes are seed-only stubs)
    """

    SALIENCY_WEIGHT = 0.30
    IMPORTANCE_WEIGHT = 0.25
    EDIT_FREQ_WEIGHT = 0.25
    SPATIAL_WEIGHT = 0.20

    def __init__(self):
        self.root: Optional[HSGNode] = None
        self._node_index: Dict[str, HSGNode] = {}  # id → node (fast lookup)
        self._edit_counts: Dict[str, int] = defaultdict(int)
        self._total_edits: int = 0
        # Graph edges stored separately for O(1) lookup
        self._edge_index: Dict[str, List[Dict]] = defaultdict(list)

    # ── Construction ─────────────────────────────────────────────────────────

    def build_from_orchestrator_output(self, scene_dict: Dict[str, Any]) -> None:
        """
        Ingest a scene_dict from the VLM orchestrator and build the full HSG.
        The input is expected to follow the HSG JSON schema from Part 2.
        """
        self.root = HSGNode.from_dict(scene_dict)
        self._rebuild_index(self.root)
        self._rebuild_edge_index()
        self._sort_tree(self.root)
        self._assign_depth(self.root, 0)

    def build_from_ssg_dict(
        self,
        ssg_dict: Dict[str, Any],
        semantic_coverage: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Build HSG from the legacy SSGBuilder output (backward compatibility).
        Computes node weights using the Huffman formula.
        """
        self.root = self._convert_ssg_node(ssg_dict, semantic_coverage or {})
        self._rebuild_index(self.root)
        self._sort_tree(self.root)
        self._assign_depth(self.root, 0)

    def _convert_ssg_node(
        self,
        d: Dict[str, Any],
        semantic_coverage: Dict[str, float],
        depth: int = 0,
    ) -> HSGNode:
        node_id = d.get("id", "unknown")
        coverage = semantic_coverage.get(node_id, d.get("weight", 0.3))

        # HSG weight formula
        visual_saliency = min(1.0, coverage * 3.0)
        semantic_importance = 0.9 if "semantic_" in node_id else 0.5
        edit_freq = self._edit_counts.get(node_id, 0) / max(1, self._total_edits)
        spatial_coverage = coverage

        weight = (
            self.SALIENCY_WEIGHT * visual_saliency
            + self.IMPORTANCE_WEIGHT * semantic_importance
            + self.EDIT_FREQ_WEIGHT * edit_freq
            + self.SPATIAL_WEIGHT * spatial_coverage
        )

        node = HSGNode(
            id=node_id,
            label=d.get("attributes", {}).get("label", node_id),
            weight=min(1.0, weight),
            depth=depth,
            bbox=d.get("bbox"),
            attributes=d.get("attributes", {}),
            edit_count=self._edit_counts.get(node_id, 0),
        )
        node.children = [
            self._convert_ssg_node(c, semantic_coverage, depth + 1)
            for c in d.get("children", [])
        ]
        return node

    # ── Lookup ────────────────────────────────────────────────────────────────

    def find_node(self, node_id: str) -> Optional[HSGNode]:
        """O(1) node lookup by ID."""
        return self._node_index.get(node_id)

    def find_nodes_by_label(self, label: str) -> List[HSGNode]:
        """Find all nodes whose label contains the given string (case-insensitive)."""
        q = label.lower()
        return [n for n in self._node_index.values() if q in n.label.lower() or q in n.id.lower()]

    def get_traversal_path(self, target_id: str) -> List[str]:
        """Return the ancestor chain from root → target_id."""
        path: List[str] = []
        self._dfs_path(self.root, target_id, path)
        return path

    def _dfs_path(self, node: Optional[HSGNode], target: str, path: List[str]) -> bool:
        if node is None:
            return False
        path.append(node.id)
        if node.id == target:
            return True
        for child in node.children:
            if self._dfs_path(child, target, path):
                return True
        path.pop()
        return False

    # ── Adaptive Rebalancing ──────────────────────────────────────────────────

    def record_edit(self, node_id: str, decay_lambda: float = 0.7) -> None:
        """
        Update node weight after a successful edit (Adaptive Huffman).
        Frequently edited nodes bubble up toward the root.
        """
        self._edit_counts[node_id] += 1
        self._total_edits += 1

        node = self._node_index.get(node_id)
        if node is None:
            return

        node.edit_count = self._edit_counts[node_id]
        frequency_signal = self._edit_counts[node_id] / self._total_edits
        node.weight = min(1.0, decay_lambda * node.weight + (1 - decay_lambda) * frequency_signal)

    def rebalance(self) -> None:
        """Re-sort the entire tree by current weights (O(n log n))."""
        if self.root:
            self._sort_tree(self.root)
            self._assign_depth(self.root, 0)

    # ── Graph Edges ───────────────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str, relation: str, strength: float = 1.0) -> None:
        edge = {"target": target_id, "relation": relation, "strength": strength}
        self._edge_index[source_id].append(edge)
        src = self._node_index.get(source_id)
        if src:
            src.graph_edges.append(edge)

    def get_dependencies(self, node_id: str) -> List[Dict[str, Any]]:
        """Return all edges originating from this node."""
        return self._edge_index.get(node_id, [])

    def get_cascade_targets(self, node_id: str) -> List[str]:
        """Return IDs of all nodes that depend on this one."""
        return [e["target"] for e in self._edge_index.get(node_id, [])]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self, depth_limit: Optional[int] = None) -> Dict[str, Any]:
        if self.root is None:
            return {}
        return self.root.to_dict(depth_limit=depth_limit)

    def to_json(self, indent: int = 2, depth_limit: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(depth_limit=depth_limit), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HuffmanSceneGraph":
        hsg = cls()
        hsg.build_from_orchestrator_output(d)
        return hsg

    @classmethod
    def from_json(cls, s: str) -> "HuffmanSceneGraph":
        return cls.from_dict(json.loads(s))

    # ── Stats & Utilities ─────────────────────────────────────────────────────

    def flatten(self) -> List[HSGNode]:
        """Return all nodes as a flat list."""
        out: List[HSGNode] = []
        self._flatten_recursive(self.root, out)
        return out

    def get_frozen_mask_keys(self, edit_target_ids: List[str]) -> List[str]:
        """
        Return mask keys for all nodes NOT in edit_target_ids.
        Used by the execution engine to identify frozen regions.
        """
        all_ids = set(self._node_index.keys())
        edit_ids = set(edit_target_ids)
        return [n.mask_key for n in self._node_index.values()
                if n.id not in edit_ids and n.mask_key is not None]

    def get_priority_ranking(self) -> List[Tuple[str, float]]:
        return sorted(
            [(n.id, n.weight) for n in self._node_index.values()],
            key=lambda x: x[1],
            reverse=True,
        )

    def stats(self) -> Dict[str, Any]:
        nodes = self.flatten()
        return {
            "total_nodes": len(nodes),
            "max_depth": max((n.depth for n in nodes), default=0),
            "total_edits": self._total_edits,
            "most_edited": sorted(self._edit_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_weight_nodes": self.get_priority_ranking()[:5],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rebuild_index(self, node: Optional[HSGNode]) -> None:
        if node is None:
            return
        self._node_index[node.id] = node
        for child in node.children:
            self._rebuild_index(child)

    def _rebuild_edge_index(self) -> None:
        self._edge_index.clear()
        for node in self._node_index.values():
            for edge in node.graph_edges:
                self._edge_index[node.id].append(edge)

    def _sort_tree(self, node: HSGNode) -> None:
        node.children.sort(key=lambda x: x.weight, reverse=True)
        for child in node.children:
            self._sort_tree(child)

    def _assign_depth(self, node: HSGNode, depth: int) -> None:
        node.depth = depth
        for child in node.children:
            self._assign_depth(child, depth + 1)

    def _flatten_recursive(self, node: Optional[HSGNode], out: List[HSGNode]) -> None:
        if node is None:
            return
        out.append(node)
        for child in node.children:
            self._flatten_recursive(child, out)


# ── Adaptive Rebalancer (standalone helper) ───────────────────────────────────


class AdaptiveRebalancer:
    """
    Session-level weight updater that wraps HuffmanSceneGraph.
    Provides a simple API for the orchestration layer.
    """

    def __init__(self, decay_lambda: float = 0.7):
        self.decay_lambda = decay_lambda
        self._hsg: Optional[HuffmanSceneGraph] = None

    def attach(self, hsg: HuffmanSceneGraph) -> None:
        self._hsg = hsg

    def record_and_rebalance(self, edited_node_ids: List[str]) -> None:
        if self._hsg is None:
            return
        for nid in edited_node_ids:
            self._hsg.record_edit(nid, self.decay_lambda)
        self._hsg.rebalance()

    def get_stats(self) -> Dict[str, Any]:
        if self._hsg is None:
            return {}
        return self._hsg.stats()


# ── Graph Serialiser ──────────────────────────────────────────────────────────


class GraphSerializer:
    """Utility to serialise / deserialise HSG to/from disk."""

    @staticmethod
    def save(hsg: HuffmanSceneGraph, path: str, depth_limit: Optional[int] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(hsg.to_dict(depth_limit=depth_limit), f, indent=2)

    @staticmethod
    def load(path: str) -> HuffmanSceneGraph:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return HuffmanSceneGraph.from_dict(data)
