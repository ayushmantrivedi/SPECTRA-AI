"""
Spectral Sync Tracker
=====================
Tracks which nodes in the Spectral Semantic Graph (SSG) are edited most frequently 
across a session. After each edit, updates node weights using a weighted decay:

    W_new(node) = lambda * W_old(node) + (1 - lambda) * edit_frequency(node)

Then rebalances the tree so frequently edited nodes bubble up toward the root,
reducing traversal cost for common edits.
"""

from collections import defaultdict


class SpectralSyncTracker:
    """
    Session-level edit frequency tracker for Spectral Semantic Graphs.
    Persists across API requests for a single server session.
    """
    def __init__(self, decay_lambda=0.85):
        # lambda: how much to retain old weights vs learn from new edit freq
        self.decay_lambda = decay_lambda
        # Maps node_id -> cumulative edit count
        self.edit_counts = defaultdict(int)
        # Maps node_id -> current adapted weight
        self.adapted_weights = {}

    def record_edit(self, node_id: str):
        """Called whenever a node is targeted by an edit."""
        self.edit_counts[node_id] += 1

    def update_weights(self, ssg_dict: dict) -> dict:
        """
        Walk the SSG tree and update all node weights based on edit frequency.
        Returns a new SSG dict with updated weights.
        """
        total_edits = sum(self.edit_counts.values()) or 1
        return self._update_node(ssg_dict, total_edits)

    def _update_node(self, node: dict, total_edits: int) -> dict:
        """Recursively update a node's weight and rebalance its children."""
        node_id = node.get("id", "")
        edit_freq = self.edit_counts.get(node_id, 0) / total_edits

        old_weight = node.get("weight", 0.0)
        # Spectral Sync formula
        new_weight = self.decay_lambda * old_weight + (1 - self.decay_lambda) * edit_freq
        self.adapted_weights[node_id] = new_weight

        updated_node = dict(node)
        updated_node["weight"] = round(new_weight, 4)
        updated_node["edit_count"] = self.edit_counts.get(node_id, 0)

        if "children" in node and node["children"]:
            updated_children = [self._update_node(c, total_edits) for c in node["children"]]
            # Rebalance: sort children descending by their NEW weight
            # (most edited / highest priority nodes bubble up)
            updated_children.sort(key=lambda x: x["weight"], reverse=True)
            updated_node["children"] = updated_children

        return updated_node

    def get_priority_ranking(self) -> list:
        """Returns nodes sorted from most-edited to least-edited."""
        return sorted(self.edit_counts.items(), key=lambda x: x[1], reverse=True)

    def get_stats(self) -> dict:
        return {
            "total_edits": sum(self.edit_counts.values()),
            "unique_nodes_edited": len(self.edit_counts),
            "top_nodes": self.get_priority_ranking()[:5],
            "adapted_weights": dict(list(self.adapted_weights.items())[:10])
        }
