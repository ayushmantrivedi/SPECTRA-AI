import os
import sys

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_core.huffman_graph import HuffmanSceneGraph

def test_hsg_creation():
    hsg = HuffmanSceneGraph()
    # Test simple scene dict
    scene_dict = {
        "id": "SCENE_ROOT",
        "weight": 1.0,
        "children": [
            {"id": "person_1", "weight": 0.9, "children": []},
            {"id": "bg", "weight": 0.3, "children": []}
        ]
    }
    hsg.build_from_orchestrator_output(scene_dict)
    
    # Root should exist
    assert hsg.root.id == "SCENE_ROOT"
    assert hsg.root.weight == 1.0
    
    # Children should be sorted by weight
    assert len(hsg.root.children) == 2
    assert hsg.root.children[0].id == "person_1"
    assert hsg.root.children[1].id == "bg"

def test_hsg_adaptive_rebalance():
    hsg = HuffmanSceneGraph()
    scene_dict = {
        "id": "SCENE_ROOT",
        "weight": 1.0,
        "children": [
            {"id": "person_1", "weight": 0.9, "children": []},
            {"id": "bg", "weight": 0.3, "children": []}
        ]
    }
    hsg.build_from_orchestrator_output(scene_dict)
    
    # Simulate editing the background multiple times
    hsg.record_edit("bg", decay_lambda=0.5)
    hsg.record_edit("bg", decay_lambda=0.5)
    hsg.record_edit("bg", decay_lambda=0.5)
    
    hsg.rebalance()
    
    # Background weight should increase
    bg_node = hsg.find_node("bg")
    assert bg_node.weight > 0.3
    assert bg_node.edit_count == 3
