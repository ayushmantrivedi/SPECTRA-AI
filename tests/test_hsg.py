import pytest
from spectra_core.huffman_graph import HuffmanSceneGraph, AdaptiveRebalancer

def test_hsg_initialization():
    hsg = HuffmanSceneGraph()
    assert hsg.root is not None
    assert hsg.root.id == "SCENE_ROOT"
    assert hsg.root.weight == 1.0

def test_add_and_retrieve_node():
    hsg = HuffmanSceneGraph()
    hsg.add_node("person_01", "person", parent_id="SCENE_ROOT", attributes={"color": "red"})
    
    node = hsg.get_node("person_01")
    assert node is not None
    assert node.label == "person"
    assert node.attributes["color"] == "red"
    assert node.parent == hsg.root

def test_adaptive_rebalancer():
    hsg = HuffmanSceneGraph()
    rebalancer = AdaptiveRebalancer(hsg)
    
    hsg.add_node("node_a", "test", parent_id="SCENE_ROOT")
    node_a = hsg.get_node("node_a")
    
    initial_weight = node_a.weight
    
    # Simulate multiple edits
    rebalancer.update_weights(["node_a"])
    rebalancer.update_weights(["node_a"])
    
    # Weight should increase due to edits
    assert node_a.weight > initial_weight
