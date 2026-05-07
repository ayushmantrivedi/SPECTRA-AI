import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_core.nlp_parser import IntentClassifier, IntentType, EditPlanGenerator

def test_intent_classifier():
    classifier = IntentClassifier()
    
    assert classifier.classify("Add sunglasses to the person") == IntentType.STRUCTURAL_ADD
    assert classifier.classify("Make the lighting a sunset") == IntentType.CASCADE
    assert classifier.classify("Change the outfit and hat") == IntentType.MULTI_NODE
    assert classifier.classify("Change the hat color to red") == IntentType.SINGLE_NODE
    assert classifier.classify("Remove the beard") == IntentType.STRUCTURAL_REMOVE

def test_parser_fallback():
    parser = EditPlanGenerator()
    hsg_dict = {
        "id": "SCENE_ROOT",
        "children": [
            {"id": "semantic_hair"},
            {"id": "semantic_face"}
        ]
    }
    
    plan = parser.parse("Change hair to red", hsg_dict)
    assert plan["type"] == IntentType.SINGLE_NODE
    assert "semantic_hair" in plan["target_nodes"]
    assert plan["modifications"][0]["new_value"].lower() == "red"
