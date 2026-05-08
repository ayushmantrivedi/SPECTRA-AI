import pytest
from spectra_core.nlp_parser import IntentClassifier, EditPlanGenerator

def test_intent_classifier():
    classifier = IntentClassifier()
    instruction = "Change the hat color to red"
    
    # Normally this would call an LLM, but we test the structure handling here
    # Assuming local parsing logic or mock
    result = classifier.classify(instruction)
    assert result is not None
    # We won't test exact outputs without mocking the LLM, 
    # but we can ensure the interface returns a dict.
    assert isinstance(result, dict)

def test_edit_plan_generator():
    generator = EditPlanGenerator()
    instruction = "Make the outfit all black"
    hsg_context = {"nodes": {"shirt": {}, "pants": {}}}
    
    plan = generator.generate_plan(instruction, hsg_context)
    assert plan is not None
    assert "type" in plan
    assert "modifications" in plan
