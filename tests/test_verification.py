import pytest
from spectra_core.verification_engine import VerificationEngine

def test_verification_engine_init():
    engine = VerificationEngine()
    assert engine is not None

def test_target_accuracy_check():
    engine = VerificationEngine()
    
    # Mock HSG data
    original_hsg = {"nodes": {"hat": {"attributes": {"color": "blue"}}}}
    edited_hsg = {"nodes": {"hat": {"attributes": {"color": "red"}}}}
    edit_plan = {"modifications": [{"node": "hat", "attribute_path": "color", "new_value": "red"}]}
    
    check_result = engine._check_target_accuracy(original_hsg, edited_hsg, edit_plan)
    assert check_result["status"] == "PASS"

def test_target_accuracy_fail():
    engine = VerificationEngine()
    
    # Mock HSG data
    original_hsg = {"nodes": {"hat": {"attributes": {"color": "blue"}}}}
    edited_hsg = {"nodes": {"hat": {"attributes": {"color": "blue"}}}}
    edit_plan = {"modifications": [{"node": "hat", "attribute_path": "color", "new_value": "red"}]}
    
    check_result = engine._check_target_accuracy(original_hsg, edited_hsg, edit_plan)
    assert check_result["status"] == "FAIL"
