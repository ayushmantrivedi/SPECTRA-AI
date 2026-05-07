import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_core.verification_engine import VerificationEngine

def test_heuristic_fulfillment():
    ve = VerificationEngine(device="cpu")
    # Simulate a parsed plan
    plan = {
        "confidence": 0.8,
        "type": "SINGLE_NODE"
    }
    
    # We can pass dummy images since the heuristic fulfillment doesn't use them
    dummy_img = Image.new("RGB", (100, 100))
    res = ve._check_instruction_fulfillment(dummy_img, plan)
    
    assert res.status == "PASS"
    assert res.confidence == 0.8
