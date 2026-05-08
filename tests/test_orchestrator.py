import pytest
from spectra_core.orchestrator_loop import OrchestratorLoop

def test_orchestrator_initialization():
    orchestrator = OrchestratorLoop()
    assert orchestrator.stage1 is not None
    assert orchestrator.stage2 is not None
    assert orchestrator.stage3 is not None
    assert orchestrator.stage4 is not None
    assert orchestrator.stage5 is not None

# E2E testing is handled by test_new_pipeline.py and test_e2e_edit.py
