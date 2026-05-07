"""
SPECTRA Pydantic API Schemas
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class ModificationSchema(BaseModel):
    node: str
    attribute_path: str
    old_value: Optional[str] = None
    new_value: str
    operation_type: str = "attribute_change"
    influence: Optional[Dict[str, Any]] = None


class CascadeTargetSchema(BaseModel):
    node: str
    reason: str
    action: str


class GenerationHintsSchema(BaseModel):
    context_for_diffusion: str = ""
    preserve_identity: bool = True
    material_consistency: bool = True


class EditPlanSchema(BaseModel):
    type: str
    confidence: float = Field(ge=0, le=1)
    target_nodes: List[str]
    primary_action: str = "MODIFY_ATTRIBUTE"
    modifications: List[ModificationSchema] = []
    cascade_targets: List[CascadeTargetSchema] = []
    frozen_regions: List[str] = []
    traversal_path: List[str] = []
    affected_masks: List[str] = []
    generation_hints: GenerationHintsSchema = GenerationHintsSchema()


# ── Analyze endpoint ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base-64 encoded JPEG/PNG image")
    force_reanalyze: bool = False


class AnalyzeResponse(BaseModel):
    status: str
    hsg: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time_ms: int


# ── Edit endpoint ─────────────────────────────────────────────────────────────

class EditRequest(BaseModel):
    image_base64: str = Field(..., description="Base-64 encoded source image")
    instruction: str = Field(..., description="Natural language edit instruction")
    force_reanalyze: bool = False


class StageAuditSchema(BaseModel):
    stage: str
    status: str
    elapsed_ms: int
    meta: Dict[str, Any] = {}


class EditAuditSchema(BaseModel):
    edit_id: str
    user_instruction: str
    stages: List[StageAuditSchema]
    total_time_s: float
    final_status: str
    verification_score: float


class EditResponse(BaseModel):
    status: str
    edit_id: str
    result_image_b64: str
    hsg: Dict[str, Any]
    verification_report: Dict[str, Any]
    edit_plans: List[Dict[str, Any]]
    audit: Dict[str, Any]


# ── Verify endpoint ───────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    original_image_base64: str
    edited_image_base64: str
    edit_plan: Dict[str, Any]
    original_hsg: Dict[str, Any] = {}


class CheckResultSchema(BaseModel):
    name: str
    status: str
    confidence: float
    details: str


class VerifyResponse(BaseModel):
    verification_result: str
    overall_score: float
    elapsed_ms: int
    checks: List[Dict[str, Any]]
    corrections_needed: Optional[Dict[str, Any]] = None
    next_action: str


# ── HSG stats ─────────────────────────────────────────────────────────────────

class HSGStatsResponse(BaseModel):
    total_nodes: int
    max_depth: int
    total_edits: int
    most_edited: List[Any]
    top_weight_nodes: List[Any]
