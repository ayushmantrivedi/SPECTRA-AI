"""
SPECTRA NLP-LLM Parser Agent
==============================
Stage 2 of the SPECTRA pipeline.

Takes the user instruction + HSG context and produces a structured Edit Plan.

Intent types:
  SINGLE_NODE      → one node, one attribute change
  MULTI_NODE       → multiple nodes, same or related attribute changes
  CASCADE          → primary edit + downstream graph propagation
  STRUCTURAL_ADD   → add new element (sunglasses, hat, etc.)
  STRUCTURAL_REMOVE→ remove existing element
  AMBIGUOUS        → needs clarification (low-confidence parse)

Routing:
  1. Local rule-based intent classifier (EditRouter, fast, no LLM needed)
  2. Gemini 2.5 Pro / LLM for complex semantic plans
  3. phi3:mini (Ollama) fallback
"""

from __future__ import annotations

import os
import re
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("spectra.nlp_parser")

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ── Intent type constants ─────────────────────────────────────────────────────

class IntentType:
    SINGLE_NODE       = "SINGLE_NODE"
    MULTI_NODE        = "MULTI_NODE"
    CASCADE           = "CASCADE"
    STRUCTURAL_ADD    = "STRUCTURAL_ADD"
    STRUCTURAL_REMOVE = "STRUCTURAL_REMOVE"
    AMBIGUOUS         = "AMBIGUOUS"
    UNKNOWN           = "UNKNOWN"


# ── Rule-based Intent Classifier ─────────────────────────────────────────────

_FINGERPRINTS: List[Tuple[str, List[str]]] = [
    (IntentType.STRUCTURAL_ADD, [
        r"\badd\b", r"\binsert\b", r"\bplace\b", r"\bput\b",
        r"\bcreate\b", r"\bgenerate\b", r"\bappear\b",
        r"\bsunglasses\b", r"\bglasses\b", r"\beard\b", r"\bhat\b.*\badd\b",
    ]),
    (IntentType.STRUCTURAL_REMOVE, [
        r"\bremove\b", r"\berase\b", r"\bdelete\b", r"\bdisappear\b",
        r"\bwithout\b", r"\bno \w+\b",
    ]),
    (IntentType.CASCADE, [
        r"\blighting\b", r"\bsunset\b", r"\bsunrise\b", r"\bdusk\b",
        r"\bdawn\b", r"\bnight\b", r"\bday\b",
        r"\bgolden hour\b", r"\bovercast\b", r"\bstorm\b",
        r"\bshadow\b.*\bchange\b", r"\bchange.*\blighting\b",
    ]),
    (IntentType.MULTI_NODE, [
        r"\boutfit\b", r"\ball\b.*\bcloth\w*\b", r"\bwhole\s+look\b",
        r"\bhead to toe\b", r"\beverything\b",
        r"\band\b.*\band\b",          # "X and Y and Z"
        r"\b\w+,\s*\w+,\s*\w+\b",    # comma-separated list
    ]),
]

_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "person":     {"ZT": 1.0, "ZL": 0.0, "ZB": 0.0, "sd": True},
    "hair":       {"ZT": 1.0, "ZL": 0.3, "ZB": 0.0, "sd": True},
    "face":       {"ZT": 0.8, "ZL": 0.2, "ZB": 0.0, "sd": True},
    "clothing":   {"ZT": 1.0, "ZL": 0.0, "ZB": 0.0, "sd": True},
    "background": {"ZT": 0.8, "ZL": 0.0, "ZB": 0.2, "sd": True},
    "sky":        {"ZT": 0.5, "ZL": 0.6, "ZB": 0.0, "sd": True},
    "lighting":   {"ZT": 0.0, "ZL": 1.0, "ZB": 0.0, "sd": False},
    "structure":  {"ZT": 0.0, "ZL": 0.0, "ZB": 1.0, "sd": True},
}


class IntentClassifier:
    """
    Lightweight rule-based classifier that identifies intent before touching LLM.
    Avoids API calls for simple, common edits.
    """

    def classify(self, instruction: str) -> str:
        lower = instruction.lower()
        scores: Dict[str, int] = {}
        for intent, patterns in _FINGERPRINTS:
            score = sum(1 for p in patterns if re.search(p, lower))
            if score:
                scores[intent] = score
        if not scores:
            return IntentType.SINGLE_NODE   # default: assume simple attribute edit
        return max(scores, key=scores.get)

    def extract_target_hints(self, instruction: str) -> List[str]:
        """Best-effort noun extraction to seed target node search."""
        lower = instruction.lower()
        known = [
            "hair", "face", "eyes", "mouth", "nose", "skin", "beard",
            "hat", "shirt", "jacket", "pants", "dress", "outfit", "clothing",
            "background", "sky", "grass", "tree", "shadow", "lighting",
            "sunglasses", "glasses",
        ]
        return [w for w in known if w in lower]


# ── LLM-based Parser ─────────────────────────────────────────────────────────

_PARSE_SYSTEM_PROMPT = """
You are SPECTRA's Edit Plan Generator for a surgical image editing system.

Given:
1. A Huffman Scene Graph (HSG) describing what is in the image.
2. A natural language instruction from the user.

Your job is to produce a structured Edit Plan JSON. Be precise and surgical.

RULES:
- Only target nodes that exist in the HSG (use their exact "id" values).
- Mark ALL other nodes as "frozen_regions" (they must not change).
- If the edit propagates (e.g., lighting change affects shadows), list cascade_targets.
- If the instruction is ambiguous or impossible, use type "AMBIGUOUS".

Output ONLY a valid JSON object — no markdown, no extra text:
{
  "type": "<SINGLE_NODE|MULTI_NODE|CASCADE|STRUCTURAL_ADD|STRUCTURAL_REMOVE|AMBIGUOUS>",
  "confidence": <float 0-1>,
  "target_nodes": ["<node_id>"],
  "primary_action": "<MODIFY_ATTRIBUTE|REPLACE|ADD|REMOVE>",
  "modifications": [
    {
      "node": "<node_id>",
      "attribute_path": "<e.g. color.primary>",
      "old_value": "<current>",
      "new_value": "<desired>",
      "operation_type": "<attribute_change|replace|add|remove>"
    }
  ],
  "cascade_targets": [
    {
      "node": "<node_id>",
      "reason": "<why it is affected>",
      "action": "<what to do>"
    }
  ],
  "frozen_regions": ["<node_id>"],
  "traversal_path": ["SCENE_ROOT", "<parent>", "<target>"],
  "affected_masks": ["<node_id>"],
  "generation_hints": {
    "context_for_diffusion": "<hint>",
    "preserve_identity": <true|false>,
    "material_consistency": <true|false>
  },
  "clarification_questions": []
}
"""


class TraversalPlanner:
    """Determines the optimal traversal path through the HSG for a given edit."""

    def plan(
        self,
        target_nodes: List[str],
        hsg_flat: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_ids = {n["id"] for n in hsg_flat}
        target_set = set(target_nodes)
        frozen = [nid for nid in all_ids if nid not in target_set]

        # Build traversal path: ROOT → ... → target (simple linear path)
        path = ["SCENE_ROOT"] + [t for t in target_nodes if t != "SCENE_ROOT"]

        return {
            "traversal_path": path,
            "frozen_regions": frozen,
            "affected_masks": list(target_set),
        }


class EditPlanGenerator:
    """
    Full NLP-LLM parser that produces structured Edit Plans.

    Routing:
      1. Gemini (cloud, best quality)
      2. Ollama phi3:mini (local fallback)
      3. Rule-based fallback (offline)
    """

    OLLAMA_BASE = "http://localhost:11434"
    PARSER_MODEL = "phi3:mini"

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.traversal_planner = TraversalPlanner()
        self._gemini_model = None
        self._init_gemini()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_gemini(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and _GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro-preview-05-06",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.05,
                        response_mime_type="application/json",
                    ),
                )
                logger.info("[Parser] Gemini parser configured.")
            except Exception as e:
                logger.warning(f"[Parser] Gemini setup failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(
        self,
        instruction: str,
        hsg_dict: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Parse a natural language instruction against the HSG.

        Returns a structured Edit Plan dict.
        """
        # Fast pre-classification (no LLM)
        intent_type = self.intent_classifier.classify(instruction)
        target_hints = self.intent_classifier.extract_target_hints(instruction)

        # Compress HSG for token efficiency
        hsg_summary = self._compress_hsg(hsg_dict, max_chars=3000)

        # Try best available backend
        if self._gemini_model:
            plan = self._parse_gemini(instruction, hsg_summary, intent_type)
        elif self._check_ollama():
            plan = self._parse_ollama(instruction, hsg_summary)
        else:
            plan = self._parse_rule_based(instruction, hsg_dict, intent_type, target_hints)

        # Enrich with traversal plan
        target_nodes = plan.get("target_nodes", [])
        hsg_flat = self._flatten_hsg(hsg_dict)
        traversal = self.traversal_planner.plan(target_nodes, hsg_flat)

        plan.setdefault("traversal_path", traversal["traversal_path"])
        plan.setdefault("frozen_regions", traversal["frozen_regions"])
        plan.setdefault("affected_masks", traversal["affected_masks"])
        plan.setdefault("type", intent_type)
        plan.setdefault("confidence", 0.75)

        return plan

    def decompose_multi_operation(
        self, instruction: str, hsg_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Break a complex instruction into sequential single-operation plans.
        E.g. "Make hat red, add sunglasses, beach sunset" → [plan1, plan2, plan3]
        """
        # Split on commas, "and", semicolons for multi-intent instructions
        sub_instructions = re.split(r",\s*|\band\b|\bthен\b|;\s*", instruction)
        sub_instructions = [s.strip() for s in sub_instructions if len(s.strip()) > 3]

        plans: List[Dict[str, Any]] = []
        for sub in sub_instructions:
            plan = self.parse(sub, hsg_dict)
            plans.append(plan)

        return plans if plans else [self.parse(instruction, hsg_dict)]

    # ── Gemini path ───────────────────────────────────────────────────────────

    def _parse_gemini(
        self, instruction: str, hsg_summary: str, hint_intent: str
    ) -> Dict[str, Any]:
        try:
            user_content = (
                f"HSG Context:\n{hsg_summary}\n\n"
                f"Pre-classified intent (use as hint): {hint_intent}\n\n"
                f"User Instruction: \"{instruction}\"\n\n"
                "Generate the Edit Plan JSON:"
            )
            response = self._gemini_model.generate_content(
                [_PARSE_SYSTEM_PROMPT, user_content]
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[Parser] Gemini parse failed: {e}. Trying Ollama.")
            if self._check_ollama():
                return self._parse_ollama(instruction, hsg_summary)
            return self._parse_rule_based(instruction, {}, hint_intent, [])

    # ── Ollama path ───────────────────────────────────────────────────────────

    def _parse_ollama(self, instruction: str, hsg_summary: str) -> Dict[str, Any]:
        prompt = (
            _PARSE_SYSTEM_PROMPT
            + f"\n\nHSG Context:\n{hsg_summary}\n\nUser Instruction: \"{instruction}\"\n\nJSON:"
        )
        try:
            resp = requests.post(
                f"{self.OLLAMA_BASE}/api/generate",
                json={"model": self.PARSER_MODEL, "prompt": prompt,
                      "stream": False, "keep_alive": 0,
                      "options": {"temperature": 0.05}},
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            start = raw.find("{"); end = raw.rfind("}") + 1
            if start != -1 and end > 0:
                return json.loads(raw[start:end])
        except Exception as e:
            logger.warning(f"[Parser] Ollama parse failed: {e}")
        return {}

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _parse_rule_based(
        self,
        instruction: str,
        hsg_dict: Dict[str, Any],
        intent_type: str,
        target_hints: List[str],
    ) -> Dict[str, Any]:
        """Generate a best-effort edit plan without any LLM."""
        hsg_flat = self._flatten_hsg(hsg_dict)
        all_ids = [n["id"] for n in hsg_flat]

        # Match target hints to actual node IDs
        target_nodes: List[str] = []
        for hint in target_hints:
            matches = [nid for nid in all_ids
                       if hint in nid.lower() or nid.lower() in hint]
            target_nodes.extend(matches)

        # If nothing matched, use SCENE_ROOT
        if not target_nodes:
            target_nodes = ["SCENE_ROOT"]

        target_nodes = list(dict.fromkeys(target_nodes))  # deduplicate

        # Try to extract new_value from instruction
        color_match = re.search(
            r"\b(red|blue|green|black|white|yellow|purple|pink|orange|silver|gold|brown|grey|gray)\b",
            instruction, re.I
        )
        new_value = color_match.group(1) if color_match else "modified"

        strategy = {}
        for hint in target_hints:
            if hint in _STRATEGY_MAP:
                strategy = _STRATEGY_MAP[hint]
                break
        if not strategy:
            strategy = {"ZT": 0.8, "ZL": 0.0, "ZB": 0.0, "sd": True}

        return {
            "type": intent_type,
            "confidence": 0.60,
            "target_nodes": target_nodes,
            "primary_action": "MODIFY_ATTRIBUTE",
            "modifications": [
                {
                    "node": t,
                    "attribute_path": "color.primary",
                    "old_value": "unknown",
                    "new_value": new_value,
                    "operation_type": "attribute_change",
                    "influence": {
                        "ZT": strategy.get("ZT", 0.8),
                        "ZL": strategy.get("ZL", 0.0),
                        "ZB": strategy.get("ZB", 0.0),
                        "intent": instruction,
                    },
                }
                for t in target_nodes
            ],
            "cascade_targets": [],
            "generation_hints": {
                "context_for_diffusion": instruction,
                "preserve_identity": True,
                "material_consistency": True,
            },
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _compress_hsg(self, hsg_dict: Dict[str, Any], max_chars: int = 3000) -> str:
        """Summarise HSG to fit within LLM context budget."""
        summary = json.dumps(hsg_dict, indent=1)
        if len(summary) <= max_chars:
            return summary
        # Include only root + depth-1 children
        shallow = dict(hsg_dict)
        if "children" in shallow:
            children_summary = []
            for c in shallow["children"][:15]:
                short = {"id": c.get("id"), "label": c.get("label"),
                         "weight": c.get("weight"), "bbox": c.get("bbox"),
                         "attributes": c.get("attributes", {})}
                children_summary.append(short)
            shallow["children"] = children_summary
        return json.dumps(shallow, indent=1)[:max_chars] + "\n...(truncated)"

    def _flatten_hsg(self, node: Dict[str, Any], out: Optional[List] = None) -> List[Dict]:
        if out is None:
            out = []
        out.append(node)
        for child in node.get("children", []):
            self._flatten_hsg(child, out)
        return out

    def _check_ollama(self) -> bool:
        try:
            resp = requests.get(f"{self.OLLAMA_BASE}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
