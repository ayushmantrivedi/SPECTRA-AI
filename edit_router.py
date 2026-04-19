"""
Edit Router
===========
Classifies the user's prompt intent BEFORE it hits phi3:mini.
Ensures the correct editing strategy and node scope is chosen for any prompt.

Intent classes and their default strategies:
  GLOBAL_PERSON   → semantic_person, ZT=1.0, SD inpainting
  GLOBAL_COLOR    → SCENE_ROOT or semantic_person, ZT=1.0
  SKY_EDIT        → semantic_sky, ZL or ZT, SD inpainting
  BACKGROUND_EDIT → semantic_grass / semantic_tree / semantic_mountain
  LIGHTING        → ZL dominant, pixel-space (fast)
  STRUCTURAL_ADD  → ZB=1.0, SD generative fill
  TEXTURE_CHANGE  → ZT=1.0, semantic target
  GLOBAL_MOOD     → SCENE_ROOT, ZL+ZT blend

Usage:
    from edit_router import classify_intent, build_intent_hint
    intent_class = classify_intent(prompt)
    hint = build_intent_hint(intent_class)
    # Prepend hint to the LLM system prompt for guidance
"""

from __future__ import annotations
import re

# ── Keyword fingerprints ──────────────────────────────────────────────────────

_FINGERPRINTS: list[tuple[str, list[str]]] = [
    ("GLOBAL_PERSON", [
        r"\bsilhouette\b", r"\bperson\b", r"\bwoman\b", r"\bman\b",
        r"\bface\b", r"\bskin\b", r"\bhair\b", r"\bportrait\b",
        r"\bcloth(es|ing)\b", r"\bshirt\b", r"\bdress\b", r"\bjacket\b",
        r"\boutfit\b", r"\bbody\b", r"\bmodel\b", r"\bpeople\b",
    ]),
    ("SKY_EDIT", [
        r"\bsky\b", r"\bclouds?\b", r"\batmosphere\b", r"\bsunset\b",
        r"\bsunrise\b", r"\bdusk\b", r"\bdawn\b", r"\bovercast\b",
        r"\bstorm(y)?\b", r"\bblue sky\b", r"\bgolden hour\b",
    ]),
    ("STRUCTURAL_ADD", [
        r"\badd\b", r"\bappear\b", r"\binsert\b", r"\bplace\b",
        r"\bput\b", r"\bcreate\b", r"\bgenerate\b", r"\bsun\b",
        r"\bmoon\b", r"\bstar(s)?\b", r"\blight(s)?\b", r"\beffect\b",
        r"\bobject\b", r"\bbird(s)?\b", r"\btree(s)?\b",
    ]),
    ("LIGHTING", [
        r"\bbright(en)?\b", r"\bdark(en)?\b", r"\bshadow\b",
        r"\blight(ing)?\b", r"\billuminat\w+\b", r"\bglow\b",
        r"\bluminous\b", r"\bexposure\b", r"\bhighlight\b", r"\bdim\b",
        r"\bnight\b", r"\bneon\b",
    ]),
    ("BACKGROUND_EDIT", [
        r"\bbackground\b", r"\bscene\b", r"\blandscape\b", r"\bnature\b",
        r"\bforest\b", r"\bgrass\b", r"\btrees?\b", r"\bmountain\b",
        r"\bfield\b", r"\bbehind\b", r"\benvironment\b",
    ]),
    ("GLOBAL_COLOR", [
        r"\bcolor(ize)?\b", r"\bcolou?r\b", r"\bhue\b", r"\btint\b",
        r"\bblack (and white|silhouette)\b", r"\bsepia\b",
        r"\bvibrant\b", r"\bpale\b", r"\bsaturati\w+\b",
        r"\bmonochrome\b", r"\bgrayscale\b",
    ]),
    ("GLOBAL_MOOD", [
        r"\bmood\b", r"\batmospher(e|ic)\b", r"\bcinematic\b",
        r"\bdramatic\b", r"\bwarm\b", r"\bcool\b", r"\bvintage\b",
        r"\bretro\b", r"\bfilm grain\b", r"\bcolor grade\b",
    ]),
]

# Default strategy per intent class
# Maps intent class → (preferred_node_prefix, ZT, ZL, ZB, use_sd)
_STRATEGIES: dict[str, dict] = {
    "GLOBAL_PERSON":   {"node": "semantic_person",   "ZT": 1.0, "ZL": 0.0, "ZB": 0.0, "sd": True},
    "SKY_EDIT":        {"node": "semantic_sky",       "ZT": 0.5, "ZL": 0.6, "ZB": 0.0, "sd": True},
    "STRUCTURAL_ADD":  {"node": "SCENE_ROOT",         "ZT": 0.0, "ZL": 0.0, "ZB": 1.0, "sd": True},
    "LIGHTING":        {"node": "SCENE_ROOT",         "ZT": 0.0, "ZL": 1.0, "ZB": 0.0, "sd": False},
    "BACKGROUND_EDIT": {"node": "semantic_grass",     "ZT": 0.8, "ZL": 0.0, "ZB": 0.0, "sd": True},
    "GLOBAL_COLOR":    {"node": "semantic_person",    "ZT": 1.0, "ZL": 0.0, "ZB": 0.0, "sd": True},
    "GLOBAL_MOOD":     {"node": "SCENE_ROOT",         "ZT": 0.4, "ZL": 0.6, "ZB": 0.0, "sd": False},
}


def classify_intent(prompt: str) -> str:
    """
    Classify a natural-language edit prompt into an intent class.
    Returns one of: GLOBAL_PERSON, SKY_EDIT, STRUCTURAL_ADD, LIGHTING,
                    BACKGROUND_EDIT, GLOBAL_COLOR, GLOBAL_MOOD, UNKNOWN
    """
    prompt_lower = prompt.lower()
    scores: dict[str, int] = {}
    for cls, patterns in _FINGERPRINTS:
        score = sum(1 for p in patterns if re.search(p, prompt_lower))
        if score:
            scores[cls] = score

    if not scores:
        return "UNKNOWN"
    return max(scores, key=scores.get)


def get_strategy(intent_class: str) -> dict:
    """Return the default editing strategy for an intent class."""
    return _STRATEGIES.get(intent_class, {
        "node": "SCENE_ROOT", "ZT": 0.5, "ZL": 0.5, "ZB": 0.0, "sd": True
    })


def build_routing_hint(intent_class: str) -> str:
    """
    Build a routing hint string to prepend to the LLM system prompt.
    Guides phi3:mini toward the correct node and influence values.
    """
    strategy = get_strategy(intent_class)
    lines = [
        f"[ROUTING HINT — Intent: {intent_class}]",
        f"Preferred target node: \"{strategy['node']}\"",
        f"Suggested influence: ZT={strategy['ZT']} ZL={strategy['ZL']} ZB={strategy['ZB']}",
        f"Use SD inpainting: {'YES' if strategy['sd'] else 'NO (use pixel-space)'}"
    ]
    return "\n".join(lines)


def route_prompt(prompt: str) -> tuple[str, dict, str]:
    """
    Full routing pipeline.
    Returns (intent_class, strategy_dict, routing_hint_str).
    """
    intent   = classify_intent(prompt)
    strategy = get_strategy(intent)
    hint     = build_routing_hint(intent)
    return intent, strategy, hint


if __name__ == "__main__":
    # Quick self-test
    test_prompts = [
        "turn the woman completely black to make a silhouette",
        "make the sky a golden sunset",
        "add a glowing sun in the upper sky",
        "brighten the whole image",
        "change background to a green forest",
        "apply a warm cinematic color grade",
        "colorize the whole image in sepia tones",
    ]
    for p in test_prompts:
        cls, strat, hint = route_prompt(p)
        print(f"\nPrompt: {p!r}")
        print(f"  Intent: {cls} | Node: {strat['node']} | SD: {strat['sd']}")
