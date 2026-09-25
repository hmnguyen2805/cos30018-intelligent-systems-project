"""
Guardrail logic for the Detection Subagent's optional LLM layer: the fixed
output schema, the system prompt that states the rules, and the
validation/grounding checks that decide whether an LLM answer is trusted or
discarded in favour of a deterministic template note.

Kept separate from subagent.py so these checks — the actual safety
boundary between "LLM suggestion" and "what goes in detector_notes" — can be
read and unit-tested without any agent-loop or MCP wiring.
"""
import json
from typing import Dict, List, Optional

ALLOWED_CATEGORIES = [
    "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet", "Infiltration", "Unknown",
]
MAX_EXPLANATION_CHARS = 300

SYSTEM_PROMPT = """You are assisting the Detection Subagent in explaining a borderline or \
anomalous network traffic event to downstream analysts.

Rules:
- You do not decide whether the event is anomalous. Never output a confidence or \
probability number — that decision is already made by deterministic code.
- Use the available tools to investigate. Call top_features before writing your \
explanation, so your explanation is grounded in real data.
- Only mention feature names that appear in the output of a top_features call you made. \
Do not name a feature you did not look up.
- Pick exactly one category from this fixed list: {categories}.
- Output ONLY a JSON object matching this schema, with no other text before or after it:
  {{"category": "<one of the list above>", "explanation": "<string, at most 300 characters>", \
"tools_used": ["<tool name>", ...]}}
""".format(categories=", ".join(ALLOWED_CATEGORIES))


def build_template_note(vote_std: Optional[float] = None) -> str:
    """Deterministic fallback note used whenever the LLM layer is skipped,
    invalid, times out, or errors. Always tagged category=Unknown so
    detector_notes has a consistent `[category=...]` prefix either way."""
    if vote_std is not None:
        return (
            f"[category=Unknown] Borderline/anomalous event, "
            f"tree_vote_std={vote_std:.3f} — LLM explanation unavailable."
        )
    return "[category=Unknown] Anomalous event — LLM explanation unavailable."


def parse_and_validate(
    raw_output: str, known_feature_names: List[str], grounded_feature_names: List[str]
) -> Optional[Dict]:
    """Parse the LLM's raw final answer and validate it against the fixed
    schema and the grounding rule.

    `known_feature_names` is every feature name the model was trained on (the
    universe of names that could plausibly be mentioned); `grounded_feature_names`
    is the subset actually returned by top_features for this event. A mention
    of any `known_feature_names` entry that isn't in `grounded_feature_names`
    fails validation — it means the LLM named a feature it didn't look up.

    Returns the parsed {"category", "explanation", "tools_used"} dict on
    success, or None if the output should be discarded for a template note.
    """
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    category = data.get("category")
    explanation = data.get("explanation")
    tools_used = data.get("tools_used")

    if category not in ALLOWED_CATEGORIES:
        return None
    if not isinstance(explanation, str) or not explanation:
        return None
    if len(explanation) > MAX_EXPLANATION_CHARS:
        return None
    if not isinstance(tools_used, list) or not all(isinstance(t, str) for t in tools_used):
        return None
    if not _is_grounded(explanation, known_feature_names, grounded_feature_names):
        return None

    return {"category": category, "explanation": explanation, "tools_used": tools_used}


def _is_grounded(
    explanation: str, known_feature_names: List[str], grounded_feature_names: List[str]
) -> bool:
    """True iff no known feature name that appears (as a substring) in
    `explanation` is missing from `grounded_feature_names`."""
    grounded = set(grounded_feature_names)
    for name in known_feature_names:
        if name in explanation and name not in grounded:
            return False
    return True
