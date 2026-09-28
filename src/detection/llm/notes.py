"""
Guardrail logic for the LLM layer: the fixed output schema, system prompts,
and validation/grounding checks deciding whether an LLM answer is trusted or
discarded for a template note.

The schema ({explanation, tools_used}) has no category field — the LLM
cannot state a category, it was never offered a place to put one. See
subagent.py._choose_category for how the category is actually decided.

Kept separate from subagent.py so this safety boundary can be unit-tested
without any agent-loop or MCP wiring.
"""
import json
from typing import Dict, List, Optional, Tuple

MAX_EXPLANATION_CHARS = 300

# Fixed answer shape — used as single_shot mode's litellm response_format. No category field.
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "maxLength": MAX_EXPLANATION_CHARS},
        "tools_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "tools_used"],
}

# parse_and_validate's failure reasons — a fixed vocabulary so callers can log/report
# a distinct trace action per reason without re-deriving it.
REASON_INVALID_JSON = "invalid_json"
REASON_UNGROUNDED = "ungrounded"

# A run only ends when the model calls final_answer as an actual tool call — printing JSON as
# plain text fails to parse. Kept short: small models follow short instructions more reliably.
SYSTEM_PROMPT = """Investigate this network traffic event, then answer via final_answer.

Code has already decided this event's category (given to you in the task, with its
confidence) — you are not choosing or restating it. Your job is only to explain, grounded in
the data, why that category is plausible for this event (or, if it's "Unknown", why the
evidence doesn't clearly point anywhere).

Steps:
1. Call top_features (and tree_vote_spread, if offered) to see this event's data. You may \
also call predict_attack_category yourself to see the model's own reasoning — it does not \
change the category already decided for you.
2. Call final_answer with your answer as a JSON string with exactly these keys:
   {"explanation": "<short reason, max 300 chars>", "tools_used": ["<tool names you called>"]}

Rules:
- Never output a confidence, probability, or category — those are decided elsewhere.
- Only mention feature names that appeared in a top_features result you saw.
- Do not call the same tool twice.
"""

# single_shot: one direct generation, no tool-calling loop — feature data is already looked up
# and embedded in the user message. Paired with ANSWER_JSON_SCHEMA as litellm's response_format.
# See docs/design-decisions.md for why (litellm/Ollama tool_choice limitation).
SINGLE_SHOT_SYSTEM_PROMPT = """You are assisting the Detection Subagent in explaining a network \
traffic event to downstream analysts. Code has already decided this event's category (given to \
you below, with its confidence) — you are not choosing or restating it, only explaining it. You \
are given the event's baseline classifier score and its most relevant features (including \
flow-shape context: destination port, flow duration, packet counts, SYN/FIN/RST flag counts), \
already looked up for you, each with a computed above/below/near-median direction.

Write a short explanation of why the given category is plausible for this event, grounded in \
the feature data — or, if the category is "Unknown", why the evidence doesn't clearly point \
anywhere.

Rules:
- Never output a confidence, probability, or category — those are decided elsewhere.
- Only mention feature names from the ones given to you.
- Respond with ONLY a JSON object matching this schema, no other text:
  {"explanation": "<short reason, max 300 chars>", "tools_used": []}
"""


def build_template_note(vote_std: Optional[float] = None) -> str:
    """Deterministic fallback explanation used whenever the LLM layer is
    skipped, invalid, times out, or errors. Does NOT include a `[category=]`
    tag — subagent.py attaches that separately, since the category is
    chosen by code independent of whether this template ends up used."""
    if vote_std is not None:
        return f"Borderline/anomalous event, tree_vote_std={vote_std:.3f} — LLM explanation unavailable."
    return "Anomalous event — LLM explanation unavailable."


def parse_and_validate(
    raw_output: str, known_feature_names: List[str], grounded_feature_names: List[str]
) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse and validate the LLM's raw answer against the fixed schema and
    the grounding rule. `known_feature_names` is every trainable feature
    name; `grounded_feature_names` is the subset top_features actually
    returned for this event — mentioning a known name outside that subset
    fails grounding. Returns (parsed_dict, None) on success, or
    (None, REASON_INVALID_JSON | REASON_UNGROUNDED) on failure."""
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None, REASON_INVALID_JSON

    if not isinstance(data, dict):
        return None, REASON_INVALID_JSON

    explanation = data.get("explanation")
    tools_used = data.get("tools_used")

    if not isinstance(explanation, str) or not explanation:
        return None, REASON_INVALID_JSON
    if len(explanation) > MAX_EXPLANATION_CHARS:
        return None, REASON_INVALID_JSON
    if not isinstance(tools_used, list) or not all(isinstance(t, str) for t in tools_used):
        return None, REASON_INVALID_JSON
    if not _is_grounded(explanation, known_feature_names, grounded_feature_names):
        return None, REASON_UNGROUNDED

    return {"explanation": explanation, "tools_used": tools_used}, None


def extract_json_object(text: str) -> Optional[str]:
    """Pull the substring from the first '{' to the last '}' out of `text`
    and return it if it parses as JSON, else None. The caller re-runs it
    through parse_and_validate — this only recovers JSON, no guardrail
    is relaxed."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start:end + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


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
