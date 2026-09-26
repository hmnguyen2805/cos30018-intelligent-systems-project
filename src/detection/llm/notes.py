"""
Guardrail logic for the Detection Subagent's optional LLM layer: the fixed
output schema, the system prompts, and the validation/grounding checks that
decide whether an LLM answer is trusted or discarded in favour of a
deterministic template note.

The LLM never decides the attack category. That's a deterministic tool
(classifier.predict_attack_category) code applies a confidence threshold to
— the same principle as is_anomalous/confidence — see subagent.py's
_choose_category. The LLM's only job is to write a short, grounded
explanation of a category code already chose, and its schema reflects that
structurally: {explanation, tools_used}, no category field. There's nothing
for a validation rule to reject here — the LLM cannot state a category, it
was never offered a place to put one.

Kept separate from subagent.py so these checks — the actual safety
boundary between "LLM suggestion" and "what goes in detector_notes" — can be
read and unit-tested without any agent-loop or MCP wiring.
"""
import json
from typing import Dict, List, Optional, Tuple

MAX_EXPLANATION_CHARS = 300

# JSON Schema for the fixed answer shape — used by subagent.py's single_shot mode
# (DETECTION_LLM_MODE=single_shot) via litellm's response_format={"type": "json_schema", ...},
# which Ollama enforces with grammar-constrained decoding. No category field: see module
# docstring.
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "maxLength": MAX_EXPLANATION_CHARS},
        "tools_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "tools_used"],
}

# parse_and_validate's second return value on failure — kept as a fixed, short
# vocabulary so subagent.py can log a distinct trace action per reason and
# evaluate.py can report fallback_reason per event without re-deriving it.
REASON_INVALID_JSON = "invalid_json"
REASON_UNGROUNDED = "ungrounded"

# ToolCallingAgent requires every step to be a tool call, and the run only ends when you
# call final_answer — it is not enough to just print JSON as your message text, that has
# no tool call in it and the step fails to parse. Keep this short: small models follow
# short, concrete instructions far more reliably than a longer rules list.
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

# single_shot mode (DETECTION_LLM_MODE=single_shot): code has already called top_features (and
# tree_vote_spread, if borderline) via MCP, already decided the category (via
# classifier.predict_attack_category directly, not through MCP), and put all of it in the user
# message below — no tool-calling loop at all here, just one direct generation. Paired with
# response_format's json_schema (ANSWER_JSON_SCHEMA) via litellm, which Ollama enforces with
# grammar-constrained decoding, this is the reliable path for small local models that don't
# consistently honor tool_choice (litellm's own ollama transformation drops tool_choice
# entirely — "causes ollama requests to hang" — so nothing can force a small model through the
# agent loop's tool-call requirement).
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
    """Parse the LLM's raw final answer and validate it against the fixed
    schema ({explanation, tools_used} — no category, see module docstring)
    and the grounding rule.

    `known_feature_names` is every feature name the model was trained on (the
    universe of names that could plausibly be mentioned); `grounded_feature_names`
    is the subset actually returned by top_features for this event. A mention
    of any `known_feature_names` entry that isn't in `grounded_feature_names`
    fails validation — it means the LLM named a feature it didn't look up.

    Returns `(parsed_dict, None)` on success, or `(None, reason)` where reason
    is REASON_INVALID_JSON or REASON_UNGROUNDED — the caller falls back to a
    template note either way, but subagent.py logs (and evaluate.py reports)
    which reason it was.
    """
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
    """Best-effort salvage: pull a JSON object substring out of raw text that
    isn't itself valid JSON — e.g. the model wrote prose around the object
    instead of submitting it as final_answer's argument. Returns the
    substring from the first '{' to the last '}' if it parses as JSON, else
    None. The caller re-runs the extracted string through parse_and_validate
    (the same schema/grounding checks apply — this only recovers the JSON,
    it doesn't relax any guardrail)."""
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
