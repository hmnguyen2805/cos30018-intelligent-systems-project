"""
LLM access for the Judge: model configuration, the litellm client, the JSON
schemas the model must answer in, the prompts, and reply parsing.

Why JSON-schema replies instead of native tool calling: Vinh found (see
src/detection/docs/design-decisions.md on the detection-llm-mcp branch) that
litellm's Ollama backend drops `tool_choice`, so smolagents' ToolCallingAgent
is unreliable on local models, while a JSON-schema `response_format` is
enforced by Ollama. So the Judge's agent loop asks for one JSON object per
turn naming the next action, and code executes it (see agent.py).

Configuration (environment variables, all optional):
    JUDGE_LLM_MODEL       litellm model id (default: ollama_chat/qwen2.5:3b)
    JUDGE_LLM_API_KEY     API key for hosted models (e.g. Gemini); unused for Ollama
    JUDGE_LLM_TIMEOUT     seconds per LLM call (default: 60)
    JUDGE_LLM_MAX_STEPS   max agent-loop turns before falling back (default: 6)
    JUDGE_LLM_MAX_TOKENS  max output tokens per call (default: 400)
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

DEFAULT_MODEL = "ollama_chat/qwen2.5:3b"

# --- configuration -------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    timeout_seconds: float = 60.0
    max_steps: int = 6
    max_tokens: int = 400

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.environ.get("JUDGE_LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("JUDGE_LLM_API_KEY") or None,
            timeout_seconds=float(os.environ.get("JUDGE_LLM_TIMEOUT", 60)),
            max_steps=int(os.environ.get("JUDGE_LLM_MAX_STEPS", 6)),
            max_tokens=int(os.environ.get("JUDGE_LLM_MAX_TOKENS", 400)),
        )


# --- client --------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


# Anything with this call signature can be the Judge's LLM (tests pass a fake).
LLMClient = Callable[[List[Dict[str, str]], dict, str], LLMResponse]


class LiteLLMClient:
    """Calls a model through litellm with a JSON-schema response format."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def __call__(self, messages: List[Dict[str, str]], schema: dict, schema_name: str) -> LLMResponse:
        import litellm  # imported lazily so the rule-only Judge doesn't need it
        litellm.suppress_debug_info = True  # no "Give Feedback" banner on every failed call

        start = time.perf_counter()
        response = litellm.completion(
            model=self.config.model,
            api_key=self.config.api_key,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


# --- schemas -------------------------------------------------------------------

TOOL_ACTIONS = ("compare_conclusions", "lookup_playbook", "check_category_consistency")
FINALIZE = "finalize"
ESCALATE = "escalate"
REQUEST_RECHECK = "request_recheck"
AGENT_ACTIONS = TOOL_ACTIONS + (FINALIZE, ESCALATE)
RECHECK_MANAGERS = ("detection", "mitigation")
MAX_RECHECK_REASON_CHARS = 200


def agent_step_schema(allow_recheck: bool = False) -> dict:
    """The JSON shape of one agent turn. request_recheck (with "manager" and
    "reason") is only offered when a recheck is still available, so the model
    can't pick an action it isn't allowed to take."""
    actions = list(AGENT_ACTIONS) + ([REQUEST_RECHECK] if allow_recheck else [])
    properties = {
        "thought": {"type": "string", "maxLength": 200},
        "action": {"type": "string", "enum": actions},
        "technique_id": {"type": "string"},
        "final_action": {"type": "string"},
        "reasoning": {"type": "string", "maxLength": 300},
    }
    if allow_recheck:
        properties["manager"] = {"type": "string", "enum": list(RECHECK_MANAGERS)}
        properties["reason"] = {"type": "string", "maxLength": MAX_RECHECK_REASON_CHARS}
    return {"type": "object", "properties": properties, "required": ["thought", "action"]}


AGENT_STEP_SCHEMA = agent_step_schema(allow_recheck=False)

SINGLE_SHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [FINALIZE, ESCALATE]},
        "final_action": {"type": "string"},
        "reasoning": {"type": "string", "maxLength": 300},
    },
    "required": ["decision", "reasoning"],
}

# --- prompts -------------------------------------------------------------------

_ROLE = """You are the Judge in a cybersecurity triage system. Two managers have analysed one network traffic event:
- the Detection Manager decides whether the traffic is anomalous, and may tag an attack category in its notes, e.g. [category=DoS];
- the Mitigation Manager matches the event to a MITRE ATT&CK technique and proposes a response action.
Your job: decide whether the proposed response can be applied automatically (finalize) or must be reviewed by a human analyst (escalate).

Rules, checked in this order:
1. Read "detection_verdict" and the "*_confident" flags as given; don't re-derive them. A flag of null means "not applicable".
2. Benign case: if detection_verdict is "benign", detection_confident is true and matched_techniques is empty, finalize with "no_action". This is normal traffic; don't escalate it.
3. Escalate when: detection_confident is false, mitigation_confident is false, the managers disagree, mitigation failed, no technique matched a detected threat, or the evidence is inconsistent (for example Detection's category does not fit the matched technique).
4. Otherwise, when the threat is confirmed, both managers are confident and the evidence is consistent, finalize with the proposed action."""

AGENT_SYSTEM_PROMPT = _ROLE + """

Work step by step. Reply with ONE JSON object per turn:
{"thought": "<one short sentence>", "action": "<action>", ...}

Actions:
- compare_conclusions: see both managers' conclusions and the allowed final actions. Do this first.
- lookup_playbook: response guidance for a matched technique. Add "technique_id" (one of matched_techniques). Skip it if nothing matched.
- check_category_consistency: check whether Detection's category fits the matched technique.
- finalize: apply an action automatically. Add "final_action" (one of the allowed final actions) and "reasoning". Before finalizing with a response action (anything other than "no_action") you must have called check_category_consistency.
- escalate: send the case to a human analyst. Add "reasoning".

After each tool action you get its result as an observation; then choose your next action.
Keep "reasoning" under 300 characters and only cite facts you observed."""

_RECHECK_PROMPT = """

You may also send the case back to ONE manager to look again, once per manager:
- request_recheck: add "manager" ("detection" or "mitigation", one listed in rechecks_available) and a short "reason".
  Use it when the evidence is inconsistent or a manager seems to have missed something, before you escalate.
  A Detection recheck adds evidence (top-3 attack labels, more traffic features) but never changes its verdict or confidence.
  After a recheck you get the managers' updated conclusions; check category consistency again before finalizing with an action."""


def agent_system_prompt(allow_recheck: bool = False) -> str:
    return AGENT_SYSTEM_PROMPT + (_RECHECK_PROMPT if allow_recheck else "")


AGENT_TASK_MESSAGE = "A new case is ready for judgement. Start by calling compare_conclusions."

SINGLE_SHOT_SYSTEM_PROMPT = _ROLE + """

All the evidence is given below. Reply with ONE JSON object:
{"decision": "finalize" or "escalate", "final_action": "<one of the allowed final actions, if finalizing>", "reasoning": "<under 300 characters>"}"""


def observation_message(action: str, result: dict) -> Dict[str, str]:
    return {"role": "user", "content": f"Observation ({action}): {json.dumps(result, default=str)}"}


def invalid_reply_message(reason: str, allow_recheck: bool = False) -> Dict[str, str]:
    actions = list(AGENT_ACTIONS) + ([REQUEST_RECHECK] if allow_recheck else [])
    return {"role": "user", "content": f"Your last reply was not valid: {reason}. "
                                       f"Reply with one JSON object using one of these actions: {', '.join(actions)}."}


def single_shot_user_message(evidence: dict) -> Dict[str, str]:
    return {"role": "user", "content": "Evidence: " + json.dumps(evidence, default=str)}


# --- parsing -------------------------------------------------------------------


class LLMOutputError(ValueError):
    """The model's reply couldn't be used. `reason` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class AgentStep:
    thought: str
    action: str
    technique_id: Optional[str] = None
    final_action: Optional[str] = None
    reasoning: Optional[str] = None
    manager: Optional[str] = None        # request_recheck only
    recheck_reason: Optional[str] = None  # request_recheck only


def _load_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):  # tolerate ```json fences some models add
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMOutputError("invalid_json", str(exc)) from exc
    if not isinstance(data, dict):
        raise LLMOutputError("invalid_json", "expected a JSON object")
    return data


def parse_agent_step(raw: str, allowed_final_actions: List[str],
                     recheck_managers: Sequence[str] = ()) -> AgentStep:
    """Parse one agent turn. `recheck_managers` lists the managers the Judge may
    still send the case back to; empty means request_recheck isn't allowed."""
    data = _load_json_object(raw)
    action = data.get("action")
    if action == REQUEST_RECHECK:
        return _parse_recheck(data, recheck_managers)
    if action not in AGENT_ACTIONS:
        raise LLMOutputError("unknown_action", repr(action))
    thought = str(data.get("thought") or "")
    reasoning = str(data.get("reasoning") or thought)
    final_action = data.get("final_action")
    if action == FINALIZE and final_action not in allowed_final_actions:
        raise LLMOutputError("invalid_final_action",
                             f"{final_action!r} is not one of {allowed_final_actions}")
    return AgentStep(
        thought=thought,
        action=action,
        technique_id=data.get("technique_id"),
        final_action=final_action if action == FINALIZE else None,
        reasoning=reasoning if action in (FINALIZE, ESCALATE) else None,
    )


def _parse_recheck(data: dict, recheck_managers: Sequence[str]) -> AgentStep:
    if not recheck_managers:
        raise LLMOutputError("recheck_unavailable", "no recheck is available for this case")
    manager = data.get("manager")
    if manager not in recheck_managers:
        raise LLMOutputError("recheck_unavailable",
                             f"manager {manager!r} can't be rechecked; available: {list(recheck_managers)}")
    reason = " ".join(str(data.get("reason") or "").split())[:MAX_RECHECK_REASON_CHARS]
    if not reason:
        raise LLMOutputError("missing_reason", "request_recheck needs a short reason")
    return AgentStep(thought=str(data.get("thought") or ""), action=REQUEST_RECHECK,
                     manager=manager, recheck_reason=reason)


def parse_single_shot(raw: str, allowed_final_actions: List[str]) -> AgentStep:
    data = _load_json_object(raw)
    decision = data.get("decision")
    if decision not in (FINALIZE, ESCALATE):
        raise LLMOutputError("unknown_action", repr(decision))
    final_action = data.get("final_action")
    if decision == FINALIZE and final_action not in allowed_final_actions:
        raise LLMOutputError("invalid_final_action",
                             f"{final_action!r} is not one of {allowed_final_actions}")
    reasoning = str(data.get("reasoning") or "")
    return AgentStep(thought=reasoning, action=decision,
                     final_action=final_action if decision == FINALIZE else None, reasoning=reasoning)
