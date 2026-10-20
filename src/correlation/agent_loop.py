"""
LLM agent loop for the Mitigation Manager (MitigationManager(mode="agent"))

Same pattern as the Judge: a code-driven loop with JSON replies. Each turn the LLM answers with 1 JSON object

    "thought" -> "..." -> "action" -> "<tool or final action>" -> "..."

the code runs the chosen tool then logs it and sends the result back as an observation, looping until the LLM finalizes or the step limit is reached.

What the LLM can do (all read-only, no generated code is ever executed):

    get_detection_evidence      Detection's verdict, category, label, summary, notes
    list_candidate_techniques   techniques that fit Detection's category
    get_technique(technique_id) catalog description + playbook guidance + action
    search_catalog(query)       the Correlation Subagent's embedding search
    choose_technique            final: technique_ids (best first) + reasoning
    no_match                    final: nothing fits + reasoning

What code enforces, whatever the LLM says:

    * only technique ids that exist in TECHNIQUE_CATALOG are accepted
    * get_detection_evidence must be called before choosing
    * the LLM never writes the action text (the manager takes it from
      ACTION_BY_TECHNIQUE) and never sets the confidence (see _confidence_for)
    * an LLM error, a timeout, 2 invalid replies in a row, or the step limit
      raise MitigationFallback, and the manager uses its rules path instead

Configuration (environment variables, all optional):
    MITIGATION_LLM_MODEL       litellm model id (default: ollama_chat/qwen2.5:3b)
    MITIGATION_LLM_API_KEY     API key for hosted models; unused for Ollama
    MITIGATION_LLM_TIMEOUT     seconds per LLM call (default: 60)
    MITIGATION_LLM_MAX_STEPS   max loop turns before falling back (default: 6)
    MITIGATION_LLM_MAX_TOKENS  max output tokens per call (default: 400)
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.correlation.technique_catalog import CATEGORY_TECHNIQUES, TECHNIQUE_CATALOG
from src.response import llm
from src.shared.schemas import DetectionResult, JudgeInput
from src.shared.tags import parse_category, strip_tags

# --- actions ---

GET_EVIDENCE = "get_detection_evidence"
LIST_CANDIDATES = "list_candidate_techniques"
GET_TECHNIQUE = "get_technique"
SEARCH_CATALOG = "search_catalog"
CHOOSE_TECHNIQUE = "choose_technique"
NO_MATCH = "no_match"

TOOL_ACTIONS = (GET_EVIDENCE, LIST_CANDIDATES, GET_TECHNIQUE, SEARCH_CATALOG)
FINAL_ACTIONS = (CHOOSE_TECHNIQUE, NO_MATCH)
ALL_ACTIONS = TOOL_ACTIONS + FINAL_ACTIONS

MAX_CONSECUTIVE_INVALID = 2        # invalid replies in a row before falling back to the rules
OBSERVATION_LOG_CHARS = 500

# A technique that has neither category support nor search evidence is capped
# below the Judge's threshold (MITIGATION_MIN_CONFIDENCE, 0.6), so the Judge escalates.
UNSUPPORTED_CONFIDENCE_CAP = 0.5


def mitigation_llm_config() -> llm.LLMConfig:
    return llm.LLMConfig(
        #it reads the model
        model=os.environ.get("MITIGATION_LLM_MODEL", llm.DEFAULT_MODEL),
        #API key
        api_key=os.environ.get("MITIGATION_LLM_API_KEY") or None,
        #timeout
        timeout_seconds=float(os.environ.get("MITIGATION_LLM_TIMEOUT", 60)),
        #max steps
        max_steps=int(os.environ.get("MITIGATION_LLM_MAX_STEPS", 6)),
        #and max tokens
        max_tokens=int(os.environ.get("MITIGATION_LLM_MAX_TOKENS", 400)),
        #from the environment variables and returns them as a single object, if it isn't set it uses the default
    )

#when an error happens, it stops and jumps out and returns the error
class MitigationFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
#what the agent decides is the final answer
class AgentOutcome:
    technique_ids: List[str]           # empty = no_match
    confidence: float                  # set by code from the evidence, never by the LLM
    reasoning: str                     # the LLM's explanation, kept for the trace and notes
    source: str                        # "category" | "search" | "unsupported" | "no_match"
    usage: Dict[str, Any] = field(default_factory=dict)


# --- schema and prompts ---

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 200},
        "action": {"type": "string", "enum": list(ALL_ACTIONS)},
        "technique_id": {"type": "string"},
        "query": {"type": "string", "maxLength": 300},
        "technique_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "reasoning": {"type": "string", "maxLength": 300},
    },
    "required": ["thought", "action"],
}

SYSTEM_PROMPT = """You are the Mitigation Manager in a cybersecurity triage system. The Detection Manager has flagged one network traffic event as an attack. Your job: pick the MITRE ATT&CK technique that best matches the event, so a response action can be proposed for it.

Work step by step. Reply with ONE JSON object per turn:
{"thought": "<one short sentence>", "action": "<action>", ...}

Actions:
- get_detection_evidence: Detection's verdict, attack category, attack label, traffic summary and notes. Always do this first.
- list_candidate_techniques: the techniques that fit Detection's attack category, with descriptions.
- get_technique: details for one technique. Add "technique_id".
- search_catalog: search the whole technique catalog with free text. Add "query". Useful when the category is "Unknown" or none of the candidates fit.
- choose_technique: final answer. Add "technique_ids" (a list of technique ids, most likely first) and "reasoning".
- no_match: final answer when no technique fits. Add "reasoning".

After each tool action you get its result as an observation; then choose your next action.
Only use technique ids you have seen in an observation. You do not write the response action and you do not give a confidence: both are decided by the system from the evidence.
Keep "reasoning" under 300 characters and only cite facts you observed."""

TASK_MESSAGE = "A new anomalous event needs a technique match. Start by calling get_detection_evidence."

RECHECK_MESSAGE = (
    "A new anomalous event needs a technique match. This is a RECHECK: the Judge sent the case back "
    "for a second look.\nJudge's reason: {reason}\n{previous}"
    "Try something different from the first pass: for example search the catalog with the traffic "
    "summary, or consider the other candidate techniques instead of the first one. "
    "Start by calling get_detection_evidence."
)


def first_message(recheck_reason: Optional[str], previous_ids: Optional[List[str]] = None) -> Dict[str, str]:
    if recheck_reason:
        previous = f"Your first pass chose: {', '.join(previous_ids)}.\n" if previous_ids else ""
        return {"role": "user", "content": RECHECK_MESSAGE.format(reason=recheck_reason, previous=previous)}
    return {"role": "user", "content": TASK_MESSAGE}


def invalid_reply_message(reason: str) -> Dict[str, str]:
    return {"role": "user", "content": f"Your last reply was not valid: {reason}. "
                                       f"Reply with one JSON object using one of these actions: {', '.join(ALL_ACTIONS)}."}


# --- parsing ---


@dataclass(frozen=True)
class Step:
    thought: str
    action: str
    technique_id: Optional[str] = None
    query: Optional[str] = None
    technique_ids: Optional[List[str]] = None
    reasoning: Optional[str] = None


def parse_step(raw: str) -> Step:
    """Parse one turn. Raises llm.LLMOutputError (reason + detail) if it can't be used.
    Whether the ids are real techniques is checked here; whether the evidence rules
    allow the action is checked by the loop."""
    text = (raw or "").strip()
    if text.startswith("```"):  # tolerate ```json fences some models add
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise llm.LLMOutputError("invalid_json", str(exc)) from exc
    if not isinstance(data, dict):
        raise llm.LLMOutputError("invalid_json", "expected a JSON object")

    action = data.get("action")
    if action not in ALL_ACTIONS:
        raise llm.LLMOutputError("unknown_action", repr(action))
    thought = str(data.get("thought") or "")
    reasoning = str(data.get("reasoning") or thought)

    if action == GET_TECHNIQUE:
        technique_id = data.get("technique_id")
        if not isinstance(technique_id, str) or not technique_id:
            raise llm.LLMOutputError("missing_technique_id", "get_technique needs a technique_id")
        return Step(thought, action, technique_id=technique_id)

    if action == SEARCH_CATALOG:
        query = " ".join(str(data.get("query") or "").split())
        if not query:
            raise llm.LLMOutputError("missing_query", "search_catalog needs a non-empty query")
        return Step(thought, action, query=query)

    if action == CHOOSE_TECHNIQUE:
        ids = data.get("technique_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise llm.LLMOutputError("missing_technique_ids",
                                     "choose_technique needs technique_ids: a non-empty list of technique ids")
        unknown = [i for i in ids if i not in TECHNIQUE_CATALOG]
        if unknown:
            raise llm.LLMOutputError(
                "unknown_technique",
                f"{unknown} are not techniques in the catalog; valid ids are {list(TECHNIQUE_CATALOG)}")
        return Step(thought, action, technique_ids=list(dict.fromkeys(ids))[:4], reasoning=reasoning)

    if action == NO_MATCH:
        return Step(thought, action, reasoning=reasoning)

    return Step(thought, action)  # get_detection_evidence, list_candidate_techniques


# --- the loop ---


def _clip(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= OBSERVATION_LOG_CHARS else text[:OBSERVATION_LOG_CHARS] + "..."


class MitigationAgentLoop:
    """One run of the Mitigation agent for one DetectionResult.

    `log_step` is the manager's own log_step, so every LLM step lands in the
    Mitigation Manager's trace. `llm_client` is anything with the LLMClient
    signature (tests pass a scripted fake). `subagent` is the Correlation
    Subagent, used only for its embedding search.
    """

    def __init__(self, detection: DetectionResult, subagent, llm_client: llm.LLMClient,
                 config: llm.LLMConfig, log_step: Callable[..., None],
                 recheck_reason: Optional[str] = None, previous_ids: Optional[List[str]] = None):
        self.detection = detection
        self.previous_ids = previous_ids
        self.subagent = subagent
        self.llm_client = llm_client
        self.config = config
        self.log_step = log_step
        self.recheck_reason = recheck_reason

        self.category = detection.attack_category or parse_category(detection.detector_notes)
        self.candidates: List[str] = list(CATEGORY_TECHNIQUES.get(self.category, [])) if self.category else []
        self.search_scores: Dict[str, float] = {}   # technique id -> best search_catalog score
        self.evidence_seen = False
        self.calls_made = set()
        self.usage = {"model": config.model, "llm_calls": 0, "iterations": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "llm_latency_ms": 0.0, "invalid_replies": 0,
                      "recovered_replies": 0}

    # --- tools (read-only) ---

    def _technique_info(self, technique_id: str) -> dict:
        from src.correlation.manager import ACTION_BY_TECHNIQUE  # lazy: manager imports this module
        return {
            "technique_id": technique_id,
            "description": TECHNIQUE_CATALOG[technique_id],
            "proposed_action": ACTION_BY_TECHNIQUE.get(technique_id),
        }

    def tool_get_detection_evidence(self) -> dict:
        d = self.detection
        return {
            "verdict": "anomalous" if d.is_anomalous else "benign",
            "confidence": d.confidence,
            "attack_category": self.category,
            "attack_label": d.attack_label,
            "category_confidence": d.category_confidence,
            "traffic_summary": d.traffic_summary,
            "notes": strip_tags(d.detector_notes),
        }

    def tool_list_candidate_techniques(self) -> dict:
        if not self.candidates:
            return {"attack_category": self.category, "candidates": [],
                    "note": "No techniques are listed for this category; try search_catalog."}
        return {"attack_category": self.category,
                "candidates": [self._technique_info(t) for t in self.candidates]}

    def tool_get_technique(self, technique_id: str) -> dict:
        if technique_id not in TECHNIQUE_CATALOG:
            return {"error": f"{technique_id!r} is not in the catalog.", "valid_ids": list(TECHNIQUE_CATALOG)}
        info = self._technique_info(technique_id)
        info["in_category_candidates"] = technique_id in self.candidates
        from src.response import tools  # lazy: keeps the import graph simple
        info["playbook"] = tools.lookup_playbook(tools.JudgeContext(
            JudgeInput(detection=self.detection)), technique_id)
        return info

    def tool_search_catalog(self, query: str) -> dict:
        embeddings = self.subagent._get_technique_embeddings()
        if embeddings is None:
            return {"error": "Embedding search is unavailable (sentence-transformers is not installed "
                             "or the model could not be loaded). Use the other tools."}
        technique_id, score = self.subagent._query_catalog(query)
        self.search_scores[technique_id] = max(score, self.search_scores.get(technique_id, 0.0))
        return {"best_technique": self._technique_info(technique_id), "score": round(float(score), 3)}

    def _run_tool(self, step: Step) -> dict:
        if step.action == GET_EVIDENCE:
            self.evidence_seen = True
            return self.tool_get_detection_evidence()
        if step.action == LIST_CANDIDATES:
            return self.tool_list_candidate_techniques()
        if step.action == GET_TECHNIQUE:
            return self.tool_get_technique(step.technique_id)
        return self.tool_search_catalog(step.query)

    # --- rules enforced in code ---

    @staticmethod
    def _call_key(step: Step):
        query = step.query.lower() if step.query else None
        return (step.action, step.technique_id, query)

    def _check_preconditions(self, step: Step) -> None:
        if step.action in FINAL_ACTIONS and not self.evidence_seen:
            raise llm.LLMOutputError("no_evidence", f"call {GET_EVIDENCE} before {step.action}")
        if step.action in TOOL_ACTIONS and self._call_key(step) in self.calls_made:
            raise llm.LLMOutputError(
                "repeated_call",
                f"you already called {step.action} with the same input and its result is in the "
                f"conversation above. Do not repeat it: use a different tool or input, or give a final answer")

    def _confidence_for(self, technique_id: str):
        """Confidence comes from the evidence for the FIRST chosen technique (the one whose
        action is proposed); the LLM never sets it. Returns (confidence, source)."""
        from src.correlation.subagent import CATEGORY_MATCH_CONFIDENCE
        if technique_id in self.candidates:
            return CATEGORY_MATCH_CONFIDENCE, "category"
        if technique_id in self.search_scores:
            return float(self.search_scores[technique_id]), "search"
        return UNSUPPORTED_CONFIDENCE_CAP, "unsupported"

    # --- the loop ---

    def run(self) -> AgentOutcome:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            first_message(self.recheck_reason, self.previous_ids),
        ]
        consecutive_invalid = 0

        for _ in range(self.config.max_steps):
            self.usage["iterations"] += 1
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response.content})

            try:
                step = parse_step(response.content)
                self._check_preconditions(step)
            except llm.LLMOutputError as err:
                consecutive_invalid += 1
                self.usage["invalid_replies"] += 1
                self.log_step(thought="LLM reply rejected.", action="invalid_llm_reply",
                              tool_input={"reason": err.reason}, observation=_clip(response.content))
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    raise MitigationFallback(f"invalid_reply:{err.reason}")
                messages.append(invalid_reply_message(str(err)))
                continue

            if consecutive_invalid:
                self.usage["recovered_replies"] += 1
            consecutive_invalid = 0

            if step.action == NO_MATCH:
                self.log_step(thought=step.thought, action=f"llm_{NO_MATCH}", observation=step.reasoning)
                return AgentOutcome([], 0.0, step.reasoning or "", "no_match", dict(self.usage))

            if step.action == CHOOSE_TECHNIQUE:
                confidence, source = self._confidence_for(step.technique_ids[0])
                self.log_step(thought=step.thought, action=f"llm_{CHOOSE_TECHNIQUE}",
                              tool_input={"technique_ids": step.technique_ids},
                              observation=f"{step.reasoning} [confidence={confidence:.2f} from {source}]")
                return AgentOutcome(step.technique_ids, confidence, step.reasoning or "", source,
                                    dict(self.usage))

            try:
                result = self._run_tool(step)
            except Exception as exc:  # a tool crashed: tell the LLM instead of failing the whole stage
                result = {"error": f"{step.action} failed: {type(exc).__name__}: {exc}"}
            self.calls_made.add(self._call_key(step))
            tool_input = {k: v for k, v in (("technique_id", step.technique_id), ("query", step.query)) if v}
            self.log_step(thought=step.thought, action=step.action, tool_input=tool_input or None,
                          observation=_clip(result))
            messages.append(llm.observation_message(step.action, result))

        raise MitigationFallback("max_steps")

    def _call_llm(self, messages: List[Dict[str, str]]) -> llm.LLMResponse:
        try:
            response = self.llm_client(messages, STEP_SCHEMA, "mitigation_step")
        except Exception as exc:  # timeouts, connection errors, provider errors
            self.log_step(thought="LLM call failed.", action="llm_error",
                          observation=_clip(f"{type(exc).__name__}: {exc}"))
            raise MitigationFallback(f"llm_error:{type(exc).__name__}") from exc
        self.usage["llm_calls"] += 1
        self.usage["prompt_tokens"] += response.prompt_tokens
        self.usage["completion_tokens"] += response.completion_tokens
        self.usage["llm_latency_ms"] += response.latency_ms
        return response