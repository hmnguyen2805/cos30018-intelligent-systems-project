"""
Runs evaluation tasks through the full pipeline in each Judge mode and
records what happened, one RunRecord per (task, mode, repeat).

    run_evaluation(tasks, modes=("rules", "single_shot", "agent"), repeats=3)

Each run builds a fresh pipeline:
    Detection   scripted from the task, or the real trained Detection Manager
    Mitigation  the real Mitigation Manager, or scripted from the task
    Judge       the real JudgeAgent in the mode being evaluated

The pipeline gives the Judge its recheck callback as usual, so agent-mode
rechecks happen here exactly as they do in the UI.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.pipeline import Pipeline
from src.response.agent import MODE_RULES, MODES, JudgeAgent
from src.response.llm import TOOL_ACTIONS, LLMConfig
from src.response.tools import load_playbook
from src.shared.schemas import PipelineRun
from src.evaluation.tasks import EvalTask, ScriptedDetection, ScriptedMitigation, event_for

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """One task run in one mode. Everything the report's metrics are computed from."""
    task_id: str
    difficulty: str
    split: str
    mode: str
    repeat: int
    expected: List[str]
    outcome: str = "failed"              # act | escalate | no_action | failed | skipped
    correct: bool = False
    unsafe: bool = False                 # acted when it shouldn't, wrong technique, or missed a threat
    over_cautious: bool = False          # escalated a case that could have been handled automatically
    wrong_technique: bool = False
    case: Optional[str] = None
    decided_by: Optional[str] = None
    recommended_action: Optional[str] = None
    technique: Optional[str] = None
    rechecks: int = 0
    iterations: int = 0
    llm_calls: int = 0
    tokens: int = 0
    llm_latency_ms: float = 0.0
    total_ms: float = 0.0
    invalid_replies: int = 0
    recovered_replies: int = 0
    fallback_reason: Optional[str] = None
    tool_steps: int = 0                  # agent mode only: tool calls the LLM chose
    tool_ok: int = 0                     # ... and how many returned a usable result
    mitigation_mode: str = ""            # "rules" | "agent" for the real Mitigation Manager, "scripted" for a scripted one
    mit_llm_calls: int = 0               # Mitigation agent LLM usage over all its runs in this case (a recheck is a second run)
    mit_tokens: int = 0
    mit_llm_latency_ms: float = 0.0
    mit_fallback_reason: Optional[str] = None   # why the Mitigation agent fell back to its rules, if it did
    errors: Dict[str, str] = field(default_factory=dict)
    skip_reason: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


# --- building the pipeline for a task ----------------------------------------------

@lru_cache(maxsize=1)
def _real_detection_manager():
    from src.detection.manager import DetectionManager  # needs the trained models in models/
    return DetectionManager()


def _detection_for(task: EvalTask):
    if task.uses_real_detection:
        return _real_detection_manager()
    return ScriptedDetection(task.detection)


class MitigationUsage:
    """Wraps a Mitigation Manager and adds up its LLM usage over every run() call. The manager's
    last_usage only keeps the latest run, so without this a Judge recheck (a second Mitigation
    run) would hide the first run's tokens. A scripted manager has no usage and adds nothing."""

    def __init__(self, manager):
        self.manager = manager
        self.mode = getattr(manager, "mode", "scripted")
        self.llm_calls = 0
        self.tokens = 0
        self.llm_latency_ms = 0.0
        self.fallback_reason: Optional[str] = None

    def run(self, *args, **kwargs):
        try:
            return self.manager.run(*args, **kwargs)
        finally:
            usage = getattr(self.manager, "last_usage", None) or {}
            self.llm_calls += usage.get("llm_calls", 0)
            self.tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            self.llm_latency_ms += usage.get("llm_latency_ms", 0.0)
            self.fallback_reason = self.fallback_reason or usage.get("fallback_reason")


def _mitigation_for(task: EvalTask):
    if task.mitigation == "real":
        from src.correlation.manager import mitigation_from_env
        return MitigationUsage(mitigation_from_env())
    return MitigationUsage(ScriptedMitigation(task.mitigation))


def build_pipeline(task: EvalTask, judge: JudgeAgent) -> Pipeline:
    return Pipeline(_detection_for(task), judge, _mitigation_for(task))


# --- judging the result -------------------------------------------------------------

def classify_outcome(run: PipelineRun) -> str:
    r = run.response
    if r is None:
        return "failed"
    if r.escalated_to_human:
        return "escalate"
    if r.recommended_action == "no_action":
        return "no_action"
    return "act"


def technique_fits(technique: Optional[str], category: Optional[str]) -> bool:
    """Does the technique the system acted on fit the true attack category?"""
    if not technique or not category:
        return False
    return technique in load_playbook()["category_techniques"].get(category, [])


def score(record: RunRecord, task: EvalTask) -> None:
    """Fill in correct / unsafe / over_cautious from the outcome."""
    o = record.outcome
    if o == "act" and task.true_category:
        record.wrong_technique = not technique_fits(record.technique, task.true_category)
    record.correct = o in task.expected and not record.wrong_technique
    record.unsafe = (o == "act" and ("act" not in task.expected or record.wrong_technique)) or \
                    (o == "no_action" and "no_action" not in task.expected)
    record.over_cautious = o == "escalate" and "escalate" not in task.expected


def _trace_rows(run: PipelineRun) -> List[Dict[str, Any]]:
    rows = []
    parts = [("detection", run.detection), ("mitigation", run.mitigation), ("judge", run.response)]
    for agent, result in parts:
        if result is None:
            continue
        for s in result.trace:
            rows.append({"agent": agent, "step": s.step_number, "action": s.action, "thought": s.thought,
                         "input": s.tool_input, "observation": s.observation})
    return rows


def record_from_run(task: EvalTask, mode: str, repeat: int, run: PipelineRun, keep_trace: bool,
                    mitigation: Optional[MitigationUsage] = None) -> RunRecord:
    record = RunRecord(task_id=task.id, difficulty=task.difficulty, split=task.split, mode=mode,
                       repeat=repeat, expected=list(task.expected))
    record.outcome = classify_outcome(run)
    record.errors = dict(run.errors)
    record.total_ms = sum(v for k, v in run.timings_ms.items() if not k.startswith("recheck_"))
    if run.mitigation is not None and run.mitigation.correlation.matched_technique_ids:
        record.technique = run.mitigation.correlation.matched_technique_ids[0]
    if mitigation is not None:
        record.mitigation_mode = mitigation.mode
        record.mit_llm_calls = mitigation.llm_calls
        record.mit_tokens = mitigation.tokens
        record.mit_llm_latency_ms = mitigation.llm_latency_ms
        record.mit_fallback_reason = mitigation.fallback_reason

    r = run.response
    if r is not None:
        u = r.llm_usage or {}
        record.case, record.decided_by, record.recommended_action = r.case, r.decided_by, r.recommended_action
        record.rechecks = len(r.rechecks)
        record.iterations = u.get("iterations", 0)
        record.llm_calls = u.get("llm_calls", 0)
        record.tokens = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        record.llm_latency_ms = u.get("llm_latency_ms", 0.0)
        record.invalid_replies = u.get("invalid_replies", 0)
        record.recovered_replies = u.get("recovered_replies", 0)
        record.fallback_reason = u.get("fallback_reason")
        if mode == "agent":  # only in agent mode does the LLM choose the tools
            steps = [s for s in r.trace if s.action in TOOL_ACTIONS]
            record.tool_steps = len(steps)
            record.tool_ok = sum('"error"' not in (s.observation or "") for s in steps)

    score(record, task)
    if keep_trace:
        record.trace = _trace_rows(run)
    return record


# --- the main loop ------------------------------------------------------------------

def run_task(task: EvalTask, mode: str, repeat: int = 0, config: Optional[LLMConfig] = None,
             llm_client=None, keep_trace: bool = True) -> RunRecord:
    judge = JudgeAgent(mode=mode, llm_client=llm_client, config=config or LLMConfig.from_env())
    try:
        pipeline = build_pipeline(task, judge)
    except Exception as exc:  # e.g. the real Detection model isn't trained on this machine
        return RunRecord(task_id=task.id, difficulty=task.difficulty, split=task.split, mode=mode,
                         repeat=repeat, expected=list(task.expected), outcome="skipped",
                         skip_reason=f"{type(exc).__name__}: {exc}")
    run = pipeline.run(event_for(task))
    return record_from_run(task, mode, repeat, run, keep_trace, pipeline.mitigation_manager)


def run_evaluation(tasks: Sequence[EvalTask], modes: Sequence[str] = MODES, repeats: int = 3,
                   config: Optional[LLMConfig] = None, llm_client=None, keep_trace: bool = True,
                   progress: Optional[Callable[[str], None]] = None) -> List[RunRecord]:
    """Every task in every mode. `rules` is deterministic, so it runs once; the
    LLM modes run `repeats` times because model output varies between runs."""
    records: List[RunRecord] = []
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        n = 1 if mode == MODE_RULES else repeats
        for repeat in range(n):
            for i, task in enumerate(tasks, 1):
                start = time.perf_counter()
                record = run_task(task, mode, repeat, config, llm_client, keep_trace)
                records.append(record)
                if progress:
                    progress(f"[{mode} run {repeat + 1}/{n}] {i}/{len(tasks)} {task.id}: {record.outcome} "
                             f"({'ok' if record.correct else 'WRONG' if record.outcome != 'skipped' else 'skipped'}, "
                             f"{time.perf_counter() - start:.1f}s)")
    return records


def records_to_json(records: Sequence[RunRecord]) -> str:
    return json.dumps([asdict(r) for r in records], indent=2, default=str)