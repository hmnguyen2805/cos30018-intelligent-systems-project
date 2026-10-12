"""
Gradio UI for the triage pipeline (assignment criteria 5C).

    python -m src.ui.app            then open http://127.0.0.1:7860

What it covers from 5C:
    submit tasks               pick a demo scenario, or paste your own event's features
    view system progress       progress bar per stage (Detection, Mitigation, Judge, rechecks)
    inspect agent actions      every agent's trace: thought, action, input, observation
    view the final result      the Judge's decision, case, who decided, reasoning
    identify errors            run status, per-stage status and every error message

Demo scenarios need no trained model: Detection is scripted (see
scenarios.py), Mitigation and the Judge are the real ones. "My own event"
runs the real Detection Manager, so it needs the trained models
(python -m src.detection.train). Judge modes "single_shot" and "agent" need
the LLM (default: Ollama with qwen2.5:3b, see src/response/llm.py); if it's
not reachable, the Judge falls back to its rule table and the UI shows that.

The functions that turn a PipelineRun into tables (status_markdown,
stage_rows, trace_rows, ...) don't depend on Gradio, so they're tested in
tests/test_ui.py without starting a server.
"""
import json
import logging
from dataclasses import asdict, replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from src.correlation.manager import MitigationManager
from src.pipeline import Pipeline
from src.response.agent import MODES, MODE_RULES, JudgeAgent
from src.response.llm import LLMConfig
from src.shared.schemas import PipelineRun, TraceStep, TrafficEvent
from src.ui.scenarios import (
    DEFAULT_SCENARIO,
    SCENARIOS,
    FailingMitigationManager,
    ScriptedDetectionManager,
)

logger = logging.getLogger(__name__)

SOURCE_DEMO = "Demo scenario"
SOURCE_CUSTOM = "My own event (needs trained models)"

STAGE_LABELS = {
    "input": "Input",
    "detection": "Detection Manager",
    "mitigation": "Mitigation Manager",
    "judge": "Judge",
    "recheck_detection": "Recheck: Detection",
    "recheck_mitigation": "Recheck: Mitigation",
}
STAGE_PROGRESS = {"detection": 0.1, "mitigation": 0.35, "judge": 0.6,
                  "recheck_detection": 0.75, "recheck_mitigation": 0.85}

STAGE_HEADERS = ["Stage", "Status", "Time (ms)", "Result"]
TRACE_HEADERS = ["Agent", "Step", "Action", "Thought", "Input", "Observation"]
RECHECK_HEADERS = ["Manager", "Reason", "Before", "After", "Error", "Time (ms)"]

EXAMPLE_FEATURES = json.dumps({"Destination Port": 22, "Flow Duration": 1200000, "Total Fwd Packets": 3}, indent=2)


# --- building and running a pipeline ------------------------------------------------

class _LazyDetectionManager:
    """Builds the real Detection Manager on first use, so a missing model file
    shows up as a failed Detection stage in the UI instead of a crash."""
    name = "detection_manager"

    def run(self, event: TrafficEvent, recheck_reason: Optional[str] = None):
        return _real_detection_manager().run(event, recheck_reason=recheck_reason)


@lru_cache(maxsize=1)
def _real_detection_manager():
    from src.detection.manager import DetectionManager  # needs the trained models
    return DetectionManager()


def make_judge(mode: str, model: Optional[str] = None) -> JudgeAgent:
    config = LLMConfig.from_env()
    if model and model.strip():
        config = replace(config, model=model.strip())
    return JudgeAgent(mode=mode, config=config)


def build_pipeline(source: str, scenario_title: str, mode: str, model: Optional[str] = None,
                   on_stage: Optional[Callable[[str], None]] = None) -> Pipeline:
    judge = make_judge(mode, model)
    if source == SOURCE_DEMO:
        scenario = SCENARIOS[scenario_title]
        mitigation = FailingMitigationManager() if scenario.mitigation_fails else MitigationManager()
        return Pipeline(ScriptedDetectionManager(scenario), judge, mitigation, on_stage=on_stage)
    return Pipeline(_LazyDetectionManager(), judge, MitigationManager(), on_stage=on_stage)


def parse_features(text: str) -> Dict[str, float]:
    """Features JSON from the textbox -> {name: float}. Raises ValueError with a readable message."""
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Features must be a JSON object: {exc.msg} (line {exc.lineno}).") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError('Features must be a non-empty JSON object, e.g. {"Destination Port": 22}.')
    try:
        return {str(k): float(v) for k, v in data.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Every feature value must be a number.") from exc


def make_event(source: str, scenario_title: str, features_text: str) -> TrafficEvent:
    if source == SOURCE_DEMO:
        return TrafficEvent(features=dict(SCENARIOS[scenario_title].features))
    return TrafficEvent(features=parse_features(features_text))


# --- turning a PipelineRun into what the UI shows ----------------------------------------

def _short(value: Any, limit: int = 300) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def detection_summary(run: PipelineRun) -> str:
    d = run.detection
    if d is None:
        return ""
    verdict = "anomalous" if d.is_anomalous else "benign"
    parts = [f"{verdict} (confidence {d.confidence:.2f})"]
    if d.attack_category:
        parts.append(f"category {d.attack_category}")
    if d.attack_label:
        parts.append(f"label {d.attack_label}")
    return ", ".join(parts)


def mitigation_summary(run: PipelineRun) -> str:
    m = run.mitigation
    if m is None:
        return ""
    techniques = ", ".join(m.correlation.matched_technique_ids) or "none matched"
    return f"techniques {techniques} (confidence {m.confidence:.2f}); proposed: {m.proposed_action}"


def judge_summary(run: PipelineRun) -> str:
    r = run.response
    if r is None:
        return ""
    outcome = "escalated to a human" if r.escalated_to_human else f"action: {r.recommended_action}"
    return f"{outcome} (case {r.case}, decided by {r.decided_by})"


def stage_rows(run: PipelineRun) -> List[List[Any]]:
    summaries = {"detection": detection_summary(run), "mitigation": mitigation_summary(run),
                 "judge": judge_summary(run)}
    rows = []
    for stage in ("detection", "mitigation", "judge", "recheck_detection", "recheck_mitigation"):
        ran = stage in run.timings_ms
        error = run.errors.get(stage)
        if not ran and not error:
            if stage.startswith("recheck_"):
                continue  # rechecks only appear when they happened
            status, summary = "Skipped", "Not run (an earlier stage failed)"
        elif error:
            status, summary = "Failed", error
        else:
            status, summary = "OK", summaries.get(stage, "Done")
        timing = round(run.timings_ms[stage], 1) if ran else ""
        rows.append([STAGE_LABELS[stage], status, timing, summary])
    return rows


def _trace_rows(agent: str, trace: List[TraceStep]) -> List[List[Any]]:
    return [[agent, s.step_number, s.action or "", s.thought or "", _short(s.tool_input), _short(s.observation)]
            for s in trace]


def trace_rows(run: PipelineRun) -> List[List[Any]]:
    """Every agent's steps, in pipeline order. After a recheck, the managers'
    traces are from their latest run (the recheck)."""
    rows: List[List[Any]] = []
    if run.detection is not None:
        rows += _trace_rows("Detection Manager", run.detection.trace)
    if run.mitigation is not None:
        rows += _trace_rows("Correlation Subagent", run.mitigation.correlation.trace)
        rows += _trace_rows("Mitigation Manager", run.mitigation.trace)
    if run.response is not None:
        rows += _trace_rows("Judge", run.response.trace)
    return rows


def recheck_rows(run: PipelineRun) -> List[List[Any]]:
    return [[r.manager, r.reason, r.before, r.after, r.error or "", round(r.duration_ms, 1)] for r in run.rechecks]


def status_markdown(run: PipelineRun) -> str:
    if run.succeeded and not run.errors:
        return "### Status: run succeeded\nAll stages completed without errors."
    if run.succeeded:
        lines = ["### Status: run completed with errors",
                 "The Judge still produced a decision, but some stages failed:"]
    else:
        lines = ["### Status: run failed", "No final decision was produced."]
    lines += [f"- **{STAGE_LABELS.get(stage, stage)}**: {error}" for stage, error in run.errors.items()]
    return "\n".join(lines)


def result_markdown(run: PipelineRun) -> str:
    r = run.response
    if r is None:
        return "### Final result\nNone: the run stopped before the Judge decided."
    if r.escalated_to_human:
        headline = "**Escalated to a human analyst.** No automated action is taken."
    elif r.recommended_action == "no_action":
        headline = "**No action:** the traffic is treated as benign."
    else:
        headline = f"**Automated response:** {r.recommended_action}"
    fallback = r.llm_usage.get("fallback_reason")
    lines = [
        "### Final result",
        headline,
        "",
        f"- Case: `{r.case}`",
        f"- Decided by: `{r.decided_by}`" + (f" (LLM unavailable: {fallback})" if fallback else ""),
        f"- Managers agree: {'yes' if r.agents_agree else 'no'}",
        f"- Rechecks: {len(r.rechecks)}",
        "",
        f"**Reasoning:** {r.reasoning or ''}",
    ]
    return "\n".join(lines)


def metrics(run: PipelineRun) -> Dict[str, Any]:
    out: Dict[str, Any] = {"timings_ms": {k: round(v, 1) for k, v in run.timings_ms.items()}}
    if run.response is not None and run.response.llm_usage:
        out["judge_llm_usage"] = run.response.llm_usage
    return out


def raw_json(run: PipelineRun) -> str:
    return json.dumps(asdict(run), indent=2, default=str)


def execute(source: str, scenario_title: str, features_text: str, mode: str, model: Optional[str] = None,
            on_stage: Optional[Callable[[str], None]] = None) -> PipelineRun:
    """Run one task. Bad input becomes a failed run with the message, not an exception."""
    try:
        event = make_event(source, scenario_title, features_text)
    except ValueError as exc:
        run = PipelineRun(event=TrafficEvent(features={}))
        run.errors["input"] = str(exc)
        return run
    pipeline = build_pipeline(source, scenario_title, mode, model, on_stage)
    return pipeline.run(event)


def render(run: PipelineRun) -> tuple:
    return (status_markdown(run), result_markdown(run), stage_rows(run), trace_rows(run),
            recheck_rows(run), metrics(run), raw_json(run))


# --- the Gradio app -----------------------------------------------------------------------

def build_app():
    import gradio as gr  # imported here so the helpers above can be used without Gradio

    def on_run(source, scenario_title, features_text, mode, model, progress=gr.Progress()):
        progress(0.0, desc="Starting")
        def on_stage(stage):
            progress(STAGE_PROGRESS.get(stage, 0.5), desc=f"Running: {STAGE_LABELS.get(stage, stage)}")
        run = execute(source, scenario_title, features_text, mode, model, on_stage)
        progress(1.0, desc="Done")
        return render(run)

    def on_source(source):
        demo = source == SOURCE_DEMO
        return gr.update(visible=demo), gr.update(visible=demo), gr.update(visible=not demo)

    def on_scenario(title):
        return f"*{SCENARIOS[title].description}*"

    with gr.Blocks(title="Cyber Triage Agents", analytics_enabled=False) as app:
        gr.Markdown("# Cyber Triage: multi-agent pipeline\n"
                    "Detection Manager -> Mitigation Manager -> Judge. Submit an event, then inspect what "
                    "each agent did and why.")
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio([SOURCE_DEMO, SOURCE_CUSTOM], value=SOURCE_DEMO, label="Event source")
                scenario = gr.Dropdown(list(SCENARIOS), value=DEFAULT_SCENARIO, label="Scenario")
                scenario_info = gr.Markdown(on_scenario(DEFAULT_SCENARIO))
                features = gr.Code(EXAMPLE_FEATURES, language="json", label="Event features (CICIDS2017 columns)",
                                   visible=False)
                mode = gr.Radio(list(MODES), value=MODE_RULES, label="Judge mode",
                                info="rules: no LLM. single_shot: one LLM call. agent: the LLM agent loop "
                                     "(can recheck).")
                model = gr.Textbox(value=LLMConfig.from_env().model, label="Judge LLM model (litellm id)")
                run_button = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown()
                result = gr.Markdown()
                stages = gr.Dataframe(headers=STAGE_HEADERS, label="Stages", wrap=True, interactive=False,
                                      column_widths=["20%", "10%", "12%", "58%"])
        with gr.Tab("Agent trace"):
            trace = gr.Dataframe(headers=TRACE_HEADERS, label="Every step each agent took", wrap=True,
                                 interactive=False, column_widths=["12%", "5%", "15%", "23%", "15%", "30%"])
        with gr.Tab("Rechecks"):
            rechecks = gr.Dataframe(headers=RECHECK_HEADERS, label="Cases the Judge sent back", wrap=True,
                                    interactive=False, column_widths=["10%", "20%", "28%", "28%", "8%", "6%"])
        with gr.Tab("Metrics"):
            usage = gr.JSON(label="Timings and Judge LLM usage")
        with gr.Tab("Raw run (JSON)"):
            raw = gr.Code(language="json", label="PipelineRun")

        source.change(on_source, source, [scenario, scenario_info, features])
        scenario.change(on_scenario, scenario, scenario_info)
        run_button.click(on_run, [source, scenario, features, mode, model],
                         [status, result, stages, trace, rechecks, usage, raw])
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    build_app().launch()


if __name__ == "__main__":
    main()
