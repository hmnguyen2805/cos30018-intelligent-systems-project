"""
Pipeline: runs one TrafficEvent through the manager-subagent system.

    Detection Manager -> Mitigation Manager -> Judge  =>  PipelineRun

Deliberately thin. Its jobs are ordering, catching each stage's failures so
one broken agent doesn't take down the run, and recording timings for the
report. All decision-making lives in the agents.

Failure handling:
    detection fails  -> run stops (nothing to judge); error recorded
    mitigation fails -> Judge still runs, told why mitigation is missing
    no mitigation manager configured (e.g. not built yet) -> same as above
    judge fails      -> error recorded, no final response

Rechecks (two-way coordination): the pipeline gives the Judge a callback,
JudgeInput.recheck(manager, reason). When the agent-mode Judge asks for one:

    "detection"   Detection Manager runs again with recheck_reason, then the
                  Mitigation Manager runs again (normally) on the new result,
                  since its analysis is built on Detection's output.
    "mitigation"  Mitigation Manager runs again with recheck_reason.

Each manager can be rechecked at most once per case. The run keeps the latest
conclusions in run.detection / run.mitigation, the Judge records each recheck
(run.rechecks), and the time is in timings_ms["recheck_<manager>"] (it is also
part of the "judge" time, since the recheck happens while the Judge works).
If the manager fails during a recheck, the error is recorded and the Judge
carries on with the earlier conclusions.
"""
import logging
import time
from dataclasses import replace
from typing import Any, Callable, List, Optional

from src.shared.schemas import JudgeInput, PipelineRun, TrafficEvent

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, detection_manager, judge, mitigation_manager: Optional[Any] = None,
                 on_stage: Optional[Callable[[str], None]] = None):
        """`on_stage(name)` is called as each stage (or recheck) starts, so a UI
        can show progress. Optional; errors in it are logged and ignored."""
        self.detection_manager = detection_manager
        self.mitigation_manager = mitigation_manager
        self.judge = judge
        self.on_stage = on_stage

    def run(self, event: TrafficEvent) -> PipelineRun:
        run = PipelineRun(event=event)

        run.detection = self._stage(run, "detection", lambda: self.detection_manager.run(event))
        if run.detection is None:
            return run

        if self.mitigation_manager is None:
            run.errors["mitigation"] = "Mitigation Manager not configured"
            logger.warning("Mitigation stage skipped: no Mitigation Manager configured")
        else:
            run.mitigation = self._stage(
                run, "mitigation", lambda: self.mitigation_manager.run(run.detection)
            )

        available = ["detection"] + (["mitigation"] if self.mitigation_manager is not None else [])
        judge_input = self._judge_input(run, available)
        run.response = self._stage(run, "judge", lambda: self.judge.run(judge_input))
        if run.response is not None:
            run.rechecks = list(run.response.rechecks)
        return run

    # --- rechecks --------------------------------------------------------------

    def _judge_input(self, run: PipelineRun, available: List[str]) -> JudgeInput:
        return JudgeInput(
            detection=run.detection,
            mitigation=run.mitigation,
            mitigation_error=run.errors.get("mitigation"),
            recheck=lambda manager, reason: self._recheck(run, available, manager, reason),
            rechecks_available=list(available),
        )

    def _recheck(self, run: PipelineRun, available: List[str], manager: str, reason: str) -> JudgeInput:
        """Re-run one manager for the Judge and return the updated JudgeInput.
        Raises if that manager fails, so the Judge can record it."""
        if manager not in available:
            raise ValueError(f"Recheck of {manager!r} not available (remaining: {available}).")
        available.remove(manager)  # once per manager, even if this attempt fails
        self._notify(f"recheck_{manager}")
        logger.info("Recheck %s: %s", manager, reason)
        start = time.perf_counter()
        try:
            if manager == "detection":
                run.detection = self.detection_manager.run(run.event, recheck_reason=reason)
                # Mitigation's analysis is built on Detection's output, so refresh it too.
                run.errors.pop("mitigation", None)
                if self.mitigation_manager is not None:
                    try:
                        run.mitigation = self.mitigation_manager.run(run.detection)
                    except Exception as exc:
                        run.mitigation = None
                        run.errors["mitigation"] = f"{type(exc).__name__}: {exc}"
                        logger.exception("Mitigation failed after the Detection recheck")
                else:
                    run.errors["mitigation"] = "Mitigation Manager not configured"
            else:
                run.mitigation = self.mitigation_manager.run(run.detection, recheck_reason=reason)
                run.errors.pop("mitigation", None)
        except Exception as exc:
            run.errors[f"recheck_{manager}"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run.timings_ms[f"recheck_{manager}"] = (time.perf_counter() - start) * 1000
        return self._judge_input(run, available)

    def _notify(self, name: str) -> None:
        if self.on_stage is None:
            return
        try:
            self.on_stage(name)
        except Exception:  # a progress display must never break a run
            logger.exception("on_stage callback failed for %s", name)

    def _stage(self, run: PipelineRun, name: str, fn: Callable[[], Any]) -> Any:
        """Run one stage, recording its duration, and its error if it raises."""
        self._notify(name)
        logger.info("Stage %s: start", name)
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # any agent failure is recorded, not propagated
            run.errors[name] = f"{type(exc).__name__}: {exc}"
            logger.exception("Stage %s: failed", name)
            result = None
        finally:
            run.timings_ms[name] = (time.perf_counter() - start) * 1000
        logger.info("Stage %s: done in %.1f ms", name, run.timings_ms[name])
        return result


def build_default_pipeline() -> Pipeline:
    """The real system as it currently exists. Imports are local so importing
    this module doesn't require a trained detection model.

    The Mitigation Manager loads its embedding model lazily, only for events
    Detection couldn't categorise, so building the pipeline stays cheap.

    The Judge's mode comes from JUDGE_MODE (rules | single_shot | agent;
    default rules), see src/response/agent.py.
    """
    from src.correlation.manager import MitigationManager
    from src.detection.manager import DetectionManager
    from src.response.agent import judge_from_env

    return Pipeline(
        detection_manager=DetectionManager(), judge=judge_from_env(), mitigation_manager=MitigationManager()
    )
