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

Week 9 adds the Judge's recheck requests (sending a case back to a manager).
"""
import logging
import time
from typing import Any, Callable, Optional

from src.shared.schemas import JudgeInput, PipelineRun, TrafficEvent

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, detection_manager, judge, mitigation_manager: Optional[Any] = None):
        self.detection_manager = detection_manager
        self.mitigation_manager = mitigation_manager
        self.judge = judge

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

        judge_input = JudgeInput(
            detection=run.detection,
            mitigation=run.mitigation,
            mitigation_error=run.errors.get("mitigation"),
        )
        run.response = self._stage(run, "judge", lambda: self.judge.run(judge_input))
        return run

    @staticmethod
    def _stage(run: PipelineRun, name: str, fn: Callable[[], Any]) -> Any:
        """Run one stage, recording its duration, and its error if it raises."""
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

    Mitigation Manager is None until Callum's implementation lands; the Judge
    handles that as incomplete input.
    """
    from src.detection.manager import DetectionManager
    from src.response.agent import JudgeAgent

    return Pipeline(detection_manager=DetectionManager(), judge=JudgeAgent(), mitigation_manager=None)
