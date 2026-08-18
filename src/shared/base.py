"""
The one interface every agent implements, plus a shared tracing helper.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.shared.schemas import TraceStep


class BaseAgent(ABC):
    """
    Subclass this, set `name`, implement `run()`. Call `self.log_step(...)`
    from inside your loop each time you reason or call a tool — that's what
    lets the pipeline/UI show what your agent actually did.
    """
    name: str = "base_agent"

    def __init__(self):
        self._trace: List[TraceStep] = []

    def log_step(self, thought: Optional[str] = None, action: Optional[str] = None,
                 tool_input: Optional[dict] = None, observation: Optional[str] = None):
        self._trace.append(TraceStep(
            step_number=len(self._trace) + 1,
            thought=thought, action=action, tool_input=tool_input, observation=observation,
        ))

    def get_trace(self) -> List[TraceStep]:
        return list(self._trace)

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Process input_data and return this agent's result."""
        raise NotImplementedError