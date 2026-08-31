# Detection Manager

**Owner:** Vinh Nghiem

## Your role just changed — but your existing work didn't

Per the tutor's manager-subagent recommendation, Detection is now two layers: a
**Detection Manager** that owns the top-level contract, and a **Detection Subagent**
underneath it that does the actual classification work.


## Suggested structure

```
src/detection/
├── subagent.py     # your existing agent.py content, renamed — unchanged
├── manager.py       # new: DetectionManager
├── classifier.py    # still needs to be pushed — see blocker below
└── README.md
```

## Contract

```
DetectionManager.run(event: TrafficEvent) -> DetectionResult
```

Same shape the old `DetectionAgent` had — the Manager delegates to the subagent
internally and owns any manager-level oversight (e.g. later, deciding whether to trust
the subagent's result as-is, or dispatch to a second detection subagent if one gets
added). It doesn't need to be complex to satisfy the "delegation" requirement — the act
of delegating is what counts, not how much logic sits in the Manager itself.

A minimal version is fine to start:

```python
# manager.py
from src.detection.subagent import DetectionSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent


class DetectionManager(BaseAgent):
    name = "detection_manager"

    def __init__(self):
        super().__init__()
        self._subagent = DetectionSubagent()

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        self._trace = []
        self.log_step(thought="Delegate to Detection Subagent.", action="delegate_to_subagent")
        return self._subagent.run(input_data)
```

The Detection Manager's conclusion (`DetectionResult`) goes two places: across to the
Mitigation Manager (Callum), since Correlation needs to know what was detected, and
down to the Judge (Minh), who compares it against the Mitigation Manager's conclusion
to produce the final result.

## Dependencies

Add anything beyond the shared `requirements.txt` to a `requirements-detection.txt`
in this folder, and note it here.
