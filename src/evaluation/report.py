"""
Turns RunRecords into the evaluation's tables (spec section 6 metrics) and
writes them to disk:

    results.json   every run, with traces (the "execution traces" deliverable)
    runs.csv       one row per run, for spreadsheets or plots
    summary.md     the tables for the report

Metrics per Judge mode:
    success rate        outcome is one of the task's expected outcomes (and, when it
                        acted, the technique fits the true attack category)
    unsafe rate         acted when it shouldn't have, acted on the wrong technique,
                        or treated a threat as benign
    over-cautious rate  escalated a case that could have been handled automatically
    iterations, LLM calls, tokens, LLM latency, total latency (averages per run)
    tool-use success    agent mode: tool calls that returned a usable result
    fallback rate       runs where the LLM path failed and the rules decided
    recovery rate       invalid LLM replies that the LLM then corrected
    guardrail overrides, rechecks
    consistency         LLM modes: tasks where every repeat gave the same outcome
"""
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from src.evaluation.runner import RunRecord
from src.evaluation.tasks import DIFFICULTIES

CSV_FIELDS = ["task_id", "difficulty", "split", "mode", "repeat", "expected", "outcome", "correct", "unsafe",
              "over_cautious", "wrong_technique", "case", "decided_by", "technique", "recommended_action",
              "rechecks", "iterations", "llm_calls", "tokens", "llm_latency_ms", "total_ms",
              "invalid_replies", "recovered_replies", "fallback_reason", "tool_steps", "tool_ok",
              "mitigation_mode", "mit_llm_calls", "mit_tokens", "mit_llm_latency_ms", "mit_fallback_reason", "errors"]


def _rate(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


def _avg(values: List[float], digits: int = 1) -> str:
    return f"{sum(values) / len(values):.{digits}f}" if values else "-"


def _scored(records: Sequence[RunRecord]) -> List[RunRecord]:
    return [r for r in records if r.outcome != "skipped"]


def by_mode(records: Sequence[RunRecord]) -> Dict[str, List[RunRecord]]:
    groups: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in _scored(records):
        groups[r.mode].append(r)
    return dict(groups)


def mode_metrics(rs: List[RunRecord]) -> Dict[str, str]:
    n = len(rs)
    invalid = sum(r.invalid_replies for r in rs)
    mit = [r for r in rs if r.mitigation_mode == "agent"]  # runs where Mitigation used its LLM agent
    tool_steps = sum(r.tool_steps for r in rs)
    per_task = defaultdict(set)
    for r in rs:
        per_task[r.task_id].add(r.outcome)
    repeats = Counter(r.task_id for r in rs)
    multi = [t for t, c in repeats.items() if c > 1]
    return {
        "runs": str(n),
        "success": _rate(sum(r.correct for r in rs), n),
        "unsafe": _rate(sum(r.unsafe for r in rs), n),
        "over-cautious": _rate(sum(r.over_cautious for r in rs), n),
        "avg iterations": _avg([r.iterations for r in rs]),
        "avg LLM calls": _avg([r.llm_calls for r in rs]),
        "avg tokens": _avg([r.tokens for r in rs], 0),
        "avg LLM latency (s)": _avg([r.llm_latency_ms / 1000 for r in rs], 2),
        "avg total latency (s)": _avg([r.total_ms / 1000 for r in rs], 2),
        "Mitigation agent runs": _rate(len(mit), n),
        "Mitigation avg LLM calls": _avg([r.mit_llm_calls for r in mit]),
        "Mitigation avg tokens": _avg([r.mit_tokens for r in mit], 0),
        "Mitigation avg LLM latency (s)": _avg([r.mit_llm_latency_ms / 1000 for r in mit], 2),
        "Mitigation fallback rate": _rate(sum(bool(r.mit_fallback_reason) for r in mit), len(mit)) if mit else "-",
        "avg tokens (Judge + Mitigation)": _avg([r.tokens + r.mit_tokens for r in rs], 0),
        "tool-use success": _rate(sum(r.tool_ok for r in rs), tool_steps) if tool_steps else "-",
        "fallback rate": _rate(sum(r.decided_by == "rules_fallback" for r in rs), n),
        "recovery rate": _rate(sum(r.recovered_replies for r in rs), invalid) if invalid else "-",
        "guardrail overrides": str(sum(r.decided_by == "guardrail_override" for r in rs)),
        "rechecks": str(sum(r.rechecks for r in rs)),
        "consistency": _rate(sum(len(per_task[t]) == 1 for t in multi), len(multi)) if multi else "-",
    }


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def summary_markdown(records: Sequence[RunRecord], title: str = "Evaluation results") -> str:
    groups = by_mode(records)
    modes = list(groups)
    skipped = [r for r in records if r.outcome == "skipped"]
    task_ids = list(dict.fromkeys(r.task_id for r in records))
    out = [f"# {title}", "",
           f"{len(task_ids)} tasks, modes: {', '.join(modes) or 'none'}. "
           f"{len(skipped)} runs skipped." + (f" First skip reason: {skipped[0].skip_reason}" if skipped else ""),
           ""]

    # 1. Headline metrics per mode
    metrics = {m: mode_metrics(groups[m]) for m in modes}
    names = list(next(iter(metrics.values()))) if metrics else []
    out += ["## Metrics per Judge mode", "",
            _table(["metric"] + modes, [[n] + [metrics[m][n] for m in modes] for n in names]), ""]

    # 2. Success by difficulty and by split
    rows = []
    for d in DIFFICULTIES:
        row = [d]
        for m in modes:
            rs = [r for r in groups[m] if r.difficulty == d]
            row.append(_rate(sum(r.correct for r in rs), len(rs)))
        rows.append(row)
    for split in ("dev", "unseen"):
        row = [f"split: {split}"]
        for m in modes:
            rs = [r for r in groups[m] if r.split == split]
            row.append(_rate(sum(r.correct for r in rs), len(rs)))
        rows.append(row)
    out += ["## Success by difficulty and split", "", _table(["group"] + modes, rows), ""]

    # 3. Per task: what each mode did (outcome counts across repeats)
    rows = []
    for t in task_ids:
        first = next(r for r in records if r.task_id == t)
        row = [t, first.difficulty, "/".join(first.expected)]
        for m in modes:
            rs = [r for r in groups[m] if r.task_id == t]
            counts = Counter(r.outcome for r in rs)
            ok = sum(r.correct for r in rs)
            cell = ", ".join(f"{o} {c}" for o, c in counts.most_common())
            row.append(f"{cell} ({ok}/{len(rs)} ok)" if rs else "skipped")
        rows.append(row)
    out += ["## Per task", "", _table(["task", "difficulty", "expected"] + modes, rows), ""]

    # 4. Where the modes disagree on correctness (the interesting cases for the report)
    if len(modes) > 1:
        diffs = []
        for t in task_ids:
            oks = {m: [r.correct for r in groups[m] if r.task_id == t] for m in modes}
            rates = {m: (sum(v) / len(v) if v else None) for m, v in oks.items()}
            if len({v for v in rates.values() if v is not None}) > 1:
                diffs.append([t] + [f"{rates[m]:.0%}" if rates[m] is not None else "-" for m in modes])
        out += ["## Tasks where the modes differ", "",
                _table(["task"] + modes, diffs) if diffs else "None: every mode scored the same on every task.", ""]
    return "\n".join(out)


def write_results(records: Sequence[RunRecord], out_dir: Path, title: str = "Evaluation results") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "results.json", "csv": out_dir / "runs.csv", "summary": out_dir / "summary.md"}
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, default=str)
    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = asdict(r)
            row["expected"] = "/".join(r.expected)
            row["errors"] = "; ".join(f"{k}: {v}" for k, v in r.errors.items())
            writer.writerow(row)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write(summary_markdown(records, title))
    return paths