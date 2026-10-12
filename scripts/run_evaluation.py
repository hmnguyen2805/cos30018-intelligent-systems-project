"""
Run the system evaluation: every task through the full pipeline in each
Judge mode, then write the results and the summary tables.

Usage (from the repo root):
    python scripts/run_evaluation.py                          # dev tasks, rules mode only (no LLM)
    python scripts/run_evaluation.py --modes all --repeats 3  # all three modes (needs Ollama)
    python scripts/run_evaluation.py --tasks src/evaluation/tasks/dev_tasks.json evaluation/tasks/unseen_tasks.json
    python scripts/run_evaluation.py --modes agent --ids D01 D02 --repeats 1   # a few tasks
    python scripts/run_evaluation.py --model gemini/<model id>                 # hosted model, needs JUDGE_LLM_API_KEY

Writes evaluation/results/<timestamp>/ with results.json (every run, with
traces), runs.csv and summary.md.
"""
import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.report import write_results  # noqa: E402
from src.evaluation.runner import run_evaluation  # noqa: E402
from src.evaluation.tasks import load_tasks  # noqa: E402
from src.response.agent import MODES  # noqa: E402
from src.response.llm import LLMConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO / "evaluation" / "tasks" / "dev_tasks.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=[str(DEFAULT_TASKS)], help="task JSON file(s)")
    parser.add_argument("--modes", nargs="+", default=["rules"], help=f"any of {list(MODES)}, or 'all'")
    parser.add_argument("--repeats", type=int, default=3, help="runs per task for the LLM modes (rules runs once)")
    parser.add_argument("--ids", nargs="+", help="only these task ids")
    parser.add_argument("--model", help="litellm model id (default: JUDGE_LLM_MODEL or ollama_chat/qwen2.5:3b)")
    parser.add_argument("--out", help="output folder (default: evaluation/results/<timestamp>)")
    parser.add_argument("--no-trace", action="store_true", help="don't store traces in results.json")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Stage failures are expected in the edge-case tasks and are recorded in the
    # results; don't print a stack trace for each one.
    logging.getLogger("src.pipeline").setLevel(logging.CRITICAL)

    modes = list(MODES) if args.modes == ["all"] else args.modes
    tasks = load_tasks(*args.tasks)
    if args.ids:
        tasks = [t for t in tasks if t.id in set(args.ids)]
    config = LLMConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)

    print(f"{len(tasks)} tasks, modes {modes}, {args.repeats} repeat(s) for LLM modes, model {config.model}")
    records = run_evaluation(tasks, modes, args.repeats, config=config, keep_trace=not args.no_trace,
                             progress=print)

    out = Path(args.out) if args.out else REPO / "evaluation" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = write_results(records, out, title=f"Evaluation results ({', '.join(modes)}; model {config.model})")
    print()
    print(paths["summary"].read_text(encoding="utf-8").split("## Success by difficulty")[0])
    print(f"Written: {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()