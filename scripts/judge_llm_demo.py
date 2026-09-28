"""
Run the Judge on a few hand-built cases against a real model and print what
it decided, how, and at what cost. Useful for checking the LLM setup and as
evidence of the agent loop working (the traces are the "execution traces"
deliverable in miniature).

Needs Ollama running with the model pulled, for the default model:
    ollama pull qwen2.5:3b

Usage (from the repo root):
    python scripts/judge_llm_demo.py                     # all modes, all cases
    python scripts/judge_llm_demo.py --mode agent --trace
    python scripts/judge_llm_demo.py --mode agent --case category_mismatch --trace   # one case only
    python scripts/judge_llm_demo.py --model gemini/gemini-2.0-flash   # needs JUDGE_LLM_API_KEY
    python scripts/judge_llm_demo.py --save judge_demo.json
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.response.agent import MODES, JudgeAgent  # noqa: E402
from src.response.llm import LLMConfig  # noqa: E402
from src.shared.schemas import (  # noqa: E402
    CorrelationResult,
    DetectionResult,
    JudgeInput,
    MitigationRecommendation,
    TrafficEvent,
)


def _detection(anomalous, confidence, notes):
    event = TrafficEvent(features={"Destination Port": 22.0, "Flow Duration": 1200.0})
    return DetectionResult(event=event, is_anomalous=anomalous, confidence=confidence, detector_notes=notes)


def _mitigation(detection, techniques, confidence, action):
    correlation = CorrelationResult(detection=detection, matched_technique_ids=list(techniques), confidence=confidence)
    return MitigationRecommendation(correlation=correlation, proposed_action=action, confidence=confidence)


def build_cases():
    cases = {}

    d = _detection(True, 0.93, "[category=BruteForce] Many short SSH connections from one source.")
    cases["agreed_brute_force"] = JudgeInput(d, _mitigation(d, ["T1110"], 0.82, "block_source_ip"))

    d = _detection(True, 0.91, "[category=DoS] High packet rate to one web server.")
    cases["category_mismatch"] = JudgeInput(d, _mitigation(d, ["T1110"], 0.78, "block_source_ip"))

    d = _detection(True, 0.88, "[category=WebAttack] Unusual request payload sizes.")
    cases["low_mitigation_confidence"] = JudgeInput(d, _mitigation(d, ["T1190"], 0.41, "isolate_service"))

    d = _detection(True, 0.90, "[category=PortScan] One source touching many ports.")
    cases["mitigation_failed"] = JudgeInput(d, None, "RuntimeError: vector DB unreachable")

    d = _detection(False, 0.97, None)
    cases["benign"] = JudgeInput(d, _mitigation(d, [], 0.2, "none"))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    parser.add_argument("--case", choices=list(build_cases()) + ["all"], default="all",
                        help="run one example case only (default: all)")
    parser.add_argument("--model", help="litellm model id (default: JUDGE_LLM_MODEL or ollama_chat/qwen2.5:3b)")
    parser.add_argument("--trace", action="store_true", help="print every trace step")
    parser.add_argument("--save", help="write all results as JSON to this path")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = LLMConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    modes = list(MODES) if args.mode == "all" else [args.mode]

    print(f"Model: {config.model}  (timeout {config.timeout_seconds:.0f}s, max steps {config.max_steps})\n")
    saved = []
    cases = build_cases()
    if args.case != "all":
        cases = {args.case: cases[args.case]}
    for name, judge_input in cases.items():
        for mode in modes:
            result = JudgeAgent(mode=mode, config=config).run(judge_input)
            usage = result.llm_usage
            print(f"[{name}] mode={mode:<11} -> {result.recommended_action:<18} "
                  f"escalated={result.escalated_to_human!s:<5} decided_by={result.decided_by:<18} "
                  f"iterations={usage.get('iterations', 0)} calls={usage.get('llm_calls', 0)} "
                  f"tokens={usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)} "
                  f"llm_ms={usage.get('llm_latency_ms', 0):.0f}")
            print(f"    reasoning: {result.reasoning}")
            if usage.get("fallback_reason"):
                print(f"    fallback:  {usage['fallback_reason']}")
            if args.trace:
                for s in result.trace:
                    print(f"      {s.step_number}. {s.action}: {s.thought or ''} | {s.observation or ''}")
            saved.append({"case": name, "mode": mode, "recommended_action": result.recommended_action,
                          "escalated": result.escalated_to_human, "decided_by": result.decided_by,
                          "rule_case": result.case, "reasoning": result.reasoning, "llm_usage": usage,
                          "trace": [asdict(s) for s in result.trace]})
        print()

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, default=str)
        print(f"Saved {len(saved)} results to {os.path.abspath(args.save)}")


if __name__ == "__main__":
    main()
