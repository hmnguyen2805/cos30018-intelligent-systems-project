# COS30018 Intelligent Systems: multi-agent cybersecurity triage

A manager-subagent system that triages network traffic events (CICIDS2017):

```text
TrafficEvent
  -> Detection Manager  (+ Detection Subagent)    is it an attack, and which kind?
  -> Mitigation Manager (+ Correlation Subagent)  which ATT&CK technique, and what response?
  -> Judge (LLM agent)                            act automatically, or escalate to a human?
  => PipelineRun                                  every result, trace, error and timing
```

The Judge can send a case back to either manager once to "look again" (a recheck).
See [docs/architecture.md](docs/architecture.md) for how the parts fit together.

| Part | Owner | Code | Details |
|---|---|---|---|
| Detection Manager + Subagent | Vinh Nghiem | `src/detection/` | [src/detection/README.md](src/detection/README.md) |
| Mitigation Manager + Correlation Subagent | Callum Fennessy | `src/correlation/` | [src/correlation/README.md](src/correlation/README.md) |
| Judge, pipeline, UI | Minh Nguyen | `src/response/`, `src/pipeline.py`, `src/ui/` | [src/response/README.md](src/response/README.md) |

## Install

Python 3.10 or later.

```sh
git clone https://github.com/hmnguyen2805/cos30018-intelligent-systems-project.git
cd cos30018-intelligent-systems-project
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt                              # core: pipeline, Judge, UI, tests
pip install -r src/detection/requirements-detection.txt      # Detection: training + optional LLM layer
pip install -r src/correlation/requirements-correlation.txt  # optional: embedding search for "Unknown" categories (pulls in torch)
```

LLM (for the Judge's `single_shot` and `agent` modes, and Detection's optional LLM layer):
install [Ollama](https://ollama.com/download), then

```sh
ollama pull qwen2.5:3b
```

To use a hosted model instead, copy `.env.example` to `.env` and set the model id and API key.
Never commit `.env`.

## Train the detection models

Needed for real traffic (not for the tests or the UI's demo scenarios). Downloads
CICIDS2017 from Kaggle via `kagglehub` and writes the models to `models/` (gitignored):

```sh
python -m src.detection.train
```

## Run

```sh
python -m src.ui.app                                       # UI at http://127.0.0.1:7860
python scripts/judge_llm_demo.py --mode agent --trace      # the Judge on example cases, real LLM
```

In the UI, "Demo scenario" runs without trained models (Detection is scripted;
Mitigation and the Judge are real). "My own event" runs the full real pipeline.
Pick the Judge mode: `rules` (no LLM), `single_shot` (one LLM call) or `agent`
(the LLM agent loop, which can recheck). If the LLM isn't reachable, the Judge
falls back to its rule table and the UI says so.

## Test

No trained model, LLM or network needed (LLMs and encoders are replaced by fakes):

```sh
python -m pytest tests/ -q
```

## Configuration

Environment variables (or a `.env` file, see `.env.example`):

| Variable | Default | Used by |
|---|---|---|
| `JUDGE_MODE` | `rules` | `build_default_pipeline()`: `rules`, `single_shot` or `agent` |
| `JUDGE_LLM_MODEL` | `ollama_chat/qwen2.5:3b` | Judge (any litellm model id) |
| `JUDGE_LLM_API_KEY` | none | Judge, hosted models only |
| `JUDGE_LLM_TIMEOUT` / `_MAX_STEPS` / `_MAX_TOKENS` | `60` / `6` / `400` | Judge |
| `DETECTION_LLM_*` | see `.env.example` | Detection's optional LLM layer |

## Repository layout

```text
src/shared/       data contracts (schemas.py), BaseAgent, tag parsing
src/detection/    Detection Manager + Subagent, training, evaluation
src/correlation/  Mitigation Manager + Correlation Subagent, technique catalog
src/response/     Judge Agent: modes, tools, playbook, guardrails, rules
src/pipeline.py   runs the agents in order, rechecks, failure handling, timings
src/ui/           Gradio UI and demo scenarios
scripts/          demo scripts
tests/            automated tests (pytest)
docs/             architecture and run logs
```
