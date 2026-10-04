# Distill

Distill builds calibrated local Laya decision models from teacher-labelled synthetic data. It is released under Apache-2.0 (`LICENSE`, `NOTICE`).

## Layout

- `distill/`: package code; `schema.py` validates the flat Laya-compatible `decision.yaml` question mapping and `cli.py` defines the staged command surface.
- `examples/decision.yaml`: worked schema example.
- `tests/`: fast unit tests; they must not load models or call external services.
- `distill/teachers/`: teacher backends, JSON validation, usage auditing and the provider registry (`registry.py`). Local Ollama models (`local.py`, `LOCAL_MODELS`) are the default for every role: writer `ollama-gemma-dense` (`DEFAULT_WRITER`), labellers `ollama-qwen` + `ollama-gemma` (chosen in `docs/T10-LABELLER-BAKEOFF.md`). `anthropic-api` and `openai-api` (`backends.py`) are optional evaluation references, inert without their API key, and the registry refuses them as writer or labeller. Live Ollama tests need `DISTILL_LIVE_OLLAMA=1`. Ollama's MLX engine ignores `format`, so validation, not `format`, is what guarantees schema-valid answers.
- `distill/labelling.py` (dual-labeller merge and review triggers), `review_queue.py` (SQLite queue; human decisions override merges), `review_app.py` (FastAPI + vendored htmx page, 127.0.0.1 only) and `suspects.py` (optional Cleanlab hook for the relabel pass). `distill label` resumes from the queue, so re-run it after `distill review` to refresh the outputs.
- `.github/workflows/ci.yml`: the fast lint-and-test CI gate.
- `proof/`: the open-data end-to-end run in `proof/support-escalation-dense-5000/` plus the synthesis experiments behind its writer choice. `proof/run.sh <dir> <stage>` runs a stage and `proof/summarize.py` rebuilds `proof/RESULTS.md`, the source of the README's numbers. `train`, `eval` and `export` need a `distill` whose Python also has torch and laya; `proof/run.sh` explains the venv. Model weights, review queues, corpora and the question spec stay local (gitignored or unpublished). Training takes about 3 h at ~4800 states with `TRAIN_ARGS=--gradient-checkpointing`; when other processes crowd RAM, cap MPS with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5` and resume with `TRAIN_RESUME=1`. Run multi-hour stages under `nohup`. Local labelling of ~4800 items takes about 16 h; re-running `label` resumes it.
- `scripts/spike_train.py` and `docs/T0-MPS-SPIKE.md`: the MPS training spike and its go/no-go report, which gives the local training recipe for `distill train`. Its tests need torch and laya, which are not project dependencies, so CI skips them. Run them with an interpreter that has torch, transformers, laya and pytest.

## Development

Use Python 3.12 and uv:

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest
```

## Provenance

- Labels come from the two local Apache-2.0 models by default, so no hosted provider's terms attach to them; keep it that way. The registry (`distill/teachers/registry.py`) is the one place that decides which provider may play which role.
- A hosted model's output must not become training data. Use `distill eval --reference` for comparisons only, and check `distill-usage.jsonl` for each call's `provider`/`model`.
- Do not add credentials, API-key defaults or paid hosts to the repository.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
