# distill

Build calibrated local decision models from teacher-labelled synthetic data.

You describe a decision in plain English. Distill drafts a [Laya](https://github.com/NandhaKishorM/laya)
question spec, has local models write and label a synthetic corpus, fine-tunes a local Laya
model on your Mac, measures it against out-of-box Laya (and against a reference teacher, if
you name one), and exports a model folder with a confidence cutoff (tau). Below tau, the
exported routing snippet returns the local answer flagged low-confidence, or asks the fallback
teacher you configured.

```bash
distill init "Should this support ticket be escalated to a human?" -o decision.yaml
# review and edit decision.yaml
distill synth  -d decision.yaml                    # synthetic.jsonl + heldout.jsonl
distill label  -d decision.yaml --gold-from heldout.jsonl --heldout-output heldout-labelled.jsonl
distill review --queue review-queue.sqlite3        # human pass over flagged items and the gold set
distill label  ...                                 # re-run to apply the human decisions
distill train  -d decision.yaml --data labelled.jsonl -o model
distill eval   -d decision.yaml -m model --heldout heldout-labelled.jsonl --gold gold.jsonl
distill export -d decision.yaml -m model -e eval-report -o export
```

`train`, `eval` and `export` need torch and laya in the same Python as `distill`
([`proof/run.sh`](proof/run.sh) shows one way to build the environment). Development:
`uv sync --all-groups && uv run ruff check . && uv run pytest`.

## The headline result: a run with local models only

The end-to-end run in [`proof/support-escalation-dense-5000/`](proof/support-escalation-dense-5000/)
used no hosted teacher in synthesis or labelling. `gemma4:31b-nvfp4` (Gemma 4 31B) wrote
4,820 retained support-ticket states in 20 languages. Qwen3.5-35B-A3B and Gemma 4 26B-A4B
labelled them. Laya was fine-tuned on a Mac for about 3.2 hours. Every number below comes from
[`proof/RESULTS.md`](proof/RESULTS.md), which `python proof/summarize.py` regenerates from the
committed logs and metrics.

| Check | Gold accuracy | Held-out ECE | Comparable? |
| --- | ---: | ---: | --- |
| In-distribution: the model's own generated gold and held-out sets | 98.0% (408 answers) | 0.034 | **No** |
| Fixed items: a withheld set written by a hosted model | 94.62% (390 answers) | 0.039 | Yes |

**Read the 98% as "the pipeline learns what it was taught", nothing more.** Those gold and
held-out sets come from the same writer, the same recipe and the same pair of labellers as the
training data, so the number is in-distribution and is not comparable with any other model's
result.

**The fixed-item check is the honest one, and it does not pass.** It scores the model on a
withheld set of items that *a hosted model wrote* (the local labellers labelled them). A
hosted-model reference scored 99.74% on the same gold answers, so the model's 94.62% is 5.13
points behind, which is **0.13 points outside the 5-point allowance. The gate is NOT MET.**
Calibration is fine (held-out ECE 0.039 against a 0.08 bound), and at tau the model answers
54.2% of decisions locally at 97.1% accuracy. Out-of-box Laya scores 75.4% on the same gold
answers. This check cannot be rerun from this repository, because neither the withheld set nor
the reference answers are published.

**Why no dataset or question spec is published.** The original wording of the support-escalation
questions was drafted with a hosted model, and that wording is embedded in every row of the
corpus and was seen by the writer, both labellers and the trained model. Rather than
republish hosted-model text, the corpus, labels, spec and weights stay out of the repository.
Nothing is claimed beyond that: the *states* and *labels* in the run came from the open models
listed above (each call is recorded in `distill-usage.jsonl`), but a hosted model took part in
drafting the question spec and in writing the withheld evaluation set. A public set with a
human-written spec would be the clean way to reproduce and extend this run.

## Teachers: every role defaults to a local model

| Role | Flag | Default | Notes |
| --- | --- | --- | --- |
| Writer (`init`, `synth`) | `--writer` / `DISTILL_WRITER` | `ollama-gemma-dense` (`gemma4:31b-nvfp4`, Gemma 4 31B) | the dense local model that wrote the headline run |
| Labeller A (`label`) | `--labeller-a` / `DISTILL_LABELLER_A` | `ollama-qwen` (`qwen3.5:35b-mlx`, Qwen3.5-35B-A3B) | Apache-2.0 |
| Labeller B (`label`) | `--labeller-b` / `DISTILL_LABELLER_B` | `ollama-gemma` (`gemma4:26b`, Gemma 4 26B-A4B) | Apache-2.0; a different family from A, so their disagreements flag items for review |
| Reference (`eval`) | `--reference` / `DISTILL_REFERENCE` | none | optional comparison; see below |
| Fallback (`export`) | `--fallback` | none | optional; see below |

Labeller B won a bake-off against Muse Glimmer 30B and Nemotron 3.5 Lightning on 150 gold
items: it tied for the best agreement (96.7%), had the best calibration (ECE 0.008) and was
4.9 times faster than the runner-up. See
[`docs/T10-LABELLER-BAKEOFF.md`](docs/T10-LABELLER-BAKEOFF.md) (its gold slice is not
published; the script takes any gold file).

The registry in [`distill/teachers/registry.py`](distill/teachers/registry.py) is a plain
provider table with no quota reading, switching or pausing. `--writer` affects only
`init` and `synth`; it never overrides the labeller pair.

**Setup:** install [Ollama](https://ollama.com), then
`ollama pull gemma4:31b-nvfp4 && ollama pull qwen3.5:35b-mlx && ollama pull gemma4:26b`
(about 19, 21 and 18 GB). Point `DISTILL_OLLAMA_URL` at another host if Ollama doesn't listen
on `127.0.0.1:11434`. Models on one Ollama host take turns, because one 64 GB Mac holds one of
them at a time.

**Optional hosted references.** `anthropic-api` and `openai-api` are off by default and inert
unless `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is already set. They may answer
`distill eval --reference <provider>` (the readiness gate then compares gold accuracy with the
reference's), or be named as an export `--fallback`. They are **refused** for `init`, `synth`
and `label`, with a one-line reason, so their output cannot become training data. Choosing a
hosted model brings its provider's terms onto your data: Anthropic's Usage Policy prohibits
training on outputs without prior authorization, and OpenAI's terms restrict using output to
develop competing models. Use an API key, not a consumer-plan sign-in. This is not legal
advice.

**With no reference**, `distill eval` reports "no reference measured": the gate checks ECE
(<= 0.08) and says that gold accuracy was not compared, instead of failing. With a reference,
the gate also needs gold accuracy within 5 points of it.

**With no fallback** (the default), the exported `route_with_fallback.py` returns the local
answer with `"low_confidence": true` when any answer is below tau, so you decide what happens
next. With `--fallback <provider>` it asks that teacher instead.

## Synthesis experiments behind the writer choice

The writer is a dense model and its language is assigned per call, for reasons measured in
`proof/` (see [`proof/RESULTS.md`](proof/RESULTS.md)):

- A mixture-of-experts writer (`qwen3.5:35b-mlx`) failed a 1,500-state breadth gate before any
  labelling: 7.5% near-duplicates, and a held-out set with four language labels, 78% English
  ([`proof/support-escalation-local-synthesis-1500/`](proof/support-escalation-local-synthesis-1500/)).
- The dense Gemma writer, free to pick languages, still collapsed to 84.5% English on its
  held-out set.
- Assigning the language before each call fixed that (20 languages), and a 39%-English
  schedule gave the headline run its shape.

## How the pipeline works

- **`init`** drafts a decision spec you review and edit; **`synth`** writes diverse, MinHash-
  deduplicated states for five styles plus an independently prompted held-out set, resuming
  from per-chunk checkpoints.
- **`label`** soft-labels every state twice. Disagreements and other review triggers
  (`distill/labelling.py`) go to a SQLite queue; `distill review` opens a page on 127.0.0.1
  where a human decides them. Human decisions override merges, and `label` resumes from the
  queue. With no human, `--teacher-gold` writes gold items both labellers agreed on, which
  are teacher-agreed, not human-verified.
- **`train`** fine-tunes Laya on MPS (recipe measured in
  [`docs/T0-MPS-SPIKE.md`](docs/T0-MPS-SPIKE.md)), fits per-question-type temperatures on a
  held-out calibration slice and checks the saved checkpoint reloads through `laya.load()`.
- **`eval`** measures accuracy, ECE, Brier, selective accuracy and adversarial slices for the
  model, out-of-box Laya and the optional reference, and selects tau, the lowest confidence
  at which accepted answers still reach the target accuracy (97% by default).
- **`export`** writes a self-contained model folder: weights, model card, `export.json` with
  tau, a `laya-serve` launcher, and the routing snippet.

## Reproduce

`python proof/summarize.py` rebuilds `proof/RESULTS.md` from the committed logs and metrics.
That is all this repository can reproduce on its own, because the corpus is not published.

To rerun the whole pipeline on a decision of your own, local-only, write your own
`decision.yaml` (start from [`examples/decision.yaml`](examples/decision.yaml)) and run
`proof/run.sh <decision-dir> <stage>` for `synth`, `label`, `train`, `eval` and `export`. The
headline run took about 7.6 h to synthesise, about 15 h to label and 3.2 h to train on an
Apple-silicon Mac with 64 GB, roughly 26-27 machine-hours. LLM sampling is not seeded, and MPS
kernels aren't bit-deterministic, so a rerun gives a new corpus of the same shape rather than
identical rows. `LABEL_MAX_ITEMS` labels in resumable slices and `TRAIN_RESUME=1` continues an
interrupted training run.

## Caveats

- Gold sets here are teacher-agreed, not human-verified, and gold labels came from the same
  models as the training labels. Treat accuracy as agreement with those models.
- The fixed-item check uses one withheld set, 390 gold answers; a 1-point difference is about
  4 answers.
- Only one decision (support escalation) went through the full open-data pipeline.
- The reference result is a quoted number, not something this repository can recompute.

## Credits and licences

Distill is released under the Apache License 2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)). It fine-tunes [Laya](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0, Convai Innovations), which is built on
[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (Apache-2.0). Distill
downloads these weights from Hugging Face and does not redistribute them. Its training loop
adapts the one in Laya's fine-tuning notebook. By default its teacher models run locally
through [Ollama](https://ollama.com) and are not bundled:
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (Apache-2.0, Alibaba Cloud),
and [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) and
[Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) (Apache-2.0, Google DeepMind).
Apache-2.0 places no conditions on model outputs. If you configure a hosted model instead, its
provider's terms govern your use of its outputs. The review page vendors
[htmx](https://github.com/bigskysoftware/htmx) (0BSD). Model and product names belong to their
owners, and no endorsement is implied. This is not legal advice.
