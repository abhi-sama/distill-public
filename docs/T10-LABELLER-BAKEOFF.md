# T10: the labeller-B bake-off

Distill labels every state twice, with two independent models, and both run locally so the
labels carry no hosted provider's terms. Labeller A is Qwen3.5-35B-A3B. Labeller B had to be a
second open-weight family, picked from three candidates by this bake-off. **Winner: Gemma 4 26B-A4B (`gemma4:26b`, Apache-2.0), now `DEFAULT_LABELLER_B`
(`ollama-gemma`).**

## Method

- **Slice:** the first 50 rows of the teacher-agreed gold set of each of three 1,500-state
  decisions. That is 150 items and 750 question decisions: content moderation 250, PR security
  review 200, support escalation 300. **The slice is not published**: those states and their
  labels were written by a hosted model, so this repository does not include them and the run
  cannot be repeated as-is. `scripts/labeller_bakeoff.py` takes any gold JSONL, so you can run
  the same comparison on your own gold set.
- **Reference:** those rows' `gold` labels. A hosted model labelled them twice and the two
  answers agreed on these items (`--teacher-gold`), so "agreement" means agreement with that
  hosted model, not accuracy against human truth.
- **Path:** the production one. `OllamaTeacher` sends Distill's own labeller prompt and strict
  schema in batches of 10 (as in `proof/run.sh`), retries up to 3 times, and `parse_batch`
  normalises the answers. Sampling uses each tag's Modelfile defaults with thinking off. Qwen
  uses its recommended non-thinking settings (`LOCAL_MODELS` in `distill/teachers/local.py`).
- **Machine:** Apple M5 Pro, 64 GB, Ollama 0.32.14. Models ran one at a time.
- **Run it on your own gold set:** `python scripts/labeller_bakeoff.py docs/t10-bakeoff
  ollama-gemma` (any providers from `LOCAL_MODELS`; `MAX_BATCHES` bounds one invocation, and a
  re-run resumes). The per-batch distributions, the per-attempt usage logs and `summary.json`
  of the run above are in [`t10-bakeoff/`](t10-bakeoff/).

## Results

| Teacher | Ollama tag (engine) | Agreement with gold | Brier | ECE (15 bins) | JSON validity (attempts) | Wall time, 15 batches | Per batch | Items flagged with Qwen as A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** (winner) | `gemma4:26b` (GGUF Q4_K_M) | **96.7%** (725/750) | 0.054 | **0.008** | 15/15 (100%) | **536 s** | **35.7 s** | **17.3%** |
| Meta Muse Glimmer 30B | `muse-glimmer:30b` (GGUF Q4_K_M) | 96.7% (725/750) | 0.053 | 0.033 | 15/15 (100%) | 2,619 s | 174.6 s | 19.3% |
| Nemotron 3.5 Lightning 30B-A3B | `nemotron-3.5-lightning:30b` (GGUF Q4_K_M) | 91.6% (687/750) | 0.135 | 0.024 | 15/15 (100%) | 766 s | 51.1 s | 32.0% |
| *Labeller A: Qwen3.5-35B-A3B* | `qwen3.5:35b-mlx` (MLX nvfp4) | 95.6% (717/750) | 0.073 | 0.018 | 15/15 (100%) | 822 s | 54.8 s | n/a |

"JSON validity" counts every answer the model returned: it has to parse and pass the strict
schema. Every candidate passed on the first attempt, so no retries were needed. The last
column is the share of the 150 items that the two-labeller review triggers
(`distill/labelling.py`) would send to a human with that candidate as labeller B.

Per decision:

| Teacher | Content moderation | PR security review | Support escalation |
| --- | ---: | ---: | ---: |
| Gemma 4 26B-A4B | 99.6% | 90.5% | 98.3% |
| Muse Glimmer 30B | 98.8% | 94.0% | 96.7% |
| Nemotron 3.5 Lightning | 96.0% | 83.0% | 93.7% |
| Qwen3.5-35B-A3B (A) | 96.4% | 92.0% | 97.3% |

Every model's weakest question is PR review's 4-level `risk_level`: Gemma 76%, Glimmer 80%,
Qwen 80%, Nemotron 66%.

## Why Gemma

- **It tied Glimmer for the best agreement and beat it on everything else.** Its ECE was 0.008
  against 0.033, it flagged fewer items when paired with Qwen, and it ran 4.9 times faster.
  Glimmer is a dense 28B model; Gemma 26B-A4B is a mixture of experts that activates about
  4B parameters per token. At 1,500 states, B's pass takes about 1.5 h with Gemma and about
  7.3 h with Glimmer.
- **Glimmer is the runner-up.** It wins PR review (94.0% against 90.5%), where Gemma is
  weakest. On 200 decisions, that 3.5-point gap is 7 decisions.
- **Nemotron came last** on agreement, Brier and flag rate.

## Findings that shaped the backend

1. **`format` works on the GGUF engine, but not on MLX.** A prompt asking for a Markdown
   list, sent with a JSON schema in `format`, came back as schema-valid JSON from all three
   GGUF tags. The MLX `qwen3.5:35b-mlx` answered in Markdown. `OllamaTeacher` therefore sends the schema in `format` and in the prompt, and
   validates every answer itself.
2. **`muse-glimmer:30b` leaks its `<|eot|>` end marker after the JSON.** `strip_wrappers`
   removes trailing chat-template markers and Markdown fences, and nothing more.
3. **Ollama keeps one of these models in memory at a time on this Mac.** Loading Nemotron
   evicted Qwen. Two labellers on one Ollama host therefore take turns (`label_pending` runs
   them one after the other) instead of evicting each other on every batch.

## Licences

| Model | Licence on the model card | Licence text embedded in the Ollama tag |
| --- | --- | --- |
| Qwen3.5-35B-A3B | Apache-2.0 | Apache-2.0 |
| Gemma 4 26B-A4B | Apache-2.0 | Apache-2.0 |
| Nemotron 3.5 Lightning | OpenMDW-1.1 (Hugging Face) | NVIDIA Open Model License (Oct 24, 2025). It says "An output is not a Derivative Model" and "NVIDIA claims no ownership rights in outputs", but it also terminates rights if you bypass a guardrail |
| Muse Glimmer 30B | Apache-2.0 (Hugging Face) | none embedded |

The chosen pair, Qwen and Gemma 4, is Apache-2.0 on both the card and the tag.

## Limits

- **150 items, one run each.** A 1-point difference is 7-8 decisions and within noise.
  Glimmer's lead on PR review is 7 of 200 decisions.
- **The reference is a hosted model's agreement with itself on the easier gold items.** A
  human-verified gold set would be harder.
- **Wall time includes loading each model** into memory on its first batch.
- **Gemma 4 31B (dense) was not tried.** Glimmer, the dense 28B candidate, was already
  4.9 times slower than Gemma 26B-A4B for the same agreement.
