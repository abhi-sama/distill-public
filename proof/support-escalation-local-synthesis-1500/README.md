# Local-synthesis support-escalation probe (negative result)

This is a 1,500-state breadth gate for writing the synthetic states with a local
mixture-of-experts model. It used 300 examples for each of the five standard synthesis styles
and a separately prompted 150-state held-out set.

Synthesis was pinned to `ollama-qwen` (`qwen3.5:35b-mlx`). The committed
`distill-usage.jsonl` records only that provider and model. No label, train, eval or export
stage was run, because the probe failed the breadth gate: MinHash removed 112 of 1,500
training states (7.47%), the retained training set was dominated by English, and the held-out
set covered only four language labels (78.2% English).

That result is why the writer for the headline run is the dense `ollama-gemma-dense` model
with assigned languages (`../support-escalation-dense-5000/`). The generated states and the
decision spec are not published; `synthesis-stats.json` has the statistics.
