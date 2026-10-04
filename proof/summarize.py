"""Collect the open-data proof run's measured numbers into proof/RESULTS.md.

Run from the repository root:

    python proof/summarize.py

It reads only files committed beside it, so every number in RESULTS.md and the README can be
regenerated and checked. It exits non-zero, naming the file, if an input is missing, rather
than writing empty tables.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROOF = Path(__file__).resolve().parent
RUN = PROOF / "support-escalation-dense-5000"

# The fixed-item check scores the dense model on 390 gold answers from a withheld set that a
# hosted model wrote. A hosted-model reference answered 389 of them correctly when that set was
# built. Neither the set nor the reference's answers are published, so this number is quoted,
# not recomputed: it is the one input of RESULTS.md that no committed file can reproduce.
WITHHELD_REFERENCE_CORRECT = 389
WITHHELD_REFERENCE_TOTAL = 390
GATE_ECE = 0.08
GATE_ALLOWANCE = 0.05

AUDIT_RUNS = (
    (
        "support-escalation-local-synthesis-1500",
        "MoE writer (`ollama-qwen`) probe. Stopped at the synthesis-breadth gate: the held-out "
        "set had 4 language labels, 78% English, and the training set lost 7.5% to near-duplicates.",
    ),
    (
        "support-escalation-dense-synthesis-1500",
        "Dense writer (`ollama-gemma-dense`), model free to choose each language: 46 labels in "
        "training, but the held-out set was 84.5% English, so free choice collapses toward English.",
    ),
    (
        "support-escalation-dense-assigned-languages-1500",
        "Dense writer with the language assigned before each call: all 20 requested languages "
        "appear, but a uniform schedule leaves English at 4.7%, which is not benchmark-shaped.",
    ),
)


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"summarize.py: missing input {path.relative_to(PROOF.parent)}")
    return json.loads(path.read_text(encoding="utf-8"))


def last_json_line(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"summarize.py: missing input {path.relative_to(PROOF.parent)}")
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    sys.exit(f"summarize.py: {path.relative_to(PROOF.parent)} has no JSON summary line")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def dec(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def teacher_calls(path: Path) -> Counter:
    if not path.exists():
        sys.exit(f"summarize.py: missing input {path.relative_to(PROOF.parent)}")
    calls: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["outcome"] in {"success", "malformed", "error"}:
            calls[(record["provider"], record["outcome"])] += 1
    return calls


def english_share(stats: dict) -> float:
    languages = stats["per_language"]
    return languages.get("English", 0) / sum(languages.values())


def calls_text(calls: Counter) -> str:
    return "; ".join(
        f"{provider} {outcome}: {count}" for (provider, outcome), count in sorted(calls.items())
    )


def pipeline_section() -> list[str]:
    synth = load(RUN / "synthesis-stats.json")
    train = load(RUN / "model-metrics.json")
    labelled = last_json_line(RUN / "label-retry.log")
    training, heldout = synth["training"], synth["heldout"]
    calls = teacher_calls(RUN / "distill-usage.jsonl")
    providers = sorted({provider for provider, _ in calls})
    if not all(provider.startswith("ollama-") for provider in providers):
        sys.exit(f"summarize.py: non-local provider in {RUN.name}/distill-usage.jsonl: {providers}")
    return [
        "## The pipeline, run entirely with local models",
        "",
        "| Stage | Result |",
        "| --- | --- |",
        f"| Writer | `ollama-gemma-dense` (Gemma 4 31B); {training['requested']:,} requested "
        f"training states, {heldout['requested']} held-out states |",
        f"| Synthesis | {training['retained']:,} training states kept after MinHash dedupe "
        f"({pct(training['near_duplicate_rate'])} removed); {heldout['retained']} held-out; "
        f"{len(training['per_language'])} languages, {pct(english_share(training))} English |",
        f"| Labellers | `ollama-qwen` (Qwen3.5-35B-A3B) and `ollama-gemma` (Gemma 4 26B-A4B); "
        f"{labelled['labelled']:,} of {labelled['total']:,} items labelled, "
        f"{labelled['flagged']} flagged ({pct(labelled['flagged_rate'])}) |",
        f"| Training | {train['data']['rows']:,} rows, {train['options']['epochs']} epochs on MPS, "
        f"{train['train_seconds'] / 60:.1f} min; internal test split accuracy "
        f"{pct(train['test']['all']['acc'])} |",
        f"| Teacher calls | {calls_text(calls)} |",
        "",
        "Every call in `distill-usage.jsonl` went to a local Ollama model; no hosted provider "
        "and no API key was used in synthesis or labelling. Gold labels are teacher-agreed, not "
        "human-verified.",
    ]


def in_distribution_section() -> list[str]:
    report = load(RUN / "eval-report" / "metrics.json")
    local, tau = report["local"], report["tau"]
    return [
        "",
        "## In-distribution result (not comparable)",
        "",
        "The model's own gold and held-out sets were synthesized by the same recipe, and "
        "labelled by the same pair of models, as its training data. This number says the "
        "pipeline learns what it was taught. It is **not** comparable with any other model's "
        "result and must not be quoted as a quality pass.",
        "",
        "| Set | n (answers) | Accuracy | ECE (15) | Brier |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| held-out | {local['heldout']['n']} | {pct(local['heldout']['accuracy'])} | "
        f"{dec(local['heldout']['ece_15'])} | {dec(local['heldout']['brier'])} |",
        f"| gold (teacher-agreed) | {local['gold']['n']} | {pct(local['gold']['accuracy'])} | "
        f"{dec(local['gold']['ece_15'])} | {dec(local['gold']['brier'])} |",
        "",
        f"Out-of-box Laya scores {pct(report['base_laya']['gold']['accuracy'])} on the same gold "
        f"answers. At tau = {dec(tau['threshold'])} the model answers "
        f"{pct(tau['coverage'])} of decisions locally at {pct(tau['accuracy'])} accuracy.",
    ]


def fixed_item_section() -> list[str]:
    report = load(RUN / "eval-on-withheld-items" / "metrics.json")
    local, tau = report["local"], report["tau"]
    reference = WITHHELD_REFERENCE_CORRECT / WITHHELD_REFERENCE_TOTAL
    gap = reference - local["gold"]["accuracy"]
    beyond = gap - GATE_ALLOWANCE
    met = beyond <= 0 and local["heldout"]["ece_15"] <= GATE_ECE
    verdict = "MET" if met else "NOT MET"
    return [
        "",
        "## Fixed-item check on a withheld set",
        "",
        "The honest comparison scores the model on items it did not generate. These "
        f"{local['gold']['n']} gold and {local['heldout']['n']} held-out answers come from a "
        "withheld set that **a hosted model wrote** and the local labellers labelled. The set "
        "is not published (nor is the spec it was written against), so this check cannot be "
        "rerun from this repository. It ran with Hugging Face and Transformers forced offline "
        "and without any teacher call. The reference accuracy below is a hosted-model "
        f"reference's score on the same gold answers ({WITHHELD_REFERENCE_CORRECT} of "
        f"{WITHHELD_REFERENCE_TOTAL}), measured when the set was built and quoted here.",
        "",
        "| Held-out acc | Held-out ECE | Gold acc | Gold ECE | tau | Local share at tau | "
        "Reference gold acc | Gap to reference | Beyond 5-pt allowance | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| {pct(local['heldout']['accuracy'])} | {dec(local['heldout']['ece_15'])} | "
        f"{pct(local['gold']['accuracy'])} | {dec(local['gold']['ece_15'])} | "
        f"{dec(tau['threshold'])} | {pct(tau['coverage'])} | {pct(reference)} | "
        f"{gap * 100:.2f} pts | {beyond * 100:+.2f} pts | {verdict} |",
        "",
        f"The model is calibrated on the fixed held-out set (ECE {dec(local['heldout']['ece_15'])}"
        f" against the {GATE_ECE} bound), but its {gap * 100:.2f}-point gold gap is "
        f"{abs(beyond) * 100:.2f} points beyond the five-point allowance. The comparable gate is "
        f"**{verdict}**.",
    ]


def audit_section() -> list[str]:
    lines = [
        "",
        "## Synthesis experiments behind the writer choice",
        "",
        "Each folder holds only the logs, `synthesis-stats.json` and `distill-usage.jsonl`; the "
        "generated states are not published. All calls went to local Ollama models.",
        "",
        "| Folder | Training kept / requested | Held-out languages (English share) | Finding |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, finding in AUDIT_RUNS:
        stats = load(PROOF / name / "synthesis-stats.json")
        calls = teacher_calls(PROOF / name / "distill-usage.jsonl")
        if not all(provider.startswith("ollama-") for provider, _ in calls):
            sys.exit(f"summarize.py: non-local provider in {name}/distill-usage.jsonl")
        heldout = stats["heldout"]
        lines.append(
            f"| `proof/{name}/` | {stats['training']['retained']:,} / "
            f"{stats['training']['requested']:,} | {len(heldout['per_language'])} "
            f"({pct(english_share(heldout))}) | {finding} |"
        )
    lines += [
        "",
        "Two further dense-model fallbacks (Muse Glimmer 30B and Qwen3.6 27B) timed out on local "
        "requests, and two earlier 5,000-state launch attempts ended before synthesis finished; "
        "none left publishable results.",
    ]
    return lines


def main() -> None:
    lines = [
        "# Open-data proof run: measured results",
        "",
        "Generated by `python proof/summarize.py` from the files in "
        "`proof/support-escalation-dense-5000/` and the synthesis experiment folders.",
        "",
        *pipeline_section(),
        *in_distribution_section(),
        *fixed_item_section(),
        *audit_section(),
        "",
    ]
    (PROOF / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {PROOF / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
