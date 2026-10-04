"""Command-line interface for the staged Distill workflow."""

import json
import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .evaluation import EvaluationError, run_evaluation
from .exporting import ExportError, run_export
from .schema import load_decision
from .teachers import DEFAULT_LABELLER_A, DEFAULT_LABELLER_B, DEFAULT_WRITER, RoleRefused
from .training import TrainingError, TrainingOptions, run_training

app = typer.Typer(
    name="distill",
    help="Build a calibrated local decision model from teacher-labelled synthetic data.",
    no_args_is_help=True,
)


class TeacherProvider(StrEnum):
    """Every teacher ``make_registry`` registers. The ollama-* models run locally.

    The two API providers parse here but are refused for every role except the evaluation
    reference (`distill eval --reference`).
    """

    ollama_qwen = "ollama-qwen"
    ollama_gemma = "ollama-gemma"
    ollama_gemma_dense = "ollama-gemma-dense"
    ollama_glimmer_dense = "ollama-glimmer-dense"
    ollama_qwen_dense = "ollama-qwen-dense"
    ollama_nemotron = "ollama-nemotron"
    ollama_glimmer = "ollama-glimmer"
    anthropic_api = "anthropic-api"
    openai_api = "openai-api"


# init and synth turn a writer's output into specs and states. The default is the dense local
# model that wrote the open-data run; it never falls back to another provider.
WriterOption = Annotated[
    TeacherProvider,
    typer.Option(
        "--writer",
        envvar="DISTILL_WRITER",
        help="Teacher that drafts the spec and writes the synthetic states. Runs locally "
        "through Ollama by default.",
    ),
]


def _writer(provider: TeacherProvider):
    """Resolve the writer teacher, or exit with the registry's one-line refusal."""
    from .synthesis import make_registry

    try:
        return make_registry().writer(provider.value)
    except RoleRefused as error:
        typer.echo(f"distill: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command()
def init(
    decision: Annotated[str, typer.Argument(help="Decision to model, in plain English.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write decision.yaml.")
    ] = Path("decision.yaml"),
    writer: WriterOption = TeacherProvider(DEFAULT_WRITER),
) -> None:
    """Draft a validated decision specification for human review."""
    from .synthesis import draft_decision, write_decision

    spec = draft_decision(decision, _writer(writer))
    write_decision(spec, output)
    typer.echo(f"Wrote validated decision draft to {output}. Review and edit it before synthesis.")


@app.command()
def synth(
    decision_file: Annotated[
        Path, typer.Option("--decision-file", "-d", help="Path to decision.yaml.")
    ] = Path("decision.yaml"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Training JSONL output path.")
    ] = Path("synthetic.jsonl"),
    heldout_output: Annotated[
        Path, typer.Option("--heldout-output", help="Held-out JSONL output path.")
    ] = Path("heldout.jsonl"),
    stats_output: Annotated[
        Path, typer.Option("--stats-output", help="Diversity-statistics JSON output path.")
    ] = Path("synthesis-stats.json"),
    samples_per_style: Annotated[
        int, typer.Option("--samples-per-style", min=1, help="Examples to request for each style.")
    ] = 100,
    heldout_count: Annotated[
        int, typer.Option("--heldout-count", min=1, help="Examples to request for held-out data.")
    ] = 100,
    writer: WriterOption = TeacherProvider(DEFAULT_WRITER),
) -> None:
    """Generate deduplicated training and independently prompted held-out states."""
    from .synthesis import (
        corpus_rows,
        diversity_stats,
        synthesize,
        write_jsonl,
        write_stats,
    )

    spec = load_decision(decision_file)
    training, heldout = synthesize(
        spec,
        _writer(writer),
        samples_per_style=samples_per_style,
        heldout_count=heldout_count,
        training_progress_path=output.with_name(f"{output.stem}.progress.jsonl"),
        heldout_progress_path=heldout_output.with_name(f"{heldout_output.stem}.progress.jsonl"),
    )
    write_jsonl(corpus_rows(spec, training.examples), output)
    write_jsonl(corpus_rows(spec, heldout.examples), heldout_output)
    write_stats(training, heldout, stats_output)
    typer.echo(
        json.dumps({"training": diversity_stats(training), "heldout": diversity_stats(heldout)})
    )


@app.command()
def label(
    decision_file: Annotated[
        Path, typer.Option("--decision-file", "-d", help="Path to decision.yaml.")
    ] = Path("decision.yaml"),
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Synthetic JSONL written by distill synth.")
    ] = Path("synthetic.jsonl"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Labelled training JSONL output path.")
    ] = Path("labelled.jsonl"),
    queue_path: Annotated[
        Path, typer.Option("--queue", help="Review-queue database; labelling resumes from it.")
    ] = Path("review-queue.sqlite3"),
    gold_from: Annotated[
        Path | None,
        typer.Option("--gold-from", help="JSONL to sample the human gold set from (heldout)."),
    ] = None,
    gold_size: Annotated[
        int, typer.Option("--gold-size", min=1, help="Number of gold-set items to sample.")
    ] = 150,
    gold_output: Annotated[
        Path, typer.Option("--gold-output", help="Human-labelled gold JSONL output path.")
    ] = Path("gold.jsonl"),
    heldout_output: Annotated[
        Path | None,
        typer.Option(
            "--heldout-output",
            help="Also teacher-label the --gold-from rows left out of the gold set and write "
            "them here, as the labelled held-out set distill eval needs.",
        ),
    ] = None,
    teacher_gold: Annotated[
        bool,
        typer.Option(
            "--teacher-gold",
            help="No human reviewer: write gold items both labellers agreed on, with their "
            "merged label. That gold set is teacher-agreed, not human-verified.",
        ),
    ] = False,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Items per labeller request.")
    ] = 8,
    max_items: Annotated[
        int | None,
        typer.Option(
            "--max-items",
            min=1,
            help="Label at most this many currently unlabelled items per labeller, then exit cleanly.",
        ),
    ] = None,
    include_unreviewed: Annotated[
        bool,
        typer.Option(
            "--include-unreviewed",
            help="Also write flagged items no human has reviewed yet, with their merged label.",
        ),
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Seed for gold-set sampling.")] = 0,
    labeller_a: Annotated[
        TeacherProvider,
        typer.Option(
            "--labeller-a",
            envvar="DISTILL_LABELLER_A",
            help="Teacher for labeller A. The default runs locally through Ollama.",
        ),
    ] = TeacherProvider(DEFAULT_LABELLER_A),
    labeller_b: Annotated[
        TeacherProvider,
        typer.Option(
            "--labeller-b",
            envvar="DISTILL_LABELLER_B",
            help="Teacher for labeller B; pick a different model family from labeller A.",
        ),
    ] = TeacherProvider(DEFAULT_LABELLER_B),
    # The labeller pair is chosen only by --labeller-a/--labeller-b, never by --writer.
) -> None:
    """Soft-label states with two labellers and queue disagreements for human review.

    Re-run it after reviewing: labelled items are never re-sent to a teacher, and the outputs
    are rewritten with every human decision applied.
    """
    from .labelling import Labeller, label_pending
    from .review_queue import ReviewQueue
    from .synthesis import make_registry, read_rows, write_jsonl

    spec = load_decision(decision_file)
    questions = spec.model_dump(mode="json")
    registry = make_registry()
    try:
        # Refuse a hosted labeller before reading the corpus or touching the queue.
        for provider in (labeller_a, labeller_b):
            registry.labeller(provider.value)
    except RoleRefused as error:
        typer.echo(f"distill label: {error}", err=True)
        raise typer.Exit(code=2) from error
    with ReviewQueue(queue_path) as queue:
        queue.add(read_rows(input_file, questions), split="train")
        if gold_from is not None:
            gold_rows = read_rows(gold_from, questions)
            random.Random(seed).shuffle(gold_rows)
            queue.add(gold_rows, split="gold", limit=gold_size)
            if heldout_output is not None:
                # Rows already in the gold set (or the training corpus) are left where they are.
                queue.add(gold_rows, split="heldout")
        elif heldout_output is not None:
            typer.echo("--heldout-output needs --gold-from", err=True)
            raise typer.Exit(code=2)
        report = label_pending(
            queue,
            registry,
            labellers=(Labeller("a", labeller_a.value), Labeller("b", labeller_b.value)),
            batch_size=batch_size,
            max_items=max_items,
            progress=lambda message: typer.echo(message, err=True),
        )
        write_jsonl(queue.export_rows("train", include_unreviewed=include_unreviewed), output)
        write_jsonl(queue.export_rows("gold", teacher_gold=teacher_gold), gold_output)
        if heldout_output is not None:
            write_jsonl(queue.export_rows("heldout"), heldout_output)
        counts = queue.counts()
    typer.echo(
        json.dumps(
            {
                **counts,
                "flagged_rate": round(counts["flagged"] / counts["labelled"], 4)
                if counts["labelled"]
                else 0.0,
                "failed_batches": len(report.failed_batches),
            }
        )
    )
    for failure in report.failed_batches:
        typer.echo(f"skipped batch ({failure}); re-run to retry it", err=True)
    if teacher_gold:
        typer.echo(
            f"{gold_output} holds teacher-agreed labels, not human-verified ones; flagged gold "
            "items are left out until a human decides them.",
            err=True,
        )
    if counts["pending"]:
        typer.echo(
            f"{counts['pending']} item(s) await a human: run `distill review --queue "
            f"{queue_path}`, then re-run distill label to refresh {output} and {gold_output}.",
            err=True,
        )
    if report.failed_batches:
        raise typer.Exit(code=1)


@app.command()
def review(
    queue_path: Annotated[
        Path, typer.Option("--queue", help="Review-queue database written by distill label.")
    ] = Path("review-queue.sqlite3"),
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help="Preferred local port (default 8765); a free one is used if taken.",
        ),
    ] = None,
) -> None:
    """Open the local review page for flagged items and the gold set (127.0.0.1 only)."""
    from .review_app import DEFAULT_PORT, HOST, find_free_port, serve

    port = DEFAULT_PORT if port is None else port
    if not queue_path.exists():
        typer.echo(f"{queue_path} does not exist; run distill label first.", err=True)
        raise typer.Exit(code=1)
    chosen = find_free_port(port)
    if chosen != port:
        typer.echo(f"Port {port} is in use; using {chosen} instead.", err=True)
    typer.echo(f"Review queue: http://{HOST}:{chosen}/  (Ctrl+C to stop)")
    serve(queue_path, chosen)


@app.command()
def train(
    decision_file: Annotated[
        Path, typer.Option("--decision-file", "-d", help="Path to decision.yaml.")
    ] = Path("decision.yaml"),
    data: Annotated[
        Path, typer.Option("--data", "-i", help="Labelled JSONL from distill label.")
    ] = Path("labelled.jsonl"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Destination Laya checkpoint directory.")
    ] = Path("model"),
    base: Annotated[
        str, typer.Option("--base", help="Laya base checkpoint directory or Hub model id.")
    ] = "convaiinnovations/laya",
    device: Annotated[
        str, typer.Option("--device", help="Training device: mps, cpu, or cuda.")
    ] = "mps",
    epochs: Annotated[int, typer.Option("--epochs", min=1, help="Training epochs.")] = 4,
    micro_batch: Annotated[int, typer.Option("--micro-batch", min=1)] = 8,
    grad_accum: Annotated[int, typer.Option("--grad-accum", min=1)] = 4,
    lr_encoder: Annotated[float, typer.Option("--lr-encoder", min=0.0)] = 2.5e-5,
    lr_head: Annotated[float, typer.Option("--lr-head", min=0.0)] = 1e-4,
    precision: Annotated[
        str | None,
        typer.Option("--precision", help="fp32, fp16, or bf16; device default if omitted."),
    ] = None,
    gradient_checkpointing: Annotated[
        bool, typer.Option("--gradient-checkpointing/--no-gradient-checkpointing")
    ] = False,
    save_dtype: Annotated[
        str, typer.Option("--save-dtype", help="fp32 or fp16 checkpoint weights.")
    ] = "fp32",
    cleanlab: Annotated[
        bool, typer.Option("--cleanlab", help="Relabel the worst ~2% then train one more pass.")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume an interrupted run for this output path.")
    ] = False,
) -> None:
    """Fine-tune, calibrate, and atomically write a local Laya decision model."""
    # Keep the staged command honest: rows carry the runtime question mapping, while
    # decision.yaml remains the reviewed contract for that mapping.
    load_decision(decision_file)
    options = TrainingOptions(
        device=device,
        epochs=epochs,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        save_dtype=save_dtype,
        cleanlab=cleanlab,
    )
    try:
        metrics = run_training(data, output, base, options, resume=resume)
    except TrainingError as error:
        typer.echo(f"distill train: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "test": metrics["test"],
                "temperatures": metrics["temperatures"],
                "train_seconds": metrics["train_seconds"],
            }
        )
    )


@app.command()
def eval(
    decision_file: Annotated[
        Path, typer.Option("--decision-file", "-d", help="Path to decision.yaml.")
    ] = Path("decision.yaml"),
    model: Annotated[
        Path, typer.Option("--model", "-m", help="Fine-tuned Laya checkpoint.")
    ] = Path("model"),
    heldout: Annotated[Path, typer.Option("--heldout", help="Labelled held-out JSONL.")] = Path(
        "heldout.jsonl"
    ),
    gold: Annotated[Path, typer.Option("--gold", help="Human-verified gold JSONL.")] = Path(
        "gold.jsonl"
    ),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Evaluation report directory.")
    ] = Path("eval-report"),
    base: Annotated[
        str, typer.Option("--base", help="Out-of-box Laya checkpoint or Hub id.")
    ] = "convaiinnovations/laya",
    device: Annotated[str, typer.Option("--device", help="Laya inference device.")] = "mps",
    target_accuracy: Annotated[
        float, typer.Option("--target-accuracy", min=0.000001, max=1.0)
    ] = 0.97,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    reference: Annotated[
        TeacherProvider | None,
        typer.Option(
            "--reference",
            envvar="DISTILL_REFERENCE",
            help="Optional reference teacher whose answers on the gold and held-out sets the "
            "model is compared with: a local Ollama model, or anthropic-api/openai-api (needs "
            "that API key). With none, the readiness gate reports 'no reference measured'.",
        ),
    ] = None,
    reference_model: Annotated[
        str | None,
        typer.Option("--reference-model", help="Override the reference provider's model id."),
    ] = None,
) -> None:
    """Measure calibration, abstention coverage, robustness, the base model and any reference."""
    load_decision(decision_file)
    try:
        report = run_evaluation(
            model=model,
            base=base,
            heldout=heldout,
            gold=gold,
            output=output,
            device=device,
            target_accuracy=target_accuracy,
            batch_size=batch_size,
            reference=None if reference is None else reference.value,
            reference_model=reference_model,
        )
    except EvaluationError as error:
        typer.echo(f"distill eval: {error}", err=True)
        raise typer.Exit(1) from error
    status = "ready" if report["pass_bar"]["ready"] else "not ready"
    typer.echo(f"distill eval: {status}; report written to {output / 'REPORT.md'}")
    for note in report["pass_bar"]["notes"]:
        typer.echo(f"- {note}")
    for failure in report["pass_bar"]["failures"]:
        typer.echo(f"- {failure}")


@app.command()
def export(
    decision_file: Annotated[
        Path, typer.Option("--decision-file", "-d", help="Path to decision.yaml.")
    ] = Path("decision.yaml"),
    model: Annotated[
        Path, typer.Option("--model", "-m", help="Finished Laya checkpoint from distill train.")
    ] = Path("model"),
    evaluation: Annotated[
        Path, typer.Option("--evaluation", "-e", help="distill eval report directory or metrics.json.")
    ] = Path("eval-report"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="New self-contained model folder.")
    ] = Path("export"),
    base_checkpoint: Annotated[
        str | None,
        typer.Option("--base", help="Base checkpoint used for training; recorded as provenance."),
    ] = None,
    teacher: Annotated[
        list[str], typer.Option("--teacher", help="Labelling teacher to record; repeat as needed.")
    ] = [],
    fallback: Annotated[
        TeacherProvider | None,
        typer.Option(
            "--fallback",
            help="Teacher the exported routing snippet asks when the local model is not "
            "confident. Default: none, so the snippet returns the local answer flagged "
            "low-confidence. A hosted API fallback needs that API key.",
        ),
    ] = None,
) -> None:
    """Package a trained checkpoint and its measured model card."""
    load_decision(decision_file)
    try:
        result = run_export(
            model=model,
            evaluation=evaluation,
            output=output,
            base_checkpoint=base_checkpoint,
            teachers=tuple(teacher),
            fallback=None if fallback is None else fallback.value,
        )
    except ExportError as error:
        typer.echo(f"distill export: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"distill export: wrote {output}; tau = {result['tau']['threshold']}")
