# Adapted from Laya's fine-tuning notebook (Apache-2.0, Convai Innovations);
# modified by the Distill authors for single-process local (Apple MPS) training.
"""Local Laya fine-tuning, adapted from the measured T0 training spike.

The public helpers in this module deliberately use only the standard library.  The
torch/Laya imports live inside :func:`run_training`, keeping the normal command
surface and unit tests free of model runtime dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LR_ENCODER = 2.5e-5
LR_HEAD = 1e-4
MICRO_BATCH = 8
GRAD_ACCUM = 4
SIGMA_START = 0.4
SIGMA_END = 0.1


@dataclass(frozen=True)
class TrainingOptions:
    """The T0 recipe and the few practical controls exposed by ``distill train``."""

    device: str = "mps"
    epochs: int = 4
    micro_batch: int = MICRO_BATCH
    grad_accum: int = GRAD_ACCUM
    lr_encoder: float = LR_ENCODER
    lr_head: float = LR_HEAD
    precision: str | None = None
    gradient_checkpointing: bool = False
    save_dtype: str = "fp32"
    seed: int = 20260922
    cleanlab: bool = False


class TrainingError(RuntimeError):
    """An actionable error from the training workflow."""


def split_rows(rows: Sequence[dict[str, Any]], seed: int = 20260922) -> tuple[list, list, list]:
    """Return deterministic 80/10/10 row splits without leaking a state across splits."""
    if len(rows) < 10:
        raise TrainingError("need at least 10 labelled rows for an 80/10/10 split")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    tenth = len(rows) // 10
    calibration = [rows[i] for i in order[:tenth]]
    test = [rows[i] for i in order[tenth : 2 * tenth]]
    train = [rows[i] for i in order[2 * tenth :]]
    return train, calibration, test


def decode_state(value: Any) -> Any:
    """A row's state: notebook rows hold JSON text, while synthesis also writes plain text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def optimizer_updates(item_count: int, micro_batch: int, grad_accum: int, epochs: int) -> int:
    """Count actual optimizer updates, including a final partial accumulation group."""
    if min(item_count, micro_batch, grad_accum, epochs) < 1:
        raise ValueError("item count, batch sizes, and epochs must be positive")
    return math.ceil(math.ceil(item_count / micro_batch) / grad_accum) * epochs


def cosine_learning_rate(
    initial_lr: float, step: int, total_steps: int, eta_min: float = 1e-6
) -> float:
    """The cosine schedule used by T0, expressed without torch for fast testing."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    step = min(max(step, 0), total_steps)
    return eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * step / total_steps)) / 2


def _nll_at_log_temperature(
    samples: Sequence[tuple[Sequence[float], Sequence[float]]], log_t: float
) -> float:
    inv_t = math.exp(-log_t)
    total = 0.0
    for logits, target in samples:
        scaled = [float(x) * inv_t for x in logits]
        peak = max(scaled)
        log_z = peak + math.log(sum(math.exp(x - peak) for x in scaled))
        total -= sum(float(y) * (x - log_z) for x, y in zip(scaled, target, strict=True))
    return total / len(samples)


def fit_temperature(samples: Sequence[tuple[Sequence[float], Sequence[float]]]) -> float:
    """Fit one bounded temperature with scalar golden-section minimisation.

    This is equivalent to T0's L-BFGS objective but avoids making torch a package
    import requirement.  A type with fewer than ten calibration items remains at 1.
    """
    if len(samples) < 10:
        return 1.0
    low, high = math.log(0.1), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    f_left, f_right = (
        _nll_at_log_temperature(samples, left),
        _nll_at_log_temperature(samples, right),
    )
    for _ in range(80):
        if f_left <= f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = _nll_at_log_temperature(samples, left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = _nll_at_log_temperature(samples, right)
    return round(math.exp((low + high) / 2), 6)


def fit_temperatures(
    logits: Sequence[Sequence[float]], items: Sequence[dict[str, Any]]
) -> list[float]:
    """Fit one calibration temperature for choice, score, and noul respectively."""
    fitted = []
    for qtype in range(3):
        samples = [
            (z, item["target"])
            for z, item in zip(logits, items, strict=True)
            if item["qtype"] == qtype
        ]
        fitted.append(fit_temperature(samples))
    return fitted


def checkpoint_layout(path: Path) -> set[str]:
    """The required root entries for a directory readable by ``laya.load(path)``."""
    return {"model.safetensors", "encoder", "tokenizer", "rl_agent_config.json"}


def write_checkpoint(
    path: Path,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    temperatures: Sequence[float],
    training_meta: dict[str, Any],
    *,
    save_weights: Callable[[dict[str, Any], str], None],
) -> None:
    """Atomically publish a Laya checkpoint, never exposing a half-written final path."""
    if path.exists():
        raise TrainingError(f"checkpoint already exists: {path}; choose a new --output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.writing-", dir=path.parent))
    try:
        save_weights(model.state_dict(), str(staging / "model.safetensors"))
        model.encoder.config.save_pretrained(staging / "encoder")
        tokenizer.save_pretrained(staging / "tokenizer")
        saved_config = dict(config)
        saved_config.update(
            fine_tuned=True,
            model_name="laya-distill",
            temperature=list(temperatures),
            training=training_meta,
        )
        saved_config.pop("temperature_by_options", None)
        (staging / "rl_agent_config.json").write_text(
            json.dumps(saved_config, indent=2, sort_keys=True), encoding="utf-8"
        )
        missing = checkpoint_layout(staging) - {child.name for child in staging.iterdir()}
        if missing:
            raise TrainingError(f"checkpoint writer missed required entries: {sorted(missing)}")
        os.replace(staging, path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise TrainingError(f"{path}:{line_no}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict) or not {"state", "questions", "gold"} <= set(row):
            raise TrainingError(f"{path}:{line_no}: expected state, questions, and gold fields")
        rows.append(row)
    if not rows:
        raise TrainingError(f"{path}: contains no labelled rows")
    return rows


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime():
    """Import training-only dependencies only when a user actually starts training."""
    try:
        import torch
        from laya.agent import Agent, _fix_tokenizer_config
        from laya.common import (
            QTYPE_NAMES,
            QTYPES,
            build_model,
            build_sequence,
            collate_items,
            proper_reward,
            render_options,
        )
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TrainingError(
            "distill train needs an environment with torch, transformers, safetensors, and laya; "
            "the project CI intentionally does not install model runtime dependencies"
        ) from error
    return (
        torch,
        Agent,
        _fix_tokenizer_config,
        QTYPE_NAMES,
        QTYPES,
        build_model,
        build_sequence,
        collate_items,
        proper_reward,
        render_options,
        load_file,
        save_file,
        AutoTokenizer,
    )


def run_training(
    data_path: Path, output: Path, base: str, options: TrainingOptions, *, resume: bool = False
) -> dict[str, Any]:
    """Train, calibrate, test, and atomically publish a Laya checkpoint.

    A sibling ``.<output>.incomplete`` directory is the only persistent work-in-progress
    state.  It contains epoch checkpoints for resume and is never a valid Laya model.
    """
    if options.device not in {"mps", "cpu", "cuda"}:
        raise TrainingError("--device must be mps, cpu, or cuda")
    if options.epochs < 1:
        raise TrainingError("--epochs must be at least one")
    rows = _read_rows(data_path)
    train_rows, calibration_rows, test_rows = split_rows(rows, options.seed)
    state_dir = output.parent / f".{output.name}.incomplete"
    if output.exists():
        raise TrainingError(f"output already exists and is complete: {output}")
    if state_dir.exists() and not resume:
        raise TrainingError(f"interrupted run found at {state_dir}; re-run with --resume")
    if resume and not state_dir.exists():
        raise TrainingError(f"no interrupted run found at {state_dir}")

    (
        torch,
        Agent,
        fix_tokenizer,
        qtype_names,
        qtypes,
        build_model,
        build_sequence,
        collate_items,
        proper_reward,
        render_options,
        load_file,
        save_file,
        AutoTokenizer,
    ) = _runtime()
    if options.device == "mps" and not torch.backends.mps.is_available():
        raise TrainingError("--device mps requested but torch.backends.mps.is_available() is False")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("--device cuda requested but torch.cuda.is_available() is False")
    device = torch.device(options.device)
    torch.manual_seed(options.seed)
    precision = options.precision or (
        "bf16" if device.type == "mps" else "fp16" if device.type == "cuda" else "fp32"
    )
    if precision not in {"fp32", "fp16", "bf16"}:
        raise TrainingError("--precision must be fp32, fp16, or bf16")
    if options.save_dtype not in {"fp32", "fp16"}:
        raise TrainingError("--save-dtype must be fp32 or fp16")

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    def resolve_model_dir(value: str) -> str:
        if os.path.isdir(value):
            return value
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(
                value,
                allow_patterns=[
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                ],
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return snapshot_download(
                value,
                allow_patterns=[
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                ],
            )

    def load_base(model_dir: str):
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights
        fix_tokenizer(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text(encoding="utf-8"))
        with no_init_weights():
            model = build_model(
                cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False
            )
        model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
        return model, tokenizer, cfg

    def row_items(
        source_rows: Iterable[dict[str, Any]], tokenizer: Any, cfg: dict[str, Any]
    ) -> tuple[list[dict], int]:
        items, dropped = [], 0
        for row_index, row in enumerate(source_rows):
            state = decode_state(row["state"])
            questions, gold = (
                json.loads(row[key]) if isinstance(row[key], str) else row[key]
                for key in ("questions", "gold")
            )
            for qid, qdef in questions.items():
                if qid not in gold:
                    continue
                Agent._check_question(qid, qdef)
                question = Agent._to_internal(qdef)
                kind, criteria = question["t"], question["crit"]
                probs = gold[qid]["probabilities"]
                if kind == "noul":
                    target = [probs.get("false", 0.5), probs.get("true", 0.5)]
                elif kind == "score":
                    target = [probs.get(str(i), 0.0) for i in range(len(criteria))]
                else:
                    target = [probs.get(key, 0.0) for key in criteria]
                total = sum(target)
                target = (
                    [value / total for value in target]
                    if total
                    else [1 / len(target)] * len(target)
                )
                sequence, markers = build_sequence(
                    tokenizer, state, question, cfg["max_len"], cfg["head_max_len"]
                )
                if len(markers) != len(render_options(question)):
                    dropped += 1
                    continue
                items.append(
                    {
                        "ids": sequence,
                        "markers": markers,
                        "qtype": qtypes[kind],
                        "target": target,
                        "label": max(range(len(target)), key=target.__getitem__),
                        "row": row_index,
                        "qid": qid,
                    }
                )
        return items, dropped

    def forward(model: Any, batch: dict[str, Any]):
        context = (
            __import__("contextlib").nullcontext()
            if precision == "fp32"
            else torch.autocast(
                device_type=device.type,
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision],
            )
        )
        with context:
            return model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )

    def rlcd_loss(logits: Any, target: Any, mask: Any, qtype: Any, sigma: float):
        logits = logits.float()
        option_count = mask.sum(-1, keepdim=True).float()
        eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
        eps = (eps - eps.sum(-1, keepdim=True) / option_count) * mask
        noisy = logits.detach().unsqueeze(0) + eps
        probabilities = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            reward = proper_reward(
                probabilities, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0
            )
            advantage = reward - reward.mean(0, keepdim=True)
            advantage = advantage / (advantage.std() + 1e-6)
        log_prob = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
        loss_rl = -(advantage * log_prob).mean()
        loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        return loss_rl, loss_ce, reward.mean()

    def fit_runtime_temperatures(logits: list[list[float]], items: list[dict]) -> list[float]:
        """T0's L-BFGS calibration fit, clamped to Laya's supported range."""
        temperatures = [1.0, 1.0, 1.0]
        for qtype in range(3):
            selected = [
                (values, item["target"])
                for values, item in zip(logits, items, strict=True)
                if item["qtype"] == qtype
            ]
            if len(selected) < 10:
                continue
            max_options = max(len(values) for values, _target in selected)
            z = torch.full((len(selected), max_options), -1e4)
            target = torch.zeros((len(selected), max_options))
            for index, (values, targets) in enumerate(selected):
                z[index, : len(values)] = torch.tensor(values)
                target[index, : len(targets)] = torch.tensor(targets)
            log_temperature = torch.zeros(1, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

            def closure():
                optimizer.zero_grad()
                loss = -(target * torch.log_softmax(z / log_temperature.exp(), -1)).sum(-1).mean()
                loss.backward()
                return loss

            optimizer.step(closure)
            temperatures[qtype] = float(torch.clamp(log_temperature.exp(), 0.5, 5.0).item())
        return temperatures

    @torch.no_grad()
    def predict(model: Any, tokenizer: Any, items: list[dict]) -> list[list[float]]:
        was_training = model.training
        model.eval()
        values = []
        for start in range(0, len(items), 16):
            chunk = items[start : start + 16]
            logits, _ = forward(model, collate_items([chunk], tokenizer.pad_token_id))
            values.extend(
                logits.float().cpu().tolist()[index][: len(item["markers"])]
                for index, item in enumerate(chunk)
            )
        model.train(was_training)
        return values

    def evaluate(logits: list[list[float]], items: list[dict]) -> dict[str, dict[str, float | int]]:
        buckets: dict[str, dict[str, float]] = {}
        for z, item in zip(logits, items, strict=True):
            peak = max(z)
            probs = [math.exp(value - peak) for value in z]
            total = sum(probs)
            probs = [value / total for value in probs]
            ce = -sum(
                target * math.log(max(prob, 1e-12))
                for target, prob in zip(item["target"], probs, strict=True)
            )
            correct = float(max(range(len(probs)), key=probs.__getitem__) == item["label"])
            for key in ("all", qtype_names[item["qtype"]]):
                bucket = buckets.setdefault(key, {"n": 0, "soft_ce": 0.0, "acc": 0.0})
                bucket["n"] += 1
                bucket["soft_ce"] += ce
                bucket["acc"] += correct
        return {
            key: {
                "n": int(value["n"]),
                "soft_ce": round(value["soft_ce"] / value["n"], 4),
                "acc": round(value["acc"] / value["n"], 4),
            }
            for key, value in buckets.items()
        }

    model_dir = resolve_model_dir(base)
    model, tokenizer, config = load_base(model_dir)
    if options.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device)
    sync()
    train_items, dropped_train = row_items(train_rows, tokenizer, config)
    calibration_items, dropped_calibration = row_items(calibration_rows, tokenizer, config)
    test_items, dropped_test = row_items(test_rows, tokenizer, config)
    if not train_items or not calibration_items or not test_items:
        raise TrainingError("each split must yield at least one trainable question item")

    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "training-state.pt"
    manifest_file = state_dir / "manifest.json"
    source_digest = _source_digest(data_path)
    completed_epochs, history = 0, []
    optimizer_state = scheduler_state = None
    if resume:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("source_digest") != source_digest or manifest.get("options") != asdict(
            options
        ):
            raise TrainingError(
                "cannot resume: data or training options differ from the interrupted run"
            )
        saved = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        completed_epochs, history = saved["completed_epochs"], saved["history"]
        optimizer_state, scheduler_state = saved["optimizer"], saved["scheduler"]
    else:
        manifest_file.write_text(
            json.dumps(
                {
                    "source": str(data_path),
                    "source_digest": source_digest,
                    "options": asdict(options),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    enc = [parameter for name, parameter in model.named_parameters() if "encoder." in name]
    head = [parameter for name, parameter in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW(
        [{"params": enc, "lr": options.lr_encoder}, {"params": head, "lr": options.lr_head}],
        weight_decay=0.01,
    )
    total_updates = optimizer_updates(
        len(train_items), options.micro_batch, options.grad_accum, options.epochs
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
    start = time.perf_counter()
    updates = scheduler.last_epoch + 1 if completed_epochs else 0
    try:
        for epoch in range(completed_epochs, options.epochs):
            model.train()
            epoch_items = list(train_items)
            random.Random(options.seed + epoch).shuffle(epoch_items)
            optimizer.zero_grad(set_to_none=True)
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, options.epochs - 1)
            epoch_losses = []
            for batch_number, offset in enumerate(
                range(0, len(epoch_items), options.micro_batch), start=1
            ):
                chunk = epoch_items[offset : offset + options.micro_batch]
                batch = collate_items([chunk], tokenizer.pad_token_id)
                logits, action = forward(model, batch)
                loss_rl, loss_ce, reward = rlcd_loss(
                    logits,
                    batch["target"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                    sigma,
                )
                loss = (loss_rl + loss_ce) / options.grad_accum + 0.0 * action.sum()
                scaler.scale(loss).backward()
                last = offset + options.micro_batch >= len(epoch_items)
                if batch_number % options.grad_accum == 0 or last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                epoch_losses.append(
                    {
                        "loss": float(loss.item() * options.grad_accum),
                        "soft_ce": float(loss_ce.item()),
                        "reward": float(reward.item()),
                    }
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": round(
                        sum(item["loss"] for item in epoch_losses) / len(epoch_losses), 6
                    ),
                    "mean_soft_ce": round(
                        sum(item["soft_ce"] for item in epoch_losses) / len(epoch_losses), 6
                    ),
                    "updates": updates,
                }
            )
            payload = {
                "completed_epochs": epoch + 1,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }
            temp_state = state_file.with_suffix(".tmp")
            torch.save(payload, temp_state)
            os.replace(temp_state, state_file)
    except KeyboardInterrupt:
        raise TrainingError(
            f"training interrupted; resume with --resume (state kept in {state_dir})"
        ) from None

    calibration_logits = predict(model, tokenizer, calibration_items)
    temperatures = fit_runtime_temperatures(calibration_logits, calibration_items)
    cleanlab_result: dict[str, Any] | None = None
    if options.cleanlab:
        try:
            from cleanlab.rank import get_label_quality_scores
        except ImportError as error:
            raise TrainingError(
                "--cleanlab needs cleanlab installed in the training environment"
            ) from error
        train_logits = predict(model, tokenizer, train_items)
        probabilities = []
        for z in train_logits:
            peak = max(z)
            exp = [math.exp(value - peak) for value in z]
            probabilities.append([value / sum(exp) for value in exp])
        # Cleanlab needs rectangular class probabilities, so rank each question type separately.
        relabeled = 0
        for qtype in range(3):
            indexes = [index for index, item in enumerate(train_items) if item["qtype"] == qtype]
            if not indexes:
                continue
            labels = [train_items[index]["label"] for index in indexes]
            probs = [probabilities[index] for index in indexes]
            quality = get_label_quality_scores(labels=labels, pred_probs=probs)
            count = max(1, math.ceil(len(indexes) * 0.02))
            for local_index in sorted(range(len(indexes)), key=quality.__getitem__)[:count]:
                item = train_items[indexes[local_index]]
                item["target"] = probs[local_index]
                item["label"] = max(
                    range(len(probs[local_index])), key=probs[local_index].__getitem__
                )
                relabeled += 1
        # The optional pass deliberately starts at the base model.  Teacher JSONL is
        # immutable: the replacement targets exist only for this second pass.
        first_history = history
        model, tokenizer, config = load_base(model_dir)
        if options.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.head_checkpointing = True
        model.to(device)
        enc = [parameter for name, parameter in model.named_parameters() if "encoder." in name]
        head = [parameter for name, parameter in model.named_parameters() if "encoder." not in name]
        optimizer = torch.optim.AdamW(
            [{"params": enc, "lr": options.lr_encoder}, {"params": head, "lr": options.lr_head}],
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=optimizer_updates(
                len(train_items), options.micro_batch, options.grad_accum, options.epochs
            ),
            eta_min=1e-6,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
        history, updates = [], 0
        for epoch in range(options.epochs):
            model.train()
            epoch_items = list(train_items)
            random.Random(options.seed + epoch).shuffle(epoch_items)
            optimizer.zero_grad(set_to_none=True)
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, options.epochs - 1)
            epoch_losses = []
            for batch_number, offset in enumerate(
                range(0, len(epoch_items), options.micro_batch), start=1
            ):
                chunk = epoch_items[offset : offset + options.micro_batch]
                batch = collate_items([chunk], tokenizer.pad_token_id)
                logits, action = forward(model, batch)
                loss_rl, loss_ce, reward = rlcd_loss(
                    logits,
                    batch["target"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                    sigma,
                )
                loss = (loss_rl + loss_ce) / options.grad_accum + 0.0 * action.sum()
                scaler.scale(loss).backward()
                last = offset + options.micro_batch >= len(epoch_items)
                if batch_number % options.grad_accum == 0 or last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                epoch_losses.append(
                    {
                        "loss": float(loss.item() * options.grad_accum),
                        "soft_ce": float(loss_ce.item()),
                        "reward": float(reward.item()),
                    }
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": round(
                        sum(item["loss"] for item in epoch_losses) / len(epoch_losses), 6
                    ),
                    "mean_soft_ce": round(
                        sum(item["soft_ce"] for item in epoch_losses) / len(epoch_losses), 6
                    ),
                    "updates": updates,
                }
            )
        calibration_logits = predict(model, tokenizer, calibration_items)
        temperatures = fit_runtime_temperatures(calibration_logits, calibration_items)
        cleanlab_result = {
            "enabled": True,
            "relabeled_items": relabeled,
            "fraction": round(relabeled / len(train_items), 4),
            "first_pass_history": first_history,
        }

    dtype = torch.float16 if options.save_dtype == "fp16" else torch.float32

    def save_weights(state_dict: dict[str, Any], target: str) -> None:
        save_file(
            {key: value.to(dtype).contiguous().cpu() for key, value in state_dict.items()}, target
        )

    test_logits = predict(model, tokenizer, test_items)
    metrics = {
        "status": "ok",
        "data": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
            "test_rows": len(test_rows),
            "train_items": len(train_items),
            "calibration_items": len(calibration_items),
            "test_items": len(test_items),
            "dropped": dropped_train + dropped_calibration + dropped_test,
        },
        "options": asdict(options),
        "history": history,
        "temperatures": temperatures,
        "test": evaluate(test_logits, test_items),
        "calibration": evaluate(calibration_logits, calibration_items),
        "train_seconds": round(time.perf_counter() - start, 2),
        "cleanlab": cleanlab_result,
    }
    write_checkpoint(
        output, model, tokenizer, config, temperatures, metrics, save_weights=save_weights
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.rmtree(state_dir)
    return metrics
