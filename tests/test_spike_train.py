"""Tests for scripts/spike_train.py, the T0 MPS training spike.

They need torch, transformers and laya but no model download: the end-to-end test builds a tiny
random ModernBERT checkpoint and a word-level tokenizer in a temp dir, trains it with the
script's own `main()`, and reloads the result through `laya.load()`.
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

# torch, transformers and laya are not project dependencies yet (T5 adds them), so CI skips this.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
Agent = pytest.importorskip("laya.agent").Agent
_laya_common = pytest.importorskip("laya.common")
QTYPES, render_options = _laya_common.QTYPES, _laya_common.render_options

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("spike_train", ROOT / "scripts" / "spike_train.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
RUN_LIVE_TRAINING = os.environ.get("DISTILL_LIVE_TRAINING") == "1"


def make_tiny_checkpoint(path: Path) -> Path:
    """A random 2-layer ModernBERT Laya checkpoint in the layout laya.load() reads."""
    from laya.common import build_model
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, PreTrainedTokenizerFast

    pre = pre_tokenizers.Whitespace()
    words = set()
    texts = [r["state"] + r["questions"] for r in st.generate_rows(200, seed=0)]
    texts += [f"{t} question:" for t in QTYPES] + [
        "false true level no yes the statement does not hold"
    ]
    for text in texts:
        words.update(w for w, _ in pre.pre_tokenize_str(text))
    vocab = {
        t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words))
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tok.save_pretrained(path / "tokenizer")

    ecfg = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        local_attention=16,
        global_attn_every_n_layers=2,
        max_position_embeddings=512,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
    )
    ecfg.save_pretrained(path / "encoder")
    cfg = {
        "encoder": str(path / "encoder"),
        "head_layers": 1,
        "max_len": 256,
        "head_max_len": 96,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.0, 1.0, 1.0],
    }
    (path / "rl_agent_config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    model = build_model(cfg, encoder_dir=str(path / "encoder"), pretrained=False)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors")
    )
    return path


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    return make_tiny_checkpoint(tmp_path_factory.mktemp("tiny_laya"))


@pytest.fixture(scope="module")
def tiny_tok_cfg(tiny_ckpt):
    _, tok, cfg = st.load_base(str(tiny_ckpt))
    return tok, cfg


# --- data -----------------------------------------------------------------------------------


def test_generator_is_deterministic():
    assert st.generate_rows(50, seed=0) == st.generate_rows(50, seed=0)
    assert st.generate_rows(50, seed=0) != st.generate_rows(50, seed=1)


def test_rows_follow_notebook_schema_and_laya_question_rules():
    rows = st.generate_rows(200, seed=3)
    assert len({r["id"] for r in rows}) == 200
    for row in rows:
        state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
        assert isinstance(state["ticket"], str) and state["ticket"]
        assert set(gold) == set(questions)
        for qid, qdef in questions.items():
            Agent._check_question(qid, qdef)  # raises on anything laya would reject
            probs, label = gold[qid]["probabilities"], gold[qid]["label"]
            assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
            if qdef["type"] == "choice":
                assert list(probs) == list(qdef["criteria"])
            elif qdef["type"] == "noul":
                assert set(probs) == {"false", "true"} and label in ("false", "true")
            else:
                assert list(probs) == [str(i) for i in range(len(qdef["criteria"]))]
            assert str(label) == max(probs, key=probs.get)


def test_generator_covers_every_label():
    golds = [json.loads(r["gold"]) for r in st.generate_rows(200, seed=0)]
    assert {g["category"]["label"] for g in golds} == set(st.QUESTIONS["category"]["criteria"])
    assert {g["urgency"]["label"] for g in golds} == {0, 1, 2, 3}
    for qid in ("escalate", "manipulation"):
        assert {g[qid]["label"] for g in golds} == {"false", "true"}


def test_pad_to_chars_makes_long_tickets():
    for row in st.generate_rows(5, seed=0, pad_to_chars=2000):
        assert len(json.loads(row["state"])["ticket"]) >= 2000


def test_training_targets_follow_option_order(tiny_tok_cfg):
    tok, cfg = tiny_tok_cfg
    row = st.generate_rows(1, seed=0)[0]
    state, gold = json.loads(row["state"]), json.loads(row["gold"])
    for qid, qdef in st.QUESTIONS.items():
        it = st.build_training_item(tok, cfg, state, qdef, gold[qid])
        opts = render_options(Agent._to_internal(qdef))
        assert len(it["target"]) == len(it["markers"]) == len(opts)
        probs = gold[qid]["probabilities"]
        if qdef["type"] == "noul":
            expected = [probs["false"], probs["true"]]
        else:
            expected = list(probs.values())
        assert it["target"] == pytest.approx(expected)
        assert it["qtype"] == QTYPES[qdef["type"]]
        assert it["label"] == it["target"].index(max(it["target"]))


def test_rows_to_items_and_calibration_split(tiny_tok_cfg):
    tok, cfg = tiny_tok_cfg
    rows = st.generate_rows(50, seed=0)
    items, dropped = st.rows_to_items(rows, tok, cfg)
    assert (len(items), dropped) == (200, 0)
    assert len(st.rows_to_items(rows, tok, cfg, limit=37)[0]) == 37

    train, calib = st.split_calibration(items)
    assert (len(train), len(calib)) == (180, 20)

    def key(it):
        return it["row"], it["qid"]

    assert not {key(i) for i in train} & {key(i) for i in calib}
    assert [key(i) for i in st.split_calibration(items)[1]] == [key(i) for i in calib]
    big = [{"row": i, "qid": "q"} for i in range(10_000)]
    assert len(st.split_calibration(big)[1]) == st.CALIB_MAX


# --- loss and calibration -------------------------------------------------------------------


def _toy_batch():
    target = torch.tensor([[0.1, 0.8, 0.1, 0.0], [0.3, 0.7, 0.0, 0.0], [0.05, 0.15, 0.6, 0.2]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    qtype = torch.tensor([QTYPES["choice"], QTYPES["noul"], QTYPES["score"]])
    return target, mask, qtype


def test_rlcd_loss_ce_term_and_masking():
    target, mask, qtype = _toy_batch()
    torch.manual_seed(0)
    logits = torch.randn(3, 4, requires_grad=True)
    loss_rl, loss_ce, reward = st.rlcd_loss(logits, target, mask, qtype, sigma=0.4)
    manual = (
        -(target * torch.log_softmax(logits.detach().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    )
    assert loss_ce.item() == pytest.approx(manual.item(), rel=1e-6)
    assert torch.isfinite(loss_rl) and torch.isfinite(reward)
    (loss_rl + loss_ce).backward()
    assert torch.all(logits.grad[~mask] == 0)


def test_rlcd_loss_goes_down_under_optimisation():
    target, mask, qtype = _toy_batch()
    torch.manual_seed(0)
    logits = torch.zeros(3, 4, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    ces = []
    for _ in range(200):
        loss_rl, loss_ce, _ = st.rlcd_loss(logits, target, mask, qtype, sigma=0.4)
        opt.zero_grad()
        (loss_rl + loss_ce).backward()
        opt.step()
        ces.append(loss_ce.item())
    entropy = -(target * torch.log(target.clamp_min(1e-12))).sum(-1).mean().item()
    assert ces[-1] < ces[0] - 0.2
    assert ces[-1] == pytest.approx(entropy, abs=0.05)  # soft CE bottoms out at the target entropy


@pytest.mark.parametrize("t_true", [0.6, 1.0, 2.5])
def test_fit_one_temp_recovers_temperature(t_true):
    g = torch.Generator().manual_seed(0)
    sel = []
    for i in range(300):
        k = 2 + i % 4
        z = torch.randn(k, generator=g) * 3
        sel.append((z.tolist(), torch.softmax(z / t_true, -1).tolist()))
    assert st.fit_one_temp(sel) == pytest.approx(t_true, rel=0.02)


def test_fit_one_temp_needs_ten_items():
    assert st.fit_one_temp([([1.0, 0.0], [1.0, 0.0])] * 9) == 1.0


def test_resolve_device_never_falls_back():
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="cuda"):
            st.resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(SystemExit, match="mps"):
            st.resolve_device("mps")
    assert st.resolve_device("cpu").type == "cpu"


def test_mps_fallbacks_and_missing_kernels_are_named(tmp_path, monkeypatch):
    """Ops that fall back to CPU (warning) or have no MPS kernel (error) land in metrics.json."""
    import warnings

    def fake_run(args, device, mem, metrics):
        warnings.warn(
            "The operator 'aten::_foo' is not currently supported on the MPS backend and will "
            "fall back to run on the CPU. This may have performance implications.",
            stacklevel=2,
        )
        raise NotImplementedError(
            "The operator 'aten::_bar' is not currently implemented for the MPS device."
        )

    monkeypatch.setattr(st, "run", fake_run)
    assert st.main(["--device", "cpu", "--out", str(tmp_path)]) == 1
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["status"] == "error" and m["error"].startswith("NotImplementedError")
    assert m["mps_cpu_fallback_ops"] == ["aten::_foo"]
    assert m["mps_unsupported_ops_errored"] == ["aten::_bar"]


# --- end to end ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.skipif(
    not RUN_LIVE_TRAINING, reason="set DISTILL_LIVE_TRAINING=1 to run a real tiny training loop"
)
def test_end_to_end_tiny_checkpoint_round_trips_through_laya(tiny_ckpt, tmp_path, device):
    import laya

    out = tmp_path / "run"
    rc = st.main(
        [
            "--device",
            device,
            "--base",
            str(tiny_ckpt),
            "--out",
            str(out),
            "--n-items",
            "48",
            "--check-rows",
            "4",
            "--save-dtype",
            "fp32",
            "--log-every",
            "0",
        ]
    )
    assert rc == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["status"] == "ok"
    assert m["mps_cpu_fallback_ops"] == []
    assert m["data"] == {
        "rows": 12,
        "items": 48,
        "dropped": 0,
        "max_len": 256,
        "head_max_len": 96,
        "train": 44,
        "calib": 4,
    }
    assert m["timing"]["micro_batches"] == 6 and m["timing"]["updates"] == 2
    assert len(m["history"]) == 6 and all(math.isfinite(h["loss"]) for h in m["history"])
    assert set(m["eval"]["before"]["heldout"]) >= {"all"}

    # The saved checkpoint is what laya itself loads and serves.
    agent = laya.load(str(out / "checkpoint"), device="cpu")
    res = agent.predict(json.loads(st.generate_rows(1, seed=7)[0]["state"]), st.QUESTIONS)
    assert set(res["answers"]) == set(st.QUESTIONS)
    assert sum(res["answers"]["category"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    cfg = json.loads((out / "checkpoint" / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"] is True and "temperature_by_options" not in cfg
    assert cfg["temperature"] == m["temperatures"]["fitted"]

    # T7 only packages the trained checkpoint and recorded T5/T6 outputs: no model is
    # downloaded or retrained on this path.  The resulting folder must still load directly.
    from distill.exporting import run_export

    checkpoint = out / "checkpoint"
    (checkpoint / "metrics.json").write_text(
        json.dumps({"data": {"rows": 12}, "options": {"epochs": 1}}), encoding="utf-8"
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "local": {
                    "heldout": {"accuracy": 1.0, "ece_15": 0.0, "brier": 0.0},
                    "latency": {"mean_ms_per_decision": 1.0},
                },
                "base_laya": {},
                "reference": None,
                "tau": {"threshold": 0.5, "coverage": 1.0, "accuracy": 1.0, "n": 1},
            }
        ),
        encoding="utf-8",
    )
    exported = tmp_path / "exported"
    run_export(model=checkpoint, evaluation=evaluation, output=exported)
    assert laya.load(str(exported), device="cpu") is not None

    # fp32 weights: laya's predictions equal the live model's up to its 4-decimal rounding
    # (plus fp16 autocast when laya runs on MPS).
    tol = 1e-3 if device == "cpu" else 2e-2
    assert m["reload"]["max_abs_prob_diff_vs_live_model"] <= tol
    assert m["reload"]["agent_device"] == device


@pytest.mark.skipif(
    not RUN_LIVE_TRAINING, reason="set DISTILL_LIVE_TRAINING=1 to run a real tiny training loop"
)
def test_bench_mode_times_long_sequences_without_saving(tiny_ckpt, tmp_path):
    out = tmp_path / "bench"
    assert (
        st.main(
            [
                "--device",
                "cpu",
                "--base",
                str(tiny_ckpt),
                "--out",
                str(out),
                "--bench-seq-len",
                "200",
                "--bench-updates",
                "2",
                "--grad-accum",
                "2",
                "--log-every",
                "0",
            ]
        )
        == 0
    )
    m = json.loads((out / "metrics.json").read_text())
    assert m["timing"]["updates"] == 2 and m["timing"]["micro_batches"] == 4
    # every sequence is truncated at the checkpoint's max_len, so nothing in the batch is padding
    assert (
        m["timing"]["mean_padded_len"]
        == m["timing"]["mean_real_tokens_per_seq"]
        == m["data"]["max_len"]
    )
    assert not (out / "checkpoint").exists() and "reload" not in m


@pytest.mark.skipif(
    not RUN_LIVE_TRAINING, reason="set DISTILL_LIVE_TRAINING=1 to run a real tiny training loop"
)
def test_training_moves_the_weights(tiny_ckpt, tmp_path):
    """With fp32 saving, the checkpoint differs from the base: training did update the model."""
    from safetensors.torch import load_file

    out = tmp_path / "run"
    assert (
        st.main(
            [
                "--device",
                "cpu",
                "--base",
                str(tiny_ckpt),
                "--out",
                str(out),
                "--n-items",
                "48",
                "--check-rows",
                "2",
                "--save-dtype",
                "fp32",
                "--log-every",
                "0",
            ]
        )
        == 0
    )
    base = load_file(str(tiny_ckpt / "model.safetensors"))
    tuned = load_file(str(out / "checkpoint" / "model.safetensors"))
    assert set(base) == set(tuned)
    moved = [k for k in base if not torch.equal(base[k], tuned[k])]
    assert any(k.startswith("encoder.") for k in moved) and any(
        k.startswith("scorer.") for k in moved
    )
