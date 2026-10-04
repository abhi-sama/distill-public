import json
import shutil

import pytest
from conftest import EXAMPLE, LOCAL_A, LOCAL_B, FakeLabeller, fake_registry
from typer.testing import CliRunner

from distill.cli import app
from distill.labelling import option_keys
from distill.review_queue import ReviewQueue
from distill.synthesis import write_jsonl

runner = CliRunner()


def test_train_reports_missing_reviewed_decision_file_without_starting_runtime(tmp_path):
    result = runner.invoke(
        app,
        ["train", "--decision-file", str(tmp_path / "missing.yaml"), "--data", "unused.jsonl"],
    )

    assert result.exit_code == 1
    assert "missing.yaml" in str(result.exception)


def test_label_ignores_the_writer_and_keeps_the_default_labeller_pair(
    tmp_path, monkeypatch, questions
) -> None:
    decision = tmp_path / "decision.yaml"
    shutil.copy(EXAMPLE, decision)
    rows = [
        {"state": {"text": f"ticket {n}"}, "questions": questions, "gold": {}} for n in range(3)
    ]
    write_jsonl(rows, tmp_path / "synthetic.jsonl")
    writer = FakeLabeller("ollama-gemma-dense", questions)
    local = [FakeLabeller(LOCAL_A, questions), FakeLabeller(LOCAL_B, questions)]
    monkeypatch.setattr(
        "distill.synthesis.make_registry", lambda **_: fake_registry(writer, *local)
    )
    monkeypatch.setenv("DISTILL_WRITER", "ollama-gemma-dense")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "label",
            "-d",
            str(decision),
            "-i",
            str(tmp_path / "synthetic.jsonl"),
            "-o",
            str(tmp_path / "labelled.jsonl"),
            "--queue",
            str(tmp_path / "queue.sqlite3"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert writer.prompts == []
    assert len(local[0].prompts) == len(local[1].prompts) == 1
    assert f"labeller a ({LOCAL_A})" in result.stderr


@pytest.mark.parametrize("command", ["init", "synth"])
@pytest.mark.parametrize("provider", ["anthropic-api", "openai-api"])
def test_hosted_api_writers_are_refused_with_a_one_line_reason(
    tmp_path, monkeypatch, command, provider
) -> None:
    monkeypatch.chdir(tmp_path)
    shutil.copy(EXAMPLE, tmp_path / "decision.yaml")
    arguments = ["init", "Escalate?"] if command == "init" else ["synth"]

    result = runner.invoke(app, [*arguments, "--writer", provider])

    assert result.exit_code == 2
    assert f"{provider} cannot be the writer" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_hosted_api_labellers_are_refused_with_a_one_line_reason(tmp_path, monkeypatch, questions) -> None:
    decision = tmp_path / "decision.yaml"
    shutil.copy(EXAMPLE, decision)
    write_jsonl(
        [{"state": {"text": "ticket"}, "questions": questions, "gold": {}}],
        tmp_path / "synthetic.jsonl",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["label", "-d", str(decision), "-i", str(tmp_path / "synthetic.jsonl"),
         "--labeller-a", "openai-api", "--queue", str(tmp_path / "queue.sqlite3")],
    )

    assert result.exit_code == 2
    assert "openai-api cannot be the labeller" in result.stderr
    assert not (tmp_path / "queue.sqlite3").exists()


def test_the_default_writer_is_the_local_dense_model_and_there_is_no_teacher_pin() -> None:
    from distill.teachers import DEFAULT_WRITER

    assert DEFAULT_WRITER == "ollama-gemma-dense"
    for command in ("init", "synth", "label"):
        result = runner.invoke(app, [command, "--help"])
        assert "--teacher-pin" not in result.output
    assert "--writer" in runner.invoke(app, ["synth", "--help"]).output


def test_label_writes_soft_labels_queues_disagreements_and_resumes(
    tmp_path, monkeypatch, questions
) -> None:
    decision = tmp_path / "decision.yaml"
    shutil.copy(EXAMPLE, decision)
    rows = [
        {"state": {"text": f"ticket {n}"}, "questions": questions, "gold": {}} for n in range(4)
    ]
    heldout = [
        {"state": {"text": f"held {n}"}, "questions": questions, "gold": {}} for n in range(5)
    ]
    write_jsonl(rows, tmp_path / "synthetic.jsonl")
    write_jsonl(heldout, tmp_path / "heldout.jsonl")
    a = FakeLabeller(LOCAL_A, questions)
    b = FakeLabeller(LOCAL_B, questions, {"ticket 2": {"category": {"legal": 1.0}}})
    monkeypatch.setattr("distill.synthesis.make_registry", lambda **_: fake_registry(a, b))
    arguments = [
        "label",
        "-d",
        str(decision),
        "-i",
        str(tmp_path / "synthetic.jsonl"),
        "-o",
        str(tmp_path / "labelled.jsonl"),
        "--queue",
        str(tmp_path / "queue.sqlite3"),
        "--gold-from",
        str(tmp_path / "heldout.jsonl"),
        "--gold-size",
        "3",
        "--gold-output",
        str(tmp_path / "gold.jsonl"),
    ]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["labelled"] == 7 and summary["gold"] == 3 and summary["flagged"] == 1
    assert summary["pending"] == 4 and "distill review" in result.stderr
    labelled = [json.loads(line) for line in (tmp_path / "labelled.jsonl").read_text().splitlines()]
    assert len(labelled) == 3 and all(
        set(row) == {"state", "questions", "gold"} for row in labelled
    )
    assert labelled[0]["gold"]["escalate"]["label"] == "false"
    assert (tmp_path / "gold.jsonl").read_text() == ""

    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        flagged = queue.items("review")[0]
        queue.decide(flagged.id, {q: {k: 1.0} for q, k in _firsts(questions).items()})
    prompts = len(a.prompts)
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert len(a.prompts) == prompts
    labelled = [json.loads(line) for line in (tmp_path / "labelled.jsonl").read_text().splitlines()]
    assert len(labelled) == 4


def test_label_without_a_reviewer_writes_teacher_gold_and_a_labelled_heldout_set(
    tmp_path, monkeypatch, questions
) -> None:
    decision = tmp_path / "decision.yaml"
    shutil.copy(EXAMPLE, decision)
    write_jsonl(
        [{"state": {"text": "ticket"}, "questions": questions, "gold": {}}],
        tmp_path / "synthetic.jsonl",
    )
    heldout = [
        {"state": {"text": f"held {n}"}, "questions": questions, "gold": {}} for n in range(6)
    ]
    write_jsonl(heldout, tmp_path / "heldout.jsonl")
    a = FakeLabeller(LOCAL_A, questions)
    b = FakeLabeller(
        LOCAL_B,
        questions,
        {f"held {n}": {"category": {"legal": 1.0}} for n in range(6) if n % 2},
    )
    monkeypatch.setattr("distill.synthesis.make_registry", lambda **_: fake_registry(a, b))
    arguments = [
        "label",
        "-d",
        str(decision),
        "-i",
        str(tmp_path / "synthetic.jsonl"),
        "-o",
        str(tmp_path / "labelled.jsonl"),
        "--queue",
        str(tmp_path / "queue.sqlite3"),
        "--gold-from",
        str(tmp_path / "heldout.jsonl"),
        "--gold-size",
        "3",
        "--gold-output",
        str(tmp_path / "gold.jsonl"),
        "--heldout-output",
        str(tmp_path / "heldout-labelled.jsonl"),
        "--teacher-gold",
    ]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "not human-verified" in result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["gold"] == 3 and summary["heldout"] == 3 and summary["labelled"] == 7
    gold = _read_jsonl(tmp_path / "gold.jsonl")
    labelled_heldout = _read_jsonl(tmp_path / "heldout-labelled.jsonl")
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        gold_items = [item for item in queue.items() if item.split == "gold"]
    agreed = [item.state for item in gold_items if not item.needs_review]
    # Gold keeps only the teacher-agreed items; held-out keeps every labelled item.
    assert [row["state"] for row in gold] == agreed
    assert len(labelled_heldout) == 3
    gold_states = {item.state["text"] for item in gold_items}
    assert {row["state"]["text"] for row in labelled_heldout} == {
        f"held {n}" for n in range(6)
    } - gold_states
    assert all(row["gold"]["category"]["label"] for row in labelled_heldout)

    # A re-run neither grows the gold set nor re-asks either teacher.
    prompts = len(a.prompts) + len(b.prompts)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(a.prompts) + len(b.prompts) == prompts
    assert json.loads(result.stdout.strip().splitlines()[-1])["gold"] == 3


def test_label_heldout_output_needs_gold_from(tmp_path) -> None:
    decision = tmp_path / "decision.yaml"
    shutil.copy(EXAMPLE, decision)
    write_jsonl([], tmp_path / "synthetic.jsonl")

    result = runner.invoke(
        app,
        [
            "label",
            "-d",
            str(decision),
            "-i",
            str(tmp_path / "synthetic.jsonl"),
            "--queue",
            str(tmp_path / "q.sqlite3"),
            "--heldout-output",
            str(tmp_path / "h.jsonl"),
        ],
    )

    assert result.exit_code == 2 and "--gold-from" in result.stderr


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _firsts(questions):
    return {qid: option_keys(question)[0] for qid, question in questions.items()}


def test_label_rejects_a_corpus_written_for_another_decision(tmp_path, monkeypatch) -> None:
    write_jsonl([{"state": "x", "questions": {"other": {}}, "gold": {}}], tmp_path / "s.jsonl")
    a, b = FakeLabeller(LOCAL_A, {}), FakeLabeller(LOCAL_B, {})
    monkeypatch.setattr("distill.synthesis.make_registry", lambda **_: fake_registry(a, b))

    result = runner.invoke(
        app,
        [
            "label",
            "-d",
            str(EXAMPLE),
            "-i",
            str(tmp_path / "s.jsonl"),
            "--queue",
            str(tmp_path / "q.sqlite3"),
        ],
    )

    assert result.exit_code != 0
    assert "differ from the decision file" in str(result.exception)
    assert a.prompts == b.prompts == []


def test_review_needs_an_existing_queue(tmp_path) -> None:
    result = runner.invoke(app, ["review", "--queue", str(tmp_path / "missing.sqlite3")])

    assert result.exit_code == 1 and "run distill label first" in result.output


def test_review_serves_on_loopback_and_moves_off_a_busy_port(tmp_path, monkeypatch) -> None:
    ReviewQueue(tmp_path / "queue.sqlite3").close()
    served = {}
    monkeypatch.setattr("distill.review_app.find_free_port", lambda port: 51234)
    monkeypatch.setattr("distill.review_app.serve", lambda path, port: served.update(port=port))

    result = runner.invoke(app, ["review", "--queue", str(tmp_path / "queue.sqlite3")])

    assert result.exit_code == 0, result.output
    assert served == {"port": 51234}
    assert "Port 8765 is in use; using 51234" in result.stderr
    assert "http://127.0.0.1:51234/" in result.stdout


def test_eval_reports_missing_reviewed_decision_file_without_starting_runtime(tmp_path):
    result = runner.invoke(app, ["eval", "--decision-file", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1
    assert "missing.yaml" in str(result.exception)
