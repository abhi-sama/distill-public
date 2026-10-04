from __future__ import annotations

import socket

import pytest
from conftest import answer
from fastapi.testclient import TestClient

from distill import review_app
from distill.labelling import LabellerAnswer, option_keys
from distill.review_app import HOST, create_app, find_free_port
from distill.review_queue import ReviewQueue

HTMX = {"HX-Request": "true"}


@pytest.fixture
def queue(tmp_path, questions, rows):
    """Item 1 agrees, item 2 is flagged, item 3 is a gold-set item."""
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows[:2], split="train")
        queue.add([{"state": {"text": "gold one"}, "questions": questions}], split="gold")
        for item_id in (1, 2, 3):
            queue.record_answer(item_id, "a", answer(questions, "ollama-qwen"))
            picks = {"escalate": {"false": 0.3, "true": 0.7}} if item_id == 2 else {}
            queue.record_answer(item_id, "b", answer(questions, "ollama-gemma", **picks))
        yield queue


@pytest.fixture
def client(queue):
    return TestClient(create_app(queue, allowed_hosts=["testserver"]))


def form_for(questions, **overrides):
    """Form fields for a one-hot-on-first-option label, with per-question overrides."""
    data = {}
    for q_index, (qid, question) in enumerate(questions.items()):
        chosen = overrides.get(qid, {option_keys(question)[0]: 1})
        for o_index, option in enumerate(option_keys(question)):
            data[f"q{q_index}_o{o_index}"] = str(chosen.get(option, 0))
    return data


def test_the_index_lists_what_awaits_a_human(client) -> None:
    page = client.get("/").text

    assert '<a href="/items/2">2</a>' in page and '<a href="/items/3">3</a>' in page
    assert '<a href="/items/1">1</a>' not in page
    assert "escalate: different_answer" in page and "gold set" in page
    assert "<b>2</b> awaiting a human" in page
    assert '<a href="/items/1">1</a>' in client.get("/?view=all").text


def test_an_item_shows_both_distributions_and_why_it_was_flagged(client) -> None:
    page = client.get("/items/2").text

    assert "A · ollama-qwen" in page and "B · ollama-gemma" in page
    assert "Why this is here" in page and "A picks &#x27;false&#x27; (0.90)" in page
    assert "B picks &#x27;true&#x27; (0.70)" in page
    assert "ollama-gemma reasoning" in page
    # Flagged items start from the merged label: escalate = (0.9+0.3)/2, (0.1+0.7)/2.
    assert 'name="q0_o0" value="0.6"' in page and 'name="q0_o1" value="0.4"' in page
    assert page.count('name="q0_o0"') == 1


def test_gold_items_fold_the_labellers_away_and_start_uniform(client) -> None:
    page = client.get("/items/3").text

    assert "every gold item needs a human label" in page
    assert "<details><summary>Show the labellers' answers</summary>" in page
    assert 'name="q0_o0" value="0.5"' in page and 'name="q1_o0" value="0.25"' in page
    assert page.count('name="q0_o0"') == 1


def test_untrusted_text_is_escaped(tmp_path, questions) -> None:
    hostile = '<script>alert("x")</script>'
    with ReviewQueue(tmp_path / "hostile.sqlite3") as queue:
        queue.add([{"state": {"text": hostile}, "questions": questions}], split="gold")
        bad = answer(questions, "ollama-qwen")
        queue.record_answer(
            1, "a", LabellerAnswer(bad.provider, bad.distributions, {"escalate": hostile})
        )
        client = TestClient(create_app(queue, allowed_hosts=["testserver"]))

        for page in (client.get("/").text, client.get("/items/1").text):
            assert hostile not in page
            assert "&lt;script&gt;" in page


def test_saving_a_decision_overwrites_the_merged_label(client, queue, questions) -> None:
    response = client.post(
        "/items/2/decision",
        data={**form_for(questions, escalate={"true": 3, "false": 1}), "note": "legal threat"},
        headers=HTMX,
    )

    assert response.status_code == 200 and "Saved" in response.text
    assert 'name="q0_o1" value="0.75"' in response.text
    item = queue.get(2)
    assert item.human["escalate"]["probabilities"] == {"false": 0.25, "true": 0.75}
    assert item.note == "legal threat" and not item.pending
    exported = [row for row in queue.export_rows("train") if row["state"] == {"text": "ticket 1"}]
    assert exported[0]["gold"] == item.human


def test_an_invalid_decision_is_refused_with_a_visible_error(client, queue, questions) -> None:
    response = client.post(
        "/items/2/decision",
        data=form_for(questions, escalate={"false": 0, "true": 0}),
        headers=HTMX,
    )

    assert response.status_code == 422
    assert "Not saved: escalate: probabilities must not all be zero" in response.text
    assert queue.get(2).human is None

    response = client.post("/items/2/decision", data={"q0_o0": "lots"}, headers=HTMX)
    assert response.status_code == 422 and "needs a number" in response.text


def test_save_and_next_moves_to_the_next_pending_item(client, queue, questions) -> None:
    response = client.post(
        "/items/2/decision", data={**form_for(questions), "action": "next"}, headers=HTMX
    )
    assert response.status_code == 204 and response.headers["HX-Redirect"] == "/items/3"

    response = client.post(
        "/items/3/decision",
        data={**form_for(questions), "action": "next"},
        follow_redirects=False,
    )
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert queue.counts()["pending"] == 0


def test_requests_from_outside_localhost_are_refused(queue, questions) -> None:
    local = TestClient(create_app(queue), base_url="http://127.0.0.1:8765")
    assert local.get("/").status_code == 200
    assert local.get("/", headers={"host": "evil.example"}).status_code == 400

    cross_site = local.post(
        "/items/2/decision",
        data=form_for(questions),
        headers={"origin": "https://evil.example"},
    )
    assert cross_site.status_code == 403 and queue.get(2).human is None
    other_port = local.post(
        "/items/2/decision", data=form_for(questions), headers={"origin": "http://127.0.0.1:3000"}
    )
    assert other_port.status_code == 403 and queue.get(2).human is None
    same_site = local.post(
        "/items/2/decision", data=form_for(questions), headers={"origin": "http://127.0.0.1:8765"}
    )
    assert same_site.status_code in {200, 303} and queue.get(2).human is not None


def test_htmx_is_served_locally(client) -> None:
    response = client.get("/static/htmx.min.js")

    assert response.status_code == 200 and "htmx" in response.text[:200]
    assert '<script src="/static/htmx.min.js">' in client.get("/").text


def test_find_free_port_skips_a_port_already_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind((HOST, 0))
        taken = busy.getsockname()[1]

        port = find_free_port(taken)

    assert port != taken and port > 0


def test_serve_binds_only_to_loopback(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.update(kwargs))

    review_app.serve(tmp_path / "queue.sqlite3", 8766)

    assert calls["host"] == "127.0.0.1" and calls["port"] == 8766
