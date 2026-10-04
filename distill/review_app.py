"""The local FastAPI + htmx review page for flagged items and the human gold set.

It binds only to 127.0.0.1. Synthetic states can hold adversarial text, so every value is
HTML-escaped, the Host header is checked against local names (against DNS rebinding), and
state-changing requests from another origin are refused.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .labelling import InvalidDistribution, normalise, option_keys
from .review_queue import VIEWS, QueueItem, ReviewQueue

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOCAL_HOSTS = ("127.0.0.1", "localhost")
_STATIC = Path(__file__).with_name("review_static")
_VIEW_TITLES = {
    "review": "Needs a human",
    "gold": "Gold set",
    "reviewed": "Reviewed",
    "all": "All items",
}


def find_free_port(preferred: int = DEFAULT_PORT) -> int:
    """``preferred`` when nothing listens on it locally, else an OS-assigned free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((HOST, port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise OSError("no free local port")


def serve(queue_path: str | Path, port: int) -> None:
    """Run the review page on 127.0.0.1 until interrupted."""
    import uvicorn

    queue = ReviewQueue(queue_path)
    try:
        uvicorn.run(create_app(queue), host=HOST, port=port, log_level="warning")
    finally:
        queue.close()


def create_app(queue: ReviewQueue, *, allowed_hosts: Sequence[str] = LOCAL_HOSTS) -> FastAPI:
    """Build the review app over an open queue."""
    app = FastAPI(title="Distill review queue", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next: Any) -> Response:
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
                return PlainTextResponse("cross-origin request refused", status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index(view: str = "review") -> HTMLResponse:
        if view not in VIEWS:
            view = "review"
        return HTMLResponse(_index_page(queue, view))

    @app.get("/next")
    def next_item(after: int = 0) -> Response:
        item = queue.next_pending(after)
        return RedirectResponse(f"/items/{item.id}" if item else "/", status_code=303)

    @app.get("/items/{item_id}", response_class=HTMLResponse)
    def item_page(item_id: int) -> HTMLResponse:
        item = queue.get(item_id)
        if item is None:
            return HTMLResponse(_page("Not found", "<p>No such item.</p>", queue), 404)
        return HTMLResponse(_page(f"Item {item.id}", _item_body(item), queue))

    @app.post("/items/{item_id}/decision")
    async def decide(item_id: int, request: Request) -> Response:
        item = queue.get(item_id)
        if item is None:
            return PlainTextResponse("no such item", status_code=404)
        form = await request.form()
        is_htmx = request.headers.get("hx-request") == "true"
        try:
            distributions = _form_distributions(item, form)
            decided = queue.decide(item_id, distributions, note=str(form.get("note", "")))
        except InvalidDistribution as error:
            return HTMLResponse(_decision_panel(item, error=str(error)), status_code=422)
        if form.get("action") == "next":
            following = queue.next_pending(item_id)
            target = f"/items/{following.id}" if following else "/"
            if is_htmx:
                return Response(status_code=204, headers={"HX-Redirect": target})
            return RedirectResponse(target, status_code=303)
        if is_htmx:
            return HTMLResponse(_decision_panel(decided, saved=True))
        return RedirectResponse(f"/items/{item_id}", status_code=303)

    return app


def _form_distributions(item: QueueItem, form: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read ``q<question index>_o<option index>`` fields into per-question probabilities."""
    distributions: dict[str, dict[str, Any]] = {}
    for q_index, (question_id, question) in enumerate(item.questions.items()):
        probabilities: dict[str, Any] = {}
        for o_index, option in enumerate(option_keys(question)):
            raw = str(form.get(f"q{q_index}_o{o_index}", "")).strip()
            try:
                probabilities[option] = float(raw) if raw else 0.0
            except ValueError as error:
                raise InvalidDistribution(
                    f"{question_id}: {option!r} needs a number, got {raw!r}"
                ) from error
        try:
            normalise(probabilities, option_keys(question))
        except InvalidDistribution as error:
            raise InvalidDistribution(f"{question_id}: {error}") from error
        distributions[question_id] = probabilities
    return distributions


def _page(title: str, body: str, queue: ReviewQueue) -> str:
    counts = queue.counts()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="htmx-config" content='{_HTMX_CONFIG}'>
<title>{escape(title)} · Distill review</title>
<script src="/static/htmx.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<header class="top">
  <a class="brand" href="/">Distill review</a>
  <span class="stat"><b>{counts["pending"]}</b> awaiting a human</span>
  <span class="stat">gold <b>{counts["gold_done"]}</b>/{counts["gold"]}</span>
  <span class="stat">flagged <b>{counts["flagged"]}</b> of {counts["labelled"]} labelled</span>
  <a class="button" href="/next">Next item →</a>
</header>
<main>{body}</main>
<script>{_SCRIPT}</script>
</body>
</html>"""


def _index_page(queue: ReviewQueue, view: str) -> str:
    tabs = "".join(
        f'<a class="tab{" active" if name == view else ""}" href="/?view={name}">'
        f"{escape(title)}</a>"
        for name, title in _VIEW_TITLES.items()
    )
    items = queue.items(view)
    if items:
        rows = "".join(_index_row(item) for item in items)
        table = (
            '<table class="list"><thead><tr><th>#</th><th>Set</th><th>Status</th>'
            f"<th>Why</th><th>State</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        table = '<p class="empty">Nothing here.</p>'
    return _page(_VIEW_TITLES[view], f'<nav class="tabs">{tabs}</nav>{table}', queue)


def _index_row(item: QueueItem) -> str:
    return (
        f'<tr><td><a href="/items/{item.id}">{item.id}</a></td>'
        f"<td>{escape(item.split)}</td><td>{_status(item)}</td>"
        f"<td>{escape(_why_short(item))}</td>"
        f'<td class="preview">{escape(_preview(item.state))}</td></tr>'
    )


def _status(item: QueueItem) -> str:
    if item.human is not None:
        return '<span class="pill done">human</span>'
    if item.pending:
        return '<span class="pill todo">awaiting human</span>'
    if item.merged is not None:
        return '<span class="pill ok">merged</span>'
    return '<span class="pill">unlabelled</span>'


def _why_short(item: QueueItem) -> str:
    reasons = sorted({f"{flag['question']}: {flag['trigger']}" for flag in item.flags})
    if item.split == "gold":
        reasons.insert(0, "gold set")
    return "; ".join(reasons)


def _preview(state: Any, limit: int = 140) -> str:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _item_body(item: QueueItem) -> str:
    state = (
        item.state
        if isinstance(item.state, str)
        else json.dumps(item.state, ensure_ascii=False, indent=2)
    )
    return f"""
<p class="crumbs"><a href="/?view=review">← queue</a> · item {item.id} · {escape(item.split)}
 · {_status(item)}</p>
{_reasons(item)}
<section class="card"><h2>State</h2><pre class="state">{escape(state)}</pre></section>
{_decision_panel(item)}
"""


def _reasons(item: QueueItem) -> str:
    lines = [
        f"<li><b>{escape(flag['question'])}</b> · {escape(flag['trigger'].replace('_', ' '))}: "
        f"{escape(flag['detail'])}</li>"
        for flag in item.flags
    ]
    if item.split == "gold":
        lines.insert(
            0,
            "<li><b>gold set</b>: every gold item needs a human label, set independently of "
            "the labellers.</li>",
        )
    if not lines:
        return ""
    return f'<section class="card why"><h2>Why this is here</h2><ul>{"".join(lines)}</ul></section>'


def _decision_panel(item: QueueItem, *, error: str | None = None, saved: bool = False) -> str:
    """The editable soft label for every question; htmx swaps this whole panel on save."""
    questions = "".join(
        _question_block(item, q_index, question_id, question)
        for q_index, (question_id, question) in enumerate(item.questions.items())
    )
    notice = ""
    if error:
        notice = f'<p class="notice error" role="alert">Not saved: {escape(error)}</p>'
    elif saved:
        notice = (
            '<p class="notice saved" role="status">Saved. This label now overrides the merge.</p>'
        )
    return f"""
<form id="decision" class="decision" method="post" action="/items/{item.id}/decision"
      hx-post="/items/{item.id}/decision" hx-target="this" hx-swap="outerHTML">
{notice}
{questions}
<section class="card">
  <label for="note"><h2>Note (optional)</h2></label>
  <textarea id="note" name="note" rows="2">{escape(item.note)}</textarea>
  <div class="actions">
    <button type="submit" name="action" value="save">Save label</button>
    <button type="submit" name="action" value="next" class="primary">Save and next →</button>
  </div>
</section>
</form>"""


def _question_block(item: QueueItem, q_index: int, question_id: str, question: Mapping) -> str:
    options = option_keys(question)
    descriptions = _option_descriptions(question, options)
    answer_a, answer_b = item.answers["a"], item.answers["b"]
    merged = (item.merged or {}).get(question_id)
    human = (item.human or {}).get(question_id)
    # Gold labels anchor truth independently of the labellers, so an undecided gold item
    # starts from uniform with the labellers' answers folded away.
    if human is not None:
        start = human["probabilities"]
    elif merged is not None and item.split != "gold":
        start = merged["probabilities"]
    else:
        start = {option: 1 / len(options) for option in options}
    show_teachers = answer_a is not None or answer_b is not None
    fold_teachers = show_teachers and item.split == "gold" and human is None

    def option_cell(o_index: int, option: str) -> str:
        return f"<td><b>{escape(option)}</b>{descriptions[o_index]}</td>"

    def input_cell(o_index: int, option: str) -> str:
        return (
            f'<td class="final"><input type="number" name="q{q_index}_o{o_index}" '
            f'value="{_fmt(start.get(option, 0.0))}" min="0" max="1" step="any" '
            f'inputmode="decimal" aria-label="{escape(question_id)} {escape(option)}">'
            f'<button type="button" class="onehot" data-question="q{q_index}" '
            f'data-option="{o_index}" title="Put all probability on this option">1.0</button></td>'
        )

    tables = []
    if show_teachers:
        header = (
            "<th>Option</th>"
            f"<th>A · {escape(answer_a.provider if answer_a else 'pending')}</th>"
            f"<th>B · {escape(answer_b.provider if answer_b else 'pending')}</th><th>Merged</th>"
        )
        if not fold_teachers:
            header += "<th>Final label</th>"
        rows = []
        for o_index, option in enumerate(options):
            cells = option_cell(o_index, option)
            cells += _prob_cell(answer_a.distributions[question_id][option] if answer_a else None)
            cells += _prob_cell(answer_b.distributions[question_id][option] if answer_b else None)
            cells += _prob_cell(merged["probabilities"][option] if merged else None)
            if not fold_teachers:
                cells += input_cell(o_index, option)
            rows.append(f"<tr>{cells}</tr>")
        tables.append(_table(header, rows))
        rationales = "".join(
            f"<p><b>{name} · {escape(answer.provider)}:</b> "
            f"{escape(answer.rationales.get(question_id, ''))}</p>"
            for name, answer in (("A", answer_a), ("B", answer_b))
            if answer is not None
        )
        if rationales:
            tables.append(f'<div class="rationales">{rationales}</div>')
        if fold_teachers:
            tables = [
                "<details><summary>Show the labellers' answers</summary>"
                f"{''.join(tables)}</details>"
            ]
    if not show_teachers or fold_teachers:
        rows = [
            f"<tr>{option_cell(o_index, option)}{input_cell(o_index, option)}</tr>"
            for o_index, option in enumerate(options)
        ]
        tables.append(_table("<th>Option</th><th>Final label</th>", rows))
    return f"""
<section class="card question" data-question="q{q_index}">
  <h2>{escape(question_id)} <span class="type">{escape(question["type"])}</span></h2>
  <p class="instructions">{escape(_text(question.get("instructions")))}</p>
  {"".join(tables)}
</section>"""


def _table(header: str, rows: list[str]) -> str:
    return (
        f'<table class="dist"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fmt(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"


def _prob_cell(value: float | None) -> str:
    if value is None:
        return '<td class="prob muted">–</td>'
    percent = max(0.0, min(100.0, value * 100))
    return (
        f'<td class="prob"><span class="bar"><span style="width:{percent:.1f}%"></span></span>'
        f"{value:.2f}</td>"
    )


def _option_descriptions(question: Mapping, options: list[str]) -> list[str]:
    criteria = question.get("criteria")
    if question["type"] == "noul":
        labels = question.get("labels") or {}
        criteria = {str(key).lower(): value for key, value in (criteria or {}).items()} or {
            option: labels.get(option) for option in options
        }
        texts = [criteria.get(option) for option in options]
    elif isinstance(criteria, Mapping):
        texts = list(criteria.values())
    elif question["type"] == "score":
        texts = list(criteria)
    else:
        texts = [None] * len(options)
    return [
        f'<div class="criterion">{escape(_text(text))}</div>' if text not in (None, "") else ""
        for text in texts
    ]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# htmx 2 ignores 4xx bodies by default; swap 422 so validation errors reach the reviewer.
_HTMX_CONFIG = json.dumps(
    {
        "responseHandling": [
            {"code": "204", "swap": False},
            {"code": "[23]..", "swap": True},
            {"code": "422", "swap": True},
            {"code": "...", "swap": False, "error": True},
        ]
    }
)

_SCRIPT = """
document.addEventListener('click', (event) => {
  const button = event.target.closest('button.onehot');
  if (!button) return;
  const q = button.dataset.question;
  button.closest('form').querySelectorAll(`input[name^="${q}_o"]`).forEach((input) => {
    input.value = input.name === `${q}_o${button.dataset.option}` ? '1' : '0';
  });
});
"""

_CSS = """
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e2e0;
  --accent:#2f5bea; --warn:#b4541a; --ok:#2d7d46; --bar:#c9d4fb; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1d1d20; --ink:#ececf0;
  --muted:#9a9aa3; --line:#2e2e33; --accent:#7c9bff; --warn:#f0a36e; --ok:#6fcf8e;
  --bar:#34427a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
a { color:var(--accent); }
.top { position:sticky; top:0; display:flex; flex-wrap:wrap; gap:.4rem 1.2rem;
  align-items:center; padding:.7rem 1rem; background:var(--card);
  border-bottom:1px solid var(--line); z-index:1; }
.brand { font-weight:700; text-decoration:none; color:var(--ink); margin-right:auto; }
.stat { color:var(--muted); } .stat b { color:var(--ink); }
main { max-width:960px; margin:0 auto; padding:1rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:.9rem 1rem; margin:0 0 1rem; min-width:0; }
.card h2 { font-size:1rem; margin:0 0 .5rem; }
.why { border-left:4px solid var(--warn); } .why ul { margin:0; padding-left:1.2rem; }
.state { white-space:pre-wrap; overflow-wrap:anywhere; margin:0; max-height:24rem;
  overflow:auto; font:13px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.type { font-weight:400; color:var(--muted); font-size:.85rem; }
.instructions { margin:.2rem 0 .7rem; }
.criterion { color:var(--muted); font-size:.85rem; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:.35rem .4rem; border-bottom:1px solid var(--line);
  vertical-align:top; }
th { font-size:.8rem; color:var(--muted); font-weight:600; }
.prob { white-space:nowrap; font-variant-numeric:tabular-nums; }
.bar { display:inline-block; width:3.5rem; height:.5rem; margin-right:.4rem;
  background:var(--line); border-radius:3px; vertical-align:middle; overflow:hidden; }
.bar span { display:block; height:100%; background:var(--accent); }
.muted { color:var(--muted); }
.final { white-space:nowrap; }
.final input { width:5.5rem; padding:.25rem .35rem; font:inherit; color:var(--ink);
  background:var(--bg); border:1px solid var(--line); border-radius:6px; }
.onehot, button, .button { font:inherit; font-size:.85rem; padding:.3rem .7rem;
  border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--ink);
  cursor:pointer; text-decoration:none; }
.onehot { margin-left:.3rem; padding:.2rem .45rem; font-size:.75rem; }
.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.rationales { font-size:.88rem; color:var(--muted); margin-top:.5rem; }
.rationales p { margin:.25rem 0; }
details summary { cursor:pointer; color:var(--accent); margin-bottom:.4rem; }
textarea { width:100%; font:inherit; color:var(--ink); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:.4rem; }
.actions { display:flex; gap:.6rem; justify-content:flex-end; margin-top:.7rem; }
.notice { padding:.6rem .8rem; border-radius:8px; margin:0 0 1rem; }
.error { background:color-mix(in srgb, var(--warn) 15%, transparent); color:var(--warn); }
.saved { background:color-mix(in srgb, var(--ok) 15%, transparent); color:var(--ok); }
.tabs { display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:1rem; }
.tab { padding:.35rem .8rem; border-radius:999px; text-decoration:none;
  border:1px solid var(--line); color:var(--ink); background:var(--card); }
.tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.list { background:var(--card); border:1px solid var(--line); border-radius:10px; }
.preview { color:var(--muted); overflow-wrap:anywhere; }
.pill { font-size:.75rem; padding:.1rem .5rem; border-radius:999px; border:1px solid var(--line);
  white-space:nowrap; }
.pill.todo { color:var(--warn); border-color:var(--warn); }
.pill.done { color:var(--ok); border-color:var(--ok); }
.crumbs { color:var(--muted); }
.empty { color:var(--muted); }
@media (max-width:640px) { .dist th:nth-child(4), .dist td:nth-child(4) { display:none; }
  .bar { display:none; } .list th:nth-child(4), .list td:nth-child(4) { display:none; } }
"""
