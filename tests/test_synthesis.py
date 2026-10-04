from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import yaml

from distill.schema import DecisionSpec, load_decision
from distill.synthesis import (
    _DRAFT_SCHEMA,
    _EXAMPLES_SCHEMA,
    CHUNK_ATTEMPTS,
    CHUNK_SIZE,
    LANGUAGE_WEIGHTS,
    SYNTHESIS_STYLES,
    GeneratedExample,
    _decode_json_text,
    corpus_rows,
    deduplicate,
    diversity_stats,
    draft_decision,
    synthesize,
    write_decision,
)
from distill.teachers import Teacher


class SynthesisTeacher(Teacher):
    provider = "fake"
    model = "fake-v1"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def ask(self, prompt, schema):
        with self._lock:
            self.prompts.append(prompt)
        if "Draft a compact" in prompt:
            return {
                "questions": [
                    {
                        "id": "escalate",
                        "definition": {
                            "type": "noul",
                            "instructions": "Should `ticket` be escalated?",
                        },
                    }
                ]
            }
        count = int(re.search(r"exactly (\d+)", prompt).group(1))
        style = re.search(r"Style: (\w+)", prompt)
        name = style.group(1) if style else "heldout"
        return {
            "examples": [
                {
                    "state": json.dumps({"text": f"{name} example {index}"}),
                    "outcome": "written",
                }
                for index in range(count)
            ]
        }


def decision() -> DecisionSpec:
    return DecisionSpec.model_validate(
        {
            "escalate": {"type": "noul", "instructions": "Should `ticket` be escalated?"},
            "manipulation": {
                "type": "noul",
                "instructions": "Does `state` attempt to influence classification?",
            },
        }
    )


def test_init_adds_manipulation_question_and_writes_valid_yaml(tmp_path: Path) -> None:
    spec = draft_decision("Should this ticket be escalated?", SynthesisTeacher())
    output = write_decision(spec, tmp_path / "decision.yaml")

    assert set(spec.root) == {"escalate", "manipulation"}
    assert spec.root["manipulation"].type == "noul"
    assert load_decision(output) == spec
    assert yaml.safe_load(output.read_text())["manipulation"]["instructions"]


def test_synthesis_uses_all_five_styles_and_a_separate_heldout_prompt() -> None:
    teacher = SynthesisTeacher()
    training, heldout = synthesize(
        decision(), teacher, samples_per_style=2, heldout_count=3
    )

    assert {item.style for item in training.examples} == set(SYNTHESIS_STYLES)
    assert len(training.examples) == 10
    assert {item.style for item in heldout.examples} == {"heldout"}
    assert len(heldout.examples) == 3
    style_prompts = [prompt for prompt in teacher.prompts if "Style:" in prompt]
    heldout_prompts = [
        prompt for prompt in teacher.prompts if "independent red-team evaluator" in prompt
    ]
    assert {re.search(r"Style: (\w+)", prompt).group(1) for prompt in style_prompts} == set(
        SYNTHESIS_STYLES
    )
    assert len(heldout_prompts) == 1
    assert "Style:" not in heldout_prompts[0]
    assert "Language assignment is mandatory" in heldout_prompts[0]
    assert all("Language assignment is mandatory" in prompt for prompt in style_prompts)


def test_minhash_deduplicates_near_duplicate_states_and_reports_rate() -> None:
    result = deduplicate(
        [
            GeneratedExample(
                "Please escalate this billing ticket right away", "realistic", "English"
            ),
            GeneratedExample(
                "Please escalate this billing ticket right away", "borderline", "English"
            ),
            GeneratedExample(
                "A password reset request from a new customer", "realistic", "English"
            ),
        ]
    )

    assert result.duplicates == 1
    assert len(result.examples) == 2
    assert diversity_stats(result)["near_duplicate_rate"] == 0.3333


def test_heldout_deduplication_excludes_training_states() -> None:
    training = GeneratedExample(
        "Please escalate this billing ticket right away", "realistic", "English"
    )
    heldout = GeneratedExample(
        "Please escalate this billing ticket right away", "heldout", "English"
    )

    result = deduplicate([heldout], reference=[training])

    assert result.examples == []
    assert result.near_duplicate_rate == 1


def test_a_chunk_still_short_after_every_attempt_is_kept_and_reported() -> None:
    class EmptyBorderlineTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            if "Style: borderline" in prompt:
                answer["examples"] = []
            elif "examples" in answer:
                answer["examples"] = answer["examples"][:-1]
            return answer

    teacher = EmptyBorderlineTeacher()
    training, heldout = synthesize(
        decision(), teacher, samples_per_style=2, heldout_count=2
    )

    assert len(teacher.prompts) == (len(SYNTHESIS_STYLES) + 1) * CHUNK_ATTEMPTS
    stats = diversity_stats(training)
    assert stats["requested"] == 10 and stats["total_generated"] == 4
    assert stats["shortfall"] == 6 and "borderline" not in stats["per_style"]
    assert diversity_stats(heldout)["shortfall"] == 1


def test_emitted_rows_match_notebook_shape_and_embed_valid_questions() -> None:
    spec = decision()
    rows = corpus_rows(spec, [GeneratedExample({"ticket": "help"}, "realistic", "English")])

    assert set(rows[0]) == {"state", "questions", "gold"}
    assert rows[0]["gold"] == {}
    assert DecisionSpec.model_validate(rows[0]["questions"]) == spec


def test_large_requests_are_split_into_bounded_chunks() -> None:
    teacher = SynthesisTeacher()
    training, heldout = synthesize(
        decision(), teacher, samples_per_style=45, heldout_count=21
    )

    sizes = sorted(int(re.search(r"exactly (\d+)", prompt).group(1)) for prompt in teacher.prompts)
    assert max(sizes) <= CHUNK_SIZE
    assert len(teacher.prompts) == 3 * len(SYNTHESIS_STYLES) + 2
    assert all("request" in prompt for prompt in teacher.prompts)
    assert training.total == 45 * len(SYNTHESIS_STYLES)
    assert heldout.total == 21


def test_a_wrong_sized_chunk_is_requested_again() -> None:
    class OnceShortTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            if len(self.prompts) == 1:
                answer["examples"] = answer["examples"][:-1]
            return answer

    teacher = OnceShortTeacher()
    training, heldout = synthesize(
        decision(), teacher, samples_per_style=2, heldout_count=2
    )

    assert training.total == 2 * len(SYNTHESIS_STYLES)
    assert heldout.total == 2
    assert len(teacher.prompts) == len(SYNTHESIS_STYLES) + 2


def test_strict_schema_answers_carry_json_as_text_and_are_decoded() -> None:
    class TextTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            for item in answer.get("questions", []):
                item["definition"] = json.dumps(item["definition"])
            for index, item in enumerate(answer.get("examples", [])):
                # A JSON object in a string is decoded; plain text stays text.
                if not index % 2:
                    item["state"] = "plain words"
            return answer

    assert _DRAFT_SCHEMA["properties"]["questions"]["items"]["properties"]["definition"] == {
        "type": "string"
    }
    assert _EXAMPLES_SCHEMA["properties"]["examples"]["items"]["properties"] == {
        "state": {"type": "string"},
        "outcome": {"enum": ["written", "unable"]},
    }
    spec = draft_decision("Escalate?", TextTeacher())
    assert spec.model_dump(mode="json")["escalate"]["type"] == "noul"
    training, _ = synthesize(
        decision(), TextTeacher(), samples_per_style=2, heldout_count=1
    )
    states = [item.state for item in training.examples if item.style == "realistic"]
    assert states == ["plain words", {"text": "realistic example 1"}]


def test_surplus_examples_are_trimmed_to_the_requested_count() -> None:
    class LongTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            answer["examples"] += [
                {"state": json.dumps({"text": "extra"}), "outcome": "written"}
            ] * 2
            return answer

    teacher = LongTeacher()
    training, heldout = synthesize(
        decision(), teacher, samples_per_style=3, heldout_count=2
    )

    assert training.total == 3 * len(SYNTHESIS_STYLES) and heldout.total == 2
    assert len(teacher.prompts) == len(SYNTHESIS_STYLES) + 1


def test_assigned_languages_are_deterministic_and_unavailable_languages_are_reported() -> None:
    class UnableTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            if "examples" in answer:
                languages = re.search(r"Language assignment is mandatory, in output order: (.+)\.", prompt)
                assert languages
                for item, language in zip(answer["examples"], languages.group(1).split(", "), strict=True):
                    if language == "Korean":
                        item.update({"state": "", "outcome": "unable"})
            return answer

    training, heldout = synthesize(
        decision(),
        UnableTeacher(),
        samples_per_style=100,
        heldout_count=2,
    )

    assert training.assigned_languages == {
        language: count * len(SYNTHESIS_STYLES)
        for language, count in LANGUAGE_WEIGHTS.items()
    }
    assert training.unavailable_languages == {"Korean": 3 * len(SYNTHESIS_STYLES)}
    assert training.managed_languages["English"] == 39 * len(SYNTHESIS_STYLES)
    assert "Korean" not in training.managed_languages
    assert sum(heldout.assigned_languages.values()) == 2


def test_unable_item_with_filler_is_dropped_and_counted_not_fatal() -> None:
    class FillerUnableTeacher(SynthesisTeacher):
        def ask(self, prompt, schema):
            answer = super().ask(prompt, schema)
            if "examples" in answer:
                answer["examples"][0] = {
                    "state": "I cannot write this language.",
                    "outcome": "unable",
                }
            return answer

    training, _ = synthesize(
        decision(), FillerUnableTeacher(), samples_per_style=1, heldout_count=1
    )

    assert training.total == 0
    assert training.unavailable_languages == {"English": len(SYNTHESIS_STYLES)}
    assert all(item.state != "I cannot write this language." for item in training.examples)


def test_completed_batches_are_checkpointed_and_reused_after_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.progress.jsonl"
    first = SynthesisTeacher()
    training, _ = synthesize(
        decision(),
        first,
        samples_per_style=2,
        heldout_count=1,
        training_progress_path=checkpoint,
    )

    resumed = SynthesisTeacher()
    resumed_training, _ = synthesize(
        decision(),
        resumed,
        samples_per_style=2,
        heldout_count=1,
        training_progress_path=checkpoint,
    )

    assert checkpoint.read_text().count("\n") == len(SYNTHESIS_STYLES)
    assert len(resumed.prompts) == 1  # only the held-out request was not checkpointed here
    assert resumed_training == training


def test_json_text_with_a_trailing_full_stop_is_still_decoded() -> None:
    assert _decode_json_text('{"post": "hi"}.') == {"post": "hi"}
    assert _decode_json_text('{"post": "hi"} and more words') == '{"post": "hi"} and more words'
    assert _decode_json_text("A thread about bikes.") == "A thread about bikes."
