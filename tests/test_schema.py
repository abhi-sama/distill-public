from pathlib import Path

import pytest
from pydantic import ValidationError

from distill.schema import DecisionSpec, NoulQuestion, load_decision


def test_worked_example_validates() -> None:
    example = Path(__file__).parents[1] / "examples" / "decision.yaml"

    decision = load_decision(example)

    assert set(decision.root) == {"escalate", "urgency", "category", "manipulation"}
    assert isinstance(decision.root["escalate"], NoulQuestion)


def test_choice_accepts_laya_compatible_list_criteria() -> None:
    decision = DecisionSpec.model_validate(
        {
            "route": {
                "type": "choice",
                "instructions": "Where should `ticket` go?",
                "criteria": ["billing", "support"],
            }
        }
    )

    assert decision.root["route"].type == "choice"


@pytest.mark.parametrize(
    "question",
    [
        ["false", "true"],
        {"type": "yesno", "instructions": "Should this pass?"},
        {"type": "score", "instructions": "How urgent is it?"},
        {"type": "noul", "instructions": "Should this pass?", "labels": ["no", "yes"]},
        {
            "type": "choice",
            "instructions": "Choose a route.",
            "criteria": ["billing"],
            "labels": {"false": "no", "true": "yes"},
        },
    ],
    ids=[
        "bare-list-of-labels",
        "yesno-type",
        "score-without-criteria",
        "noul-label-list",
        "labels-on-choice",
    ],
)
def test_rejects_invalid_laya_question_shapes(question: object) -> None:
    with pytest.raises(ValidationError):
        DecisionSpec.model_validate({"decision": question})


def test_rejects_noul_criteria_with_non_boolean_key() -> None:
    with pytest.raises(ValidationError, match="criteria"):
        DecisionSpec.model_validate(
            {
                "decision": {
                    "type": "noul",
                    "instructions": "Should this pass?",
                    "criteria": {"maybe": "unclear"},
                }
            }
        )


def test_noul_allows_omitted_criteria_and_valid_custom_labels() -> None:
    decision = DecisionSpec.model_validate(
        {
            "review": {
                "type": "noul",
                "instructions": "Does this need review?",
                "labels": {"false": "no review", "true": "review"},
            }
        }
    )

    assert decision.root["review"].labels == {"false": "no review", "true": "review"}
