"""Tests for TemporalGuard class."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.guard import TemporalGuard
from temporalguard.policy import Policy, Proposition
from temporalguard.session import Session


class StaticGrounding(GroundingMethod):
    """Mock grounding that returns configurable static results."""

    def __init__(self, results: dict[str, bool] | None = None) -> None:
        self._results = results or {}

    async def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        match = self._results.get(proposition.prop_id, False)
        return GroundingResult(
            match=match,
            confidence=1.0 if match else 0.0,
            reasoning="static",
            method="mock",
            prop_id=proposition.prop_id,
        )


@pytest.fixture
def props() -> list[Proposition]:
    return [
        Proposition(prop_id="p1", role="user", description="User asks about data"),
    ]


@pytest.fixture
def policies() -> list[Policy]:
    return [
        Policy(name="test_policy", formula="H(p1 -> user_turn)"),
    ]


@pytest.fixture
def grounding() -> StaticGrounding:
    return StaticGrounding(results={"p1": False})


def test_basic_construction(props, policies, grounding):
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    assert guard._propositions == props
    assert guard._policies == policies
    assert guard._grounding is grounding


def test_session_creation_auto_id(props, policies, grounding):
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    session = guard.session()
    assert isinstance(session, Session)
    # Auto-generated ID should be a valid UUID
    uuid.UUID(session.session_id)


def test_session_creation_custom_id(props, policies, grounding):
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    session = guard.session(session_id="my-session-42")
    assert session.session_id == "my-session-42"


def test_multiple_sessions_are_independent(props, policies, grounding):
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    s1 = guard.session(session_id="s1")
    s2 = guard.session(session_id="s2")
    assert s1.session_id != s2.session_id
    assert s1._monitor is not s2._monitor


def test_empty_policies_allowed(props, grounding):
    guard = TemporalGuard(propositions=props, policies=[], grounding=grounding)
    session = guard.session()
    assert isinstance(session, Session)


def test_from_yaml(grounding):
    yaml_content = """\
propositions:
  - id: ask_data
    role: user
    description: User asks about personal data
policies:
  - name: data_policy
    formula: "H(ask_data -> user_turn)"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        guard = TemporalGuard.from_yaml(tmp_path, grounding=grounding)
        assert len(guard._propositions) == 1
        assert guard._propositions[0].prop_id == "ask_data"
        assert len(guard._policies) == 1
        assert guard._policies[0].name == "data_policy"
        session = guard.session()
        assert isinstance(session, Session)
    finally:
        Path(tmp_path).unlink()


# ---------------------------------------------------------------------------
# Auto-generation of few-shot examples
# ---------------------------------------------------------------------------


def test_auto_generate_skipped_for_non_llm_grounding(props, policies, grounding):
    """StaticGrounding has no _call_llm — auto-generation should be silently skipped."""
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    # Propositions should remain without few-shot examples
    assert guard._propositions[0].few_shot_positive == []
    assert guard._propositions[0].few_shot_negative == []


def test_auto_generate_disabled(props, policies, grounding):
    """With auto_generate_few_shots=False, no generation should happen."""
    guard = TemporalGuard(
        propositions=props, policies=policies, grounding=grounding,
        auto_generate_few_shots=False,
    )
    assert guard._propositions[0].few_shot_positive == []


def test_auto_generate_skips_when_examples_provided(policies, grounding):
    """Propositions with existing few-shot examples are left untouched."""
    props = [
        Proposition("p1", "user", "desc",
                    few_shot_positive=["my example"],
                    few_shot_negative=["my neg"]),
    ]
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)
    assert guard._propositions[0].few_shot_positive == ["my example"]
    assert guard._propositions[0].few_shot_negative == ["my neg"]


def test_generate_few_shots_raises_for_non_llm_grounding(props, policies, grounding):
    """Explicit generate_few_shots() raises TypeError for non-LLM grounding."""
    guard = TemporalGuard(
        propositions=props, policies=policies, grounding=grounding,
        auto_generate_few_shots=False,
    )
    with pytest.raises(TypeError, match="requires LLMGrounding"):
        guard.generate_few_shots()
