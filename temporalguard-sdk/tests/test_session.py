"""Tests for Session class."""
from __future__ import annotations

import pytest

from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.guard import TemporalGuard
from temporalguard.policy import Policy, Proposition


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
def guard() -> TemporalGuard:
    props = [
        Proposition(prop_id="p1", role="user", description="User asks about data"),
    ]
    policies = [
        Policy(name="simple", formula="H(p1 -> user_turn)"),
    ]
    grounding = StaticGrounding(results={"p1": False})
    return TemporalGuard(propositions=props, policies=policies, grounding=grounding)


@pytest.mark.asyncio
async def test_benign_message_passes(guard):
    session = guard.session(session_id="test-benign")
    verdict = await session.check("user", "Hello there!")
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_verdict_has_labeling_with_user_turn(guard):
    session = guard.session(session_id="test-labeling")
    verdict = await session.check("user", "Hello")
    assert "user_turn" in verdict.labeling
    assert verdict.labeling["user_turn"] is True


@pytest.mark.asyncio
async def test_trace_index_increments(guard):
    session = guard.session(session_id="test-index")
    v1 = await session.check("user", "First message")
    v2 = await session.check("assistant", "Second message")
    assert v1.trace_index == 0
    assert v2.trace_index == 1


@pytest.mark.asyncio
async def test_session_trace_returns_messages(guard):
    session = guard.session(session_id="test-trace")
    await session.check("user", "Hello")
    await session.check("assistant", "Hi there")
    assert len(session.trace) == 2
    assert session.trace[0].role == "user"
    assert session.trace[0].text == "Hello"
    assert session.trace[1].role == "assistant"
    assert session.trace[1].text == "Hi there"


@pytest.mark.asyncio
async def test_reset_clears_trace(guard):
    session = guard.session(session_id="test-reset")
    await session.check("user", "Hello")
    assert len(session.trace) == 1
    session.reset()
    assert len(session.trace) == 0


@pytest.mark.asyncio
async def test_user_turn_true_on_user_false_on_assistant(guard):
    session = guard.session(session_id="test-user-turn")
    v_user = await session.check("user", "I am the user")
    assert v_user.labeling["user_turn"] is True

    v_assistant = await session.check("assistant", "I am the assistant")
    assert v_assistant.labeling["user_turn"] is False
