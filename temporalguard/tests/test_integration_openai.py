"""Tests for the OpenAI SDK wrapper integration."""
from __future__ import annotations

import pytest

pytest.importorskip("openai")

from unittest.mock import AsyncMock, MagicMock

from temporalguard import TemporalGuard, Proposition, Policy
from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.integrations.openai import guarded_chat, GuardedChatViolation


# ---------------------------------------------------------------------------
# Mock grounding
# ---------------------------------------------------------------------------


class MockGrounding(GroundingMethod):
    """Grounding that returns configurable static results."""

    def __init__(self, results: dict[str, bool] | None = None) -> None:
        self._results = results or {}

    async def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        match = self._results.get(proposition.prop_id, False)
        return GroundingResult(
            match=match,
            confidence=1.0 if match else 0.0,
            reasoning="mock",
            method="mock",
            prop_id=proposition.prop_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(content: str) -> MagicMock:
    """Build a fake OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_client(response: MagicMock) -> MagicMock:
    """Build a fake async OpenAI client."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _make_guard(grounding_results: dict[str, bool] | None = None) -> TemporalGuard:
    """Build a TemporalGuard with a simple policy."""
    props = [
        Proposition(prop_id="harmful", role="user", description="User sends harmful content"),
        Proposition(prop_id="toxic", role="assistant", description="Assistant produces toxic output"),
    ]
    policies = [
        Policy(name="no_harm", formula="H(!harmful)"),
        Policy(name="no_toxic", formula="H(!toxic)"),
    ]
    grounding = MockGrounding(results=grounding_results or {})
    return TemporalGuard(propositions=props, policies=policies, grounding=grounding)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guarded_chat_passes_clean_conversation():
    """A benign user message and clean assistant response should pass."""
    guard = _make_guard({"harmful": False, "toxic": False})
    session = guard.session(session_id="test-clean")

    response = _make_openai_response("Hello! How can I help?")
    client = _make_client(response)

    result = await guarded_chat(
        session,
        client,
        messages=[{"role": "user", "content": "Hi there"}],
        model="gpt-4",
    )

    assert result is response
    client.chat.completions.create.assert_awaited_once_with(
        messages=[{"role": "user", "content": "Hi there"}],
        model="gpt-4",
    )


@pytest.mark.asyncio
async def test_guarded_chat_raises_on_user_violation():
    """A harmful user message should raise before calling the LLM."""
    guard = _make_guard({"harmful": True, "toxic": False})
    session = guard.session(session_id="test-user-violation")

    response = _make_openai_response("Sure thing!")
    client = _make_client(response)

    with pytest.raises(GuardedChatViolation) as exc_info:
        await guarded_chat(
            session,
            client,
            messages=[{"role": "user", "content": "Something harmful"}],
            model="gpt-4",
        )

    assert exc_info.value.phase == "user"
    assert "no_harm" in str(exc_info.value)
    # LLM should NOT have been called
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_guarded_chat_raises_on_assistant_violation():
    """A toxic assistant response should raise after the LLM call."""
    guard = _make_guard({"harmful": False, "toxic": True})
    session = guard.session(session_id="test-assistant-violation")

    response = _make_openai_response("Something toxic")
    client = _make_client(response)

    with pytest.raises(GuardedChatViolation) as exc_info:
        await guarded_chat(
            session,
            client,
            messages=[{"role": "user", "content": "Innocent question"}],
            model="gpt-4",
        )

    assert exc_info.value.phase == "assistant"
    assert "no_toxic" in str(exc_info.value)
    # LLM should have been called (violation is on the response side)
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_guarded_chat_skips_non_user_messages():
    """System messages should not be checked against user policies."""
    guard = _make_guard({"harmful": False, "toxic": False})
    session = guard.session(session_id="test-system-skip")

    response = _make_openai_response("Hello!")
    client = _make_client(response)

    result = await guarded_chat(
        session,
        client,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
        model="gpt-4",
    )

    assert result is response


@pytest.mark.asyncio
async def test_guarded_chat_violation_has_verdict():
    """The exception should carry the full Verdict object."""
    guard = _make_guard({"harmful": True})
    session = guard.session(session_id="test-verdict-access")

    response = _make_openai_response("OK")
    client = _make_client(response)

    with pytest.raises(GuardedChatViolation) as exc_info:
        await guarded_chat(
            session,
            client,
            messages=[{"role": "user", "content": "Bad input"}],
            model="gpt-4",
        )

    exc = exc_info.value
    assert exc.verdict is not None
    assert exc.verdict.passed is False
    assert len(exc.verdict.violations) >= 1


@pytest.mark.asyncio
async def test_guarded_chat_handles_empty_assistant_content():
    """An assistant choice with None content should not trigger a check."""
    guard = _make_guard({"harmful": False, "toxic": True})
    session = guard.session(session_id="test-empty-content")

    # Response with None content -- should not be checked
    message = MagicMock()
    message.content = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = _make_client(response)

    result = await guarded_chat(
        session,
        client,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
    )

    # Should pass because there is no text to check
    assert result is response
