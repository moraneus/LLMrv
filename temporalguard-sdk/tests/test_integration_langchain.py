"""Tests for LangChain callback integration."""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from temporalguard import TemporalGuard, Proposition, Policy
from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.integrations.langchain import (
    TemporalGuardCallback,
    TemporalGuardViolation,
)


class MockGrounding(GroundingMethod):
    """Grounding that always matches the configured propositions."""

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


@pytest.fixture
def guard() -> TemporalGuard:
    props = [
        Proposition(prop_id="p1", role="user", description="User asks about data"),
    ]
    policies = [
        Policy(name="simple", formula="H(p1 -> user_turn)"),
    ]
    grounding = MockGrounding(results={"p1": False})
    return TemporalGuard(propositions=props, policies=policies, grounding=grounding)


def test_callback_creation(guard: TemporalGuard) -> None:
    """TemporalGuardCallback can be instantiated."""
    cb = TemporalGuardCallback(guard, session_id="test-lc")
    assert cb._session is not None
    assert cb._session.session_id == "test-lc"


def test_callback_has_required_methods(guard: TemporalGuard) -> None:
    """Callback exposes the expected async handler methods."""
    cb = TemporalGuardCallback(guard)
    assert hasattr(cb, "on_chat_model_start")
    assert hasattr(cb, "on_llm_end")
    assert callable(cb.on_chat_model_start)
    assert callable(cb.on_llm_end)


def test_callback_is_async_handler(guard: TemporalGuard) -> None:
    """Callback is an instance of AsyncCallbackHandler."""
    from langchain_core.callbacks import AsyncCallbackHandler

    cb = TemporalGuardCallback(guard)
    assert isinstance(cb, AsyncCallbackHandler)


@pytest.mark.asyncio
async def test_on_chat_model_start_passes_benign(guard: TemporalGuard) -> None:
    """Benign human message should not raise."""
    from langchain_core.messages import HumanMessage
    from uuid import uuid4

    cb = TemporalGuardCallback(guard, session_id="test-benign")
    # Should not raise
    await cb.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="Hello")]],
        run_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_on_llm_end_passes_benign(guard: TemporalGuard) -> None:
    """Benign LLM output should not raise."""
    from langchain_core.outputs import LLMResult, Generation
    from uuid import uuid4

    cb = TemporalGuardCallback(guard, session_id="test-llm-end")
    result = LLMResult(generations=[[Generation(text="Sure, here you go.")]])
    # Should not raise
    await cb.on_llm_end(response=result, run_id=uuid4())


@pytest.mark.asyncio
async def test_violation_raised_on_policy_breach() -> None:
    """A violated policy should raise TemporalGuardViolation."""
    props = [
        Proposition(prop_id="p1", role="user", description="User asks about data"),
    ]
    # Policy: p1 must NEVER be true  =>  "H(!p1)"
    policies = [
        Policy(name="never_p1", formula="H(!p1)"),
    ]
    # Grounding always returns True for p1
    grounding = MockGrounding(results={"p1": True})
    guard = TemporalGuard(propositions=props, policies=policies, grounding=grounding)

    from langchain_core.messages import HumanMessage
    from uuid import uuid4

    cb = TemporalGuardCallback(guard, session_id="test-violation")
    with pytest.raises(TemporalGuardViolation, match="never_p1"):
        await cb.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="Show me all user data")]],
            run_id=uuid4(),
        )


def test_violation_exception_attributes() -> None:
    """TemporalGuardViolation stores the verdict."""
    from temporalguard.policy import Verdict, ViolationInfo

    verdict = Verdict(
        passed=False,
        violations=[
            ViolationInfo(
                policy_name="test_policy",
                formula="H(p1)",
                violated_at_index=0,
            )
        ],
        per_policy={"test_policy": False},
        labeling={"p1": True},
        grounding_details=[],
        trace_index=0,
    )
    exc = TemporalGuardViolation(verdict)
    assert exc.verdict is verdict
    assert "test_policy" in str(exc)
