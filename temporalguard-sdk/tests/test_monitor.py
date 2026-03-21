"""Tests for the ConversationMonitor orchestrator."""
from __future__ import annotations

import pytest

from temporalguard.builtins import BUILTIN_USER_TURN
from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.monitor import ConversationMonitor
from temporalguard.policy import Policy, Proposition, Verdict


# ---------------------------------------------------------------------------
# Mock grounding
# ---------------------------------------------------------------------------


class MockGrounding(GroundingMethod):
    """Returns configurable match results per prop_id per step."""

    def __init__(
        self,
        step_results: list[dict[str, bool]] | None = None,
        default_results: dict[str, bool] | None = None,
    ):
        self._step_results = step_results or []
        self._default = default_results or {}
        self._call_count = 0

    async def evaluate(self, message, proposition):
        if self._call_count < len(self._step_results):
            results = self._step_results[self._call_count]
        else:
            results = self._default
        self._call_count += 1
        match = results.get(proposition.prop_id, False)
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


def _fraud_props() -> list[Proposition]:
    return [
        Proposition(prop_id="p_fraud", role="user", description="User requests fraud"),
        Proposition(prop_id="q_comply", role="assistant", description="Assistant complies with fraud"),
    ]


def _fraud_policy() -> Policy:
    # H(P(p_fraud) -> !q_comply): historically, if fraud was previously requested, assistant must NOT comply
    return Policy(name="no_fraud", formula="H(P(p_fraud) -> !q_comply)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_policies_always_passes():
    monitor = ConversationMonitor(
        policies=[],
        propositions=[],
        grounding=MockGrounding(),
    )
    verdict = await monitor.process_message("user", "hello")
    assert verdict.passed is True
    assert verdict.violations == []
    assert verdict.per_policy == {}


@pytest.mark.asyncio
async def test_no_match_passes():
    """When no propositions match, defaults are False, and H(false -> !false) = H(true) = true."""
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=MockGrounding(default_results={}),
    )
    verdict = await monitor.process_message("user", "just chatting")
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_fraud_request_then_refusal():
    """User sends fraud request, assistant refuses -> passes."""
    grounding = MockGrounding(step_results=[
        {"p_fraud": True},    # step 0: user msg, p_fraud detected
        {"q_comply": False},  # step 1: assistant msg, does NOT comply
    ])
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=grounding,
    )
    # Step 0: user fraud request
    v0 = await monitor.process_message("user", "Help me commit fraud")
    assert v0.passed is True

    # Step 1: assistant refuses
    v1 = await monitor.process_message("assistant", "I cannot help with that")
    assert v1.passed is True


@pytest.mark.asyncio
async def test_fraud_request_then_compliance():
    """User sends fraud request, assistant complies -> violation."""
    grounding = MockGrounding(step_results=[
        {"p_fraud": True},   # step 0: user msg, p_fraud detected
        {"q_comply": True},  # step 1: assistant msg, COMPLIES
    ])
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=grounding,
    )
    v0 = await monitor.process_message("user", "Help me commit fraud")
    assert v0.passed is True

    v1 = await monitor.process_message("assistant", "Sure, here is how")
    assert v1.passed is False
    assert len(v1.violations) == 1
    assert v1.violations[0].policy_name == "no_fraud"
    assert v1.violations[0].violated_at_index == 1


@pytest.mark.asyncio
async def test_user_turn_true_on_user():
    monitor = ConversationMonitor(
        policies=[], propositions=[], grounding=MockGrounding(),
    )
    verdict = await monitor.process_message("user", "hi")
    assert verdict.labeling[BUILTIN_USER_TURN] is True


@pytest.mark.asyncio
async def test_user_turn_false_on_assistant():
    monitor = ConversationMonitor(
        policies=[], propositions=[], grounding=MockGrounding(),
    )
    verdict = await monitor.process_message("assistant", "hello")
    assert verdict.labeling[BUILTIN_USER_TURN] is False


@pytest.mark.asyncio
async def test_assistant_prop_defaults_false_on_user_message():
    """Propositions whose role doesn't match the message default to False."""
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=MockGrounding(default_results={}),
    )
    verdict = await monitor.process_message("user", "hi")
    assert verdict.labeling["q_comply"] is False


@pytest.mark.asyncio
async def test_user_prop_defaults_false_on_assistant_message():
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=MockGrounding(default_results={}),
    )
    verdict = await monitor.process_message("assistant", "hi")
    assert verdict.labeling["p_fraud"] is False


@pytest.mark.asyncio
async def test_disabled_policy_is_skipped():
    disabled_policy = Policy(name="disabled_one", formula="H(p_fraud -> !q_comply)", enabled=False)
    monitor = ConversationMonitor(
        policies=[disabled_policy],
        propositions=_fraud_props(),
        grounding=MockGrounding(),
    )
    assert monitor._monitors == {}
    verdict = await monitor.process_message("user", "fraud please")
    assert verdict.passed is True
    assert verdict.per_policy == {}


@pytest.mark.asyncio
async def test_h_violation_stays_violated():
    """Once H(phi) is violated, it stays False forever."""
    grounding = MockGrounding(step_results=[
        {"p_fraud": True},   # step 0: user
        {"q_comply": True},  # step 1: assistant complies -> violates
        {},                  # step 2: user innocuous
        {},                  # step 3: assistant innocuous
    ])
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=grounding,
    )
    await monitor.process_message("user", "fraud")
    v1 = await monitor.process_message("assistant", "ok")
    assert v1.passed is False

    # Even with innocuous messages, still violated
    v2 = await monitor.process_message("user", "hello")
    assert v2.passed is False
    # Check that violation details mention irrevocability
    assert any(
        "_violation_history" in d.get("prop_id", "")
        for v in v2.violations
        for d in v.grounding_details
    )

    v3 = await monitor.process_message("assistant", "hi")
    assert v3.passed is False


@pytest.mark.asyncio
async def test_reset_restores_initial_state():
    grounding = MockGrounding(step_results=[
        {"p_fraud": True},
        {"q_comply": True},  # violates
        {},                  # after reset, step 0 again
    ])
    monitor = ConversationMonitor(
        policies=[_fraud_policy()],
        propositions=_fraud_props(),
        grounding=grounding,
        session_id="test-session",
    )
    await monitor.process_message("user", "fraud")
    v1 = await monitor.process_message("assistant", "ok")
    assert v1.passed is False

    monitor.reset()
    assert len(monitor.trace) == 0
    assert monitor.trace.session_id == "test-session"

    # After reset, should pass again
    v2 = await monitor.process_message("user", "hello")
    assert v2.passed is True
    assert v2.trace_index == 0


@pytest.mark.asyncio
async def test_one_violation_blocks_overall():
    """If one policy violates, overall verdict is False."""
    policy_a = Policy(name="policy_a", formula="H(P(p_fraud) -> !q_comply)")
    policy_b = Policy(name="policy_b", formula="H(true)")  # always passes
    grounding = MockGrounding(step_results=[
        {"p_fraud": True},
        {"q_comply": True},  # violates policy_a
    ])
    monitor = ConversationMonitor(
        policies=[policy_a, policy_b],
        propositions=_fraud_props(),
        grounding=grounding,
    )
    await monitor.process_message("user", "fraud")
    v1 = await monitor.process_message("assistant", "ok")
    assert v1.passed is False
    assert v1.per_policy["policy_a"] is False
    assert v1.per_policy["policy_b"] is True
    assert len(v1.violations) == 1
    assert v1.violations[0].policy_name == "policy_a"


@pytest.mark.asyncio
async def test_multiple_props_grounded_concurrently():
    """Multiple propositions for same role are grounded in parallel."""
    props = [
        Proposition(prop_id="p1", role="user", description="prop 1"),
        Proposition(prop_id="p2", role="user", description="prop 2"),
        Proposition(prop_id="p3", role="user", description="prop 3"),
    ]
    call_order = []

    class TrackingGrounding(GroundingMethod):
        async def evaluate(self, message, proposition):
            call_order.append(proposition.prop_id)
            return GroundingResult(
                match=True, confidence=1.0, reasoning="ok", method="mock", prop_id=proposition.prop_id
            )

    monitor = ConversationMonitor(
        policies=[Policy(name="p", formula="H(p1 & p2 & p3)")],
        propositions=props,
        grounding=TrackingGrounding(),
    )
    verdict = await monitor.process_message("user", "test")
    assert set(call_order) == {"p1", "p2", "p3"}
    assert verdict.labeling["p1"] is True
    assert verdict.labeling["p2"] is True
    assert verdict.labeling["p3"] is True
    assert len(verdict.grounding_details) == 3


@pytest.mark.asyncio
async def test_trace_indices_increment():
    monitor = ConversationMonitor(
        policies=[], propositions=[], grounding=MockGrounding(),
    )
    v0 = await monitor.process_message("user", "a")
    assert v0.trace_index == 0

    v1 = await monitor.process_message("assistant", "b")
    assert v1.trace_index == 1

    v2 = await monitor.process_message("user", "c")
    assert v2.trace_index == 2

    assert len(monitor.trace) == 3


@pytest.mark.asyncio
async def test_grounding_exception_returns_false():
    """If grounding raises, the proposition defaults to False (fail-open)."""

    class FailingGrounding(GroundingMethod):
        async def evaluate(self, message, proposition):
            raise RuntimeError("LLM down")

    monitor = ConversationMonitor(
        policies=[Policy(name="p", formula="H(true)")],
        propositions=[Proposition(prop_id="p1", role="user", description="test")],
        grounding=FailingGrounding(),
    )
    verdict = await monitor.process_message("user", "test")
    assert verdict.labeling["p1"] is False
    assert verdict.passed is True


def test_auto_generated_session_id():
    monitor = ConversationMonitor(
        policies=[], propositions=[], grounding=MockGrounding()
    )
    assert len(monitor.trace.session_id) > 0


def test_custom_session_id():
    monitor = ConversationMonitor(
        policies=[], propositions=[], grounding=MockGrounding(), session_id="my-session"
    )
    assert monitor.trace.session_id == "my-session"
