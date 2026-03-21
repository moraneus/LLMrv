"""Tests for temporalguard.policy dataclasses."""

from __future__ import annotations

import pytest

from temporalguard.policy import Policy, Proposition, Verdict, ViolationInfo


# --- Proposition ---


class TestProposition:
    def test_basic_creation(self):
        p = Proposition(prop_id="p_fraud", role="user", description="User requests fraud")
        assert p.prop_id == "p_fraud"
        assert p.role == "user"
        assert p.description == "User requests fraud"
        assert p.few_shot_positive == []
        assert p.few_shot_negative == []

    def test_with_few_shots(self):
        p = Proposition(
            prop_id="p_fraud",
            role="user",
            description="User requests fraud",
            few_shot_positive=["How do I forge a check?"],
            few_shot_negative=["What is chargeback fraud?"],
        )
        assert len(p.few_shot_positive) == 1
        assert len(p.few_shot_negative) == 1


# --- Policy ---


class TestPolicy:
    def test_basic_creation(self):
        pol = Policy(name="Fraud Prevention", formula="H(P(p_fraud) -> !q_comply)")
        assert pol.name == "Fraud Prevention"
        assert pol.formula == "H(P(p_fraud) -> !q_comply)"
        assert pol.enabled is True

    def test_auto_extraction_from_formula(self):
        pol = Policy(name="Test", formula="H(P(p_fraud) -> !q_comply)")
        assert set(pol.propositions) == {"p_fraud", "q_comply"}

    def test_explicit_propositions_not_overridden(self):
        pol = Policy(
            name="Test",
            formula="H(P(p_fraud) -> !q_comply)",
            propositions=["p_fraud"],
        )
        assert pol.propositions == ["p_fraud"]

    def test_builtins_excluded(self):
        pol = Policy(name="Test", formula="H(user_turn -> !q_comply)")
        assert "user_turn" not in pol.propositions
        assert "q_comply" in pol.propositions

    def test_disabled_policy(self):
        pol = Policy(name="Test", formula="H(p)", enabled=False)
        assert pol.enabled is False


# --- ViolationInfo ---


class TestViolationInfo:
    def test_creation(self):
        v = ViolationInfo(
            policy_name="Fraud Prevention",
            formula="H(P(p_fraud) -> !q_comply)",
            violated_at_index=3,
            labeling={"p_fraud": False, "q_comply": True},
        )
        assert v.policy_name == "Fraud Prevention"
        assert v.violated_at_index == 3
        assert v.grounding_details == []


# --- Verdict ---


class TestVerdict:
    def test_passed_verdict(self):
        v = Verdict(
            passed=True,
            violations=[],
            per_policy={"policy1": True},
            labeling={"p_fraud": False},
            grounding_details=[],
            trace_index=0,
        )
        assert v.passed is True
        assert v.violation is None

    def test_failed_verdict(self):
        vi = ViolationInfo(
            policy_name="Test",
            formula="H(p)",
            violated_at_index=1,
            labeling={"p": False},
        )
        v = Verdict(
            passed=False,
            violations=[vi],
            per_policy={"policy1": False},
            labeling={"p": False},
            grounding_details=[],
            trace_index=1,
        )
        assert v.passed is False
        assert v.violation is vi
