"""Tests for temporalguard.policy dataclasses."""

from __future__ import annotations

import pytest

from temporalguard.policy import Policy, Proposition, Verdict, ViolationInfo


# --- Proposition ---


class TestProposition:
    def test_basic_creation(self):
        p = Proposition(prop_id="grounded", role="assistant", description="Is grounded")
        assert p.prop_id == "grounded"
        assert p.role == "assistant"
        assert p.description == "Is grounded"
        assert p.few_shot_positive is None
        assert p.few_shot_negative is None

    def test_with_few_shots(self):
        p = Proposition(
            prop_id="polite",
            role="user",
            description="Is polite",
            few_shot_positive="Please help me.",
            few_shot_negative="Do it now!",
        )
        assert p.few_shot_positive == "Please help me."
        assert p.few_shot_negative == "Do it now!"


# --- Policy ---


class TestPolicy:
    def test_basic_creation(self):
        pol = Policy(name="p1", formula="H(grounded)")
        assert pol.name == "p1"
        assert pol.formula == "H(grounded)"
        assert pol.enabled is True

    def test_auto_extraction_from_formula(self):
        pol = Policy(name="p1", formula="H(grounded & polite)")
        assert sorted(pol.propositions) == ["grounded", "polite"]

    def test_explicit_propositions_not_overridden(self):
        pol = Policy(name="p1", formula="H(grounded & polite)", propositions=["custom"])
        assert pol.propositions == ["custom"]

    def test_builtins_excluded(self):
        pol = Policy(name="p1", formula="H(user_turn -> grounded) & true")
        assert "user_turn" not in pol.propositions
        assert "true" not in pol.propositions
        assert "false" not in pol.propositions
        assert "grounded" in pol.propositions

    def test_disabled_policy(self):
        pol = Policy(name="p1", formula="H(a)", enabled=False)
        assert pol.enabled is False


# --- ViolationInfo ---


class TestViolationInfo:
    def test_creation(self):
        v = ViolationInfo(
            policy_name="p1",
            formula="H(grounded)",
            violated_at_index=3,
            labeling={"grounded": False},
            grounding_details={"grounded": "not grounded"},
        )
        assert v.policy_name == "p1"
        assert v.formula == "H(grounded)"
        assert v.violated_at_index == 3
        assert v.labeling == {"grounded": False}
        assert v.grounding_details == {"grounded": "not grounded"}


# --- Verdict ---


class TestVerdict:
    def test_passed(self):
        v = Verdict(passed=True, violations=[], per_policy={})
        assert v.passed is True
        assert v.violation is False

    def test_failed_with_violations(self):
        vi = ViolationInfo(
            policy_name="p1",
            formula="H(grounded)",
            violated_at_index=2,
            labeling={"grounded": False},
        )
        v = Verdict(passed=False, violations=[vi], per_policy={"p1": False})
        assert v.passed is False
        assert v.violation is True

    def test_defaults(self):
        v = Verdict(passed=True)
        assert v.violations == []
        assert v.per_policy == {}
        assert v.labeling is None
        assert v.grounding_details is None
        assert v.trace_index is None
