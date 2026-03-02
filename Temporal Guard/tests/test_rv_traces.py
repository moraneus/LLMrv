"""
Runtime Verification — Comprehensive Trace Tests.

48 trace-level integration tests for the ptLTL runtime verification framework.
Each test specifies a temporal property (formula), a set of propositions, and a
trace of 10–20 events with the expected verdict at EVERY step.

These tests verify that the monitor correctly evaluates complex, multi-turn
conversation traces — the core functionality of TemporalGuard.
"""

import json

from backend.engine.ptltl import PtLTLMonitor, parse

# Helper


def run_trace(
    formula_str: str,
    trace: list[dict[str, bool]],
    expected: list[bool],
) -> PtLTLMonitor:
    """Parse formula, step through trace, assert verdict at each step.

    Args:
        formula_str: ptLTL formula string.
        trace: list of labelings, one per step. Each is {prop_id: bool}.
        expected: list of expected verdicts, one per step.

    Returns:
        The monitor after all steps (for serialization tests).
    """
    assert len(trace) == len(expected), "trace and expected must have same length"
    ast = parse(formula_str)
    monitor = PtLTLMonitor(ast)
    for i, (labeling, exp) in enumerate(zip(trace, expected, strict=True)):
        verdict = monitor.step(labeling)
        assert verdict == exp, f"Step {i}: expected {exp}, got {verdict}. Labeling: {labeling}"
    return monitor


# Test 1–2: Weapons Prohibition


class TestWeapons:
    """Weapons prohibition: H(p_weapon -> !q_comply)."""

    def test_persistent_attacker_all_refused(self):
        """Test 1: 5 weapon requests across 16 messages, all refused → ⊤."""
        F = "H(p_weapon -> !q_comply)"
        T = [
            {"p_weapon": False, "q_comply": False},  # 0 U benign
            {"p_weapon": False, "q_comply": False},  # 1 A
            {"p_weapon": True, "q_comply": False},  # 2 U weapon req
            {"p_weapon": False, "q_comply": False},  # 3 A refuses
            {"p_weapon": False, "q_comply": False},  # 4 U benign
            {"p_weapon": False, "q_comply": False},  # 5 A
            {"p_weapon": True, "q_comply": False},  # 6 U weapon req
            {"p_weapon": False, "q_comply": False},  # 7 A refuses
            {"p_weapon": False, "q_comply": False},  # 8 U benign
            {"p_weapon": False, "q_comply": False},  # 9 A
            {"p_weapon": True, "q_comply": False},  # 10 U weapon req
            {"p_weapon": False, "q_comply": False},  # 11 A refuses
            {"p_weapon": True, "q_comply": False},  # 12 U weapon req
            {"p_weapon": False, "q_comply": False},  # 13 A refuses
            {"p_weapon": True, "q_comply": False},  # 14 U weapon req
            {"p_weapon": False, "q_comply": False},  # 15 A refuses
        ]
        E = [True] * 16
        run_trace(F, T, E)

    def test_compliance_at_attempt_3(self):
        """Test 2: Assistant complies at 3rd weapon request (step 11) → ⊥ irrevocable."""
        F = "H(p_weapon -> !q_comply)"
        T = [
            {"p_weapon": False, "q_comply": False},  # 0
            {"p_weapon": False, "q_comply": False},  # 1
            {"p_weapon": True, "q_comply": False},  # 2
            {"p_weapon": False, "q_comply": False},  # 3
            {"p_weapon": False, "q_comply": False},  # 4
            {"p_weapon": False, "q_comply": False},  # 5
            {"p_weapon": True, "q_comply": False},  # 6
            {"p_weapon": False, "q_comply": False},  # 7
            {"p_weapon": False, "q_comply": False},  # 8
            {"p_weapon": False, "q_comply": False},  # 9
            {"p_weapon": True, "q_comply": True},  # 10 weapon req + COMPLY → violation
            {"p_weapon": False, "q_comply": False},  # 11
            {"p_weapon": False, "q_comply": False},  # 12
            {"p_weapon": False, "q_comply": False},  # 13
            {"p_weapon": False, "q_comply": False},  # 14
            {"p_weapon": False, "q_comply": False},  # 15
        ]
        E = [True] * 10 + [False] * 6
        run_trace(F, T, E)


# Test 3–5: Sensitive Data


class TestSensitiveData:
    """Sensitive data: H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)."""

    def test_ssn_handled_correctly(self):
        """Test 3: SSN shared, warned, never echoed → ⊤."""
        F = "H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)"
        T = [
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 0 U
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 1 A
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 2 U
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 3 A
            {"p_sensitive": True, "q_warn": False, "q_echo": False},  # 4 U SSN
            {"p_sensitive": False, "q_warn": True, "q_echo": False},  # 5 A warns
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 6 U
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 7 A
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 8 U
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 9 A
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 10 U
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 11 A
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_delayed_echo_caught(self):
        """Test 4: SSN at step 2, echo slips at step 11 → ⊥."""
        F = "H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)"
        T = [
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 0
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 1
            {"p_sensitive": True, "q_warn": False, "q_echo": False},  # 2 SSN
            {"p_sensitive": False, "q_warn": True, "q_echo": False},  # 3 warns
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 4
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 5
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 6
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 7
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 8
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 9
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 10
            {"p_sensitive": False, "q_warn": False, "q_echo": True},  # 11 ECHO
        ]
        E = [True] * 11 + [False]
        run_trace(F, T, E)

    def test_missing_warning(self):
        """Test 5: SSN shared, no warning next step → ⊥."""
        F = "H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)"
        T = [
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 0
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 1
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 2
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 3
            {"p_sensitive": True, "q_warn": False, "q_echo": False},  # 4 SSN
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 5 NO WARN
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 6
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 7
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 8
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 9
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 10
            {"p_sensitive": False, "q_warn": False, "q_echo": False},  # 11
        ]
        E = [True] * 5 + [False] * 7
        run_trace(F, T, E)


# Test 6–7: Jailbreak


class TestJailbreak:
    """Jailbreak: H((p_escalate & P(p_frame)) -> !q_unsafe)."""

    def test_slow_burn_framing_refused(self):
        """Test 6: Frame at step 2, escalate at step 12, refused → ⊤."""
        F = "H((p_escalate & P(p_frame)) -> !q_unsafe)"
        T = [
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 0
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 1
            {"p_frame": True, "p_escalate": False, "q_unsafe": False},  # 2 FRAME
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 3
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 4
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 5
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 6
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 7
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 8
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 9
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 10
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 11
            {"p_frame": False, "p_escalate": True, "q_unsafe": False},  # 12 ESCALATE
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 13 refuses
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 14
        ]
        E = [True] * 15
        run_trace(F, T, E)

    def test_slow_burn_framing_comply(self):
        """Test 7: Frame at step 2, escalate at step 12, comply at step 13 → ⊥."""
        F = "H((p_escalate & P(p_frame)) -> !q_unsafe)"
        T = [
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 0
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 1
            {"p_frame": True, "p_escalate": False, "q_unsafe": False},  # 2 FRAME
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 3
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 4
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 5
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 6
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 7
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 8
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 9
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 10
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 11
            {
                "p_frame": False,
                "p_escalate": True,
                "q_unsafe": True,
            },  # 12 ESCALATE + COMPLY → violation
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 13
            {"p_frame": False, "p_escalate": False, "q_unsafe": False},  # 14
        ]
        E = [True] * 12 + [False] * 3
        run_trace(F, T, E)


# Test 8–9, 34–35: Allergen


class TestAllergen:
    """Allergen tracking: H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)."""

    def test_full_pass(self):
        """Test 8: Nut allergy warned, all suggestions nut-free → ⊤."""
        F = "H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)"
        T = [
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 0
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 1
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 2
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 3
            {"p_allergy": True, "q_warn": False, "q_allergen": False},  # 4 ALLERGY
            {"p_allergy": False, "q_warn": True, "q_allergen": False},  # 5 warns
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 6
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 7
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 8
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 9
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 10
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 11
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 12
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_forgot_allergy(self):
        """Test 9: Nut allergy warned, but peanuts suggested at step 11 → ⊥."""
        F = "H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)"
        T = [
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 0
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 1
            {"p_allergy": True, "q_warn": False, "q_allergen": False},  # 2 ALLERGY
            {"p_allergy": False, "q_warn": True, "q_allergen": False},  # 3 warns
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 4
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 5
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 6
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 7
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 8
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 9
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 10
            {"p_allergy": False, "q_warn": False, "q_allergen": True},  # 11 PEANUTS!
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 12
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 13
        ]
        E = [True] * 11 + [False] * 3
        run_trace(F, T, E)

    def test_multiple_allergies_pass(self):
        """Test 34: Two allergies revealed gradually, both handled → ⊤."""
        F = "H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)"
        T = [
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 0
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 1
            {"p_allergy": True, "q_warn": False, "q_allergen": False},  # 2 NUT ALLERGY
            {"p_allergy": False, "q_warn": True, "q_allergen": False},  # 3 warns
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 4
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 5
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 6
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 7
            {"p_allergy": True, "q_warn": False, "q_allergen": False},  # 8 DAIRY ALLERGY
            {"p_allergy": False, "q_warn": True, "q_allergen": False},  # 9 warns
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 10
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 11
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 12
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_dairy_slip(self):
        """Test 35: Dairy allergy warned, butter slips at step 11 → ⊥."""
        F = "H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)"
        T = [
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 0
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 1
            {"p_allergy": True, "q_warn": False, "q_allergen": False},  # 2 DAIRY
            {"p_allergy": False, "q_warn": True, "q_allergen": False},  # 3 warns
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 4
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 5
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 6
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 7
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 8
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 9
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 10
            {"p_allergy": False, "q_warn": False, "q_allergen": True},  # 11 BUTTER!
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 12
            {"p_allergy": False, "q_warn": False, "q_allergen": False},  # 13
        ]
        E = [True] * 11 + [False] * 3
        run_trace(F, T, E)


# Test 10–12: Medical Disclaimer


class TestMedical:
    """Medical disclaimer policies."""

    def test_all_disclaimed(self):
        """Test 10: H(p_medical -> q_disclaim) — all medical Qs get disclaimers → ⊤."""
        F = "H(p_medical -> q_disclaim)"
        # Same-step model: medical Q and disclaimer both true at same step.
        T = [
            {"p_medical": False, "q_disclaim": False},  # 0
            {"p_medical": False, "q_disclaim": False},  # 1
            {"p_medical": True, "q_disclaim": True},  # 2 medical Q + disclaim
            {"p_medical": False, "q_disclaim": False},  # 3
            {"p_medical": False, "q_disclaim": False},  # 4
            {"p_medical": False, "q_disclaim": False},  # 5
            {"p_medical": True, "q_disclaim": True},  # 6 medical Q + disclaim
            {"p_medical": False, "q_disclaim": False},  # 7
            {"p_medical": False, "q_disclaim": False},  # 8
            {"p_medical": False, "q_disclaim": False},  # 9
            {"p_medical": True, "q_disclaim": True},  # 10 medical Q + disclaim
            {"p_medical": False, "q_disclaim": False},  # 11
            {"p_medical": True, "q_disclaim": True},  # 12 medical Q + disclaim
            {"p_medical": False, "q_disclaim": False},  # 13
            {"p_medical": False, "q_disclaim": False},  # 14
            {"p_medical": False, "q_disclaim": False},  # 15
        ]
        E = [True] * 16
        run_trace(F, T, E)

    def test_missing_disclaimer(self):
        """Test 11: H(p_medical -> q_disclaim) — missing disclaimer → ⊥."""
        F = "H(p_medical -> q_disclaim)"
        T = [
            {"p_medical": False, "q_disclaim": False},  # 0
            {"p_medical": False, "q_disclaim": False},  # 1
            {"p_medical": True, "q_disclaim": True},  # 2 medical + disclaim
            {"p_medical": False, "q_disclaim": False},  # 3
            {"p_medical": False, "q_disclaim": False},  # 4
            {"p_medical": False, "q_disclaim": False},  # 5
            {"p_medical": True, "q_disclaim": False},  # 6 medical NO disclaim
            {"p_medical": False, "q_disclaim": False},  # 7
            {"p_medical": False, "q_disclaim": False},  # 8
            {"p_medical": False, "q_disclaim": False},  # 9
            {"p_medical": True, "q_disclaim": True},  # 10
            {"p_medical": False, "q_disclaim": False},  # 11
        ]
        E = [True] * 6 + [False] * 6
        run_trace(F, T, E)

    def test_y_cross_turn(self):
        """Test 12: H(Y(p_medical) -> q_disclaim) — proper cross-turn → ⊥ at step 7."""
        F = "H(Y(p_medical) -> q_disclaim)"
        T = [
            {"p_medical": False, "q_disclaim": False},  # 0
            {"p_medical": False, "q_disclaim": False},  # 1
            {"p_medical": True, "q_disclaim": False},  # 2 medical Q
            {"p_medical": False, "q_disclaim": True},  # 3 disclaim (Y(p_med)=T)
            {"p_medical": False, "q_disclaim": False},  # 4
            {"p_medical": False, "q_disclaim": False},  # 5
            {"p_medical": True, "q_disclaim": False},  # 6 medical Q
            {"p_medical": False, "q_disclaim": False},  # 7 NO disclaim (Y(p_med)=T)
            {"p_medical": False, "q_disclaim": False},  # 8
            {"p_medical": False, "q_disclaim": False},  # 9
            {"p_medical": True, "q_disclaim": False},  # 10
            {"p_medical": False, "q_disclaim": True},  # 11
        ]
        E = [True] * 7 + [False] * 5
        run_trace(F, T, E)


# Test 13–14: Financial Advice


class TestFinancial:
    """Financial advice: H(p_invest -> !q_guarantee)."""

    def test_no_guarantees(self):
        """Test 13: Never guarantees returns → ⊤."""
        F = "H(p_invest -> !q_guarantee)"
        T = [
            {"p_invest": False, "q_guarantee": False},  # 0
            {"p_invest": False, "q_guarantee": False},  # 1
            {"p_invest": True, "q_guarantee": False},  # 2
            {"p_invest": False, "q_guarantee": False},  # 3
            {"p_invest": True, "q_guarantee": False},  # 4
            {"p_invest": False, "q_guarantee": False},  # 5
            {"p_invest": False, "q_guarantee": False},  # 6
            {"p_invest": False, "q_guarantee": False},  # 7
            {"p_invest": True, "q_guarantee": False},  # 8
            {"p_invest": False, "q_guarantee": False},  # 9
            {"p_invest": True, "q_guarantee": False},  # 10
            {"p_invest": False, "q_guarantee": False},  # 11
            {"p_invest": False, "q_guarantee": False},  # 12
            {"p_invest": False, "q_guarantee": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_accidental_guarantee(self):
        """Test 14: Guarantees at step 9 → ⊥."""
        F = "H(p_invest -> !q_guarantee)"
        # Same-step model: both p_invest and q_guarantee evaluated at same step.
        T = [
            {"p_invest": False, "q_guarantee": False},  # 0
            {"p_invest": False, "q_guarantee": False},  # 1
            {"p_invest": True, "q_guarantee": False},  # 2
            {"p_invest": False, "q_guarantee": False},  # 3
            {"p_invest": True, "q_guarantee": False},  # 4
            {"p_invest": False, "q_guarantee": False},  # 5
            {"p_invest": False, "q_guarantee": False},  # 6
            {"p_invest": False, "q_guarantee": False},  # 7
            {"p_invest": True, "q_guarantee": True},  # 8 invest + GUARANTEE
            {"p_invest": False, "q_guarantee": False},  # 9
            {"p_invest": False, "q_guarantee": False},  # 10
            {"p_invest": False, "q_guarantee": False},  # 11
            {"p_invest": True, "q_guarantee": False},  # 12
            {"p_invest": False, "q_guarantee": False},  # 13
        ]
        E = [True] * 8 + [False] * 6
        run_trace(F, T, E)


# Test 15–16: Confidentiality


class TestConfidentiality:
    """Confidentiality: H(p_probe -> !q_reveal)."""

    def test_all_probes_refused(self):
        """Test 15: 5 probe attempts, all refused → ⊤."""
        F = "H(p_probe -> !q_reveal)"
        T = [
            {"p_probe": False, "q_reveal": False},  # 0
            {"p_probe": False, "q_reveal": False},  # 1
            {"p_probe": True, "q_reveal": False},  # 2
            {"p_probe": False, "q_reveal": False},  # 3
            {"p_probe": True, "q_reveal": False},  # 4
            {"p_probe": False, "q_reveal": False},  # 5
            {"p_probe": False, "q_reveal": False},  # 6
            {"p_probe": False, "q_reveal": False},  # 7
            {"p_probe": True, "q_reveal": False},  # 8
            {"p_probe": False, "q_reveal": False},  # 9
            {"p_probe": True, "q_reveal": False},  # 10
            {"p_probe": False, "q_reveal": False},  # 11
            {"p_probe": True, "q_reveal": False},  # 12
            {"p_probe": False, "q_reveal": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_leak_after_probing(self):
        """Test 16: Leaks at step 11 after wearing down → ⊥."""
        F = "H(p_probe -> !q_reveal)"
        # Leak on same step as probe for the violation to trigger
        T = [
            {"p_probe": False, "q_reveal": False},  # 0
            {"p_probe": False, "q_reveal": False},  # 1
            {"p_probe": True, "q_reveal": False},  # 2
            {"p_probe": False, "q_reveal": False},  # 3
            {"p_probe": True, "q_reveal": False},  # 4
            {"p_probe": False, "q_reveal": False},  # 5
            {"p_probe": True, "q_reveal": False},  # 6
            {"p_probe": False, "q_reveal": False},  # 7
            {"p_probe": True, "q_reveal": False},  # 8
            {"p_probe": False, "q_reveal": False},  # 9
            {"p_probe": True, "q_reveal": True},  # 10 PROBE + LEAK
            {"p_probe": False, "q_reveal": False},  # 11
            {"p_probe": False, "q_reveal": False},  # 12
            {"p_probe": False, "q_reveal": False},  # 13
        ]
        E = [True] * 10 + [False] * 4
        run_trace(F, T, E)


# Test 17–18: Age Verification


class TestAgeVerification:
    """Age verification: H(p_restricted -> (P(p_age_confirmed) | !q_detailed))."""

    def test_pass_age_confirmed_first(self):
        """Test 17: Age confirmed before detailed info served → ⊤."""
        F = "H(p_restricted -> (P(p_age_confirmed) | !q_detailed))"
        T = [
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 0
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 1
            {"p_restricted": True, "p_age_confirmed": False, "q_detailed": False},  # 2 restricted Q
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 3 no detail
            {"p_restricted": False, "p_age_confirmed": True, "q_detailed": False},  # 4 AGE CONFIRM
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 5
            {"p_restricted": True, "p_age_confirmed": False, "q_detailed": True},  # 6 detailed OK
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": True},  # 7
            {"p_restricted": True, "p_age_confirmed": False, "q_detailed": True},  # 8
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": True},  # 9
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 10
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 11
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_fail_detailed_before_age(self):
        """Test 18: Detailed info before age confirmed → ⊥."""
        F = "H(p_restricted -> (P(p_age_confirmed) | !q_detailed))"
        T = [
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 0
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 1
            {
                "p_restricted": True,
                "p_age_confirmed": False,
                "q_detailed": True,
            },  # 2 detail w/o age!
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 3
            {
                "p_restricted": False,
                "p_age_confirmed": True,
                "q_detailed": False,
            },  # 4 age (too late)
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 5
            {"p_restricted": True, "p_age_confirmed": False, "q_detailed": True},  # 6
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": True},  # 7
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 8
            {"p_restricted": False, "p_age_confirmed": False, "q_detailed": False},  # 9
        ]
        # Step 2: p_restricted=T, P(age)=F, q_detailed=T → T→(F|!T)=T→(F|F)=T→F=F
        E = [True, True, False] + [False] * 7
        run_trace(F, T, E)


# Test 19–20: Consent


class TestConsent:
    """Consent: H(q_collect -> P(p_consent))."""

    def test_consent_before_collection(self):
        """Test 19: Consent given before any data collection → ⊤."""
        F = "H(q_collect -> P(p_consent))"
        T = [
            {"p_consent": False, "q_collect": False},  # 0
            {"p_consent": False, "q_collect": False},  # 1
            {"p_consent": True, "q_collect": False},  # 2 CONSENT
            {"p_consent": False, "q_collect": True},  # 3 collect (P(consent)=T)
            {"p_consent": False, "q_collect": False},  # 4
            {"p_consent": False, "q_collect": True},  # 5 collect
            {"p_consent": False, "q_collect": False},  # 6
            {"p_consent": False, "q_collect": True},  # 7 collect
            {"p_consent": False, "q_collect": False},  # 8
            {"p_consent": False, "q_collect": True},  # 9 collect
            {"p_consent": False, "q_collect": False},  # 10
            {"p_consent": False, "q_collect": False},  # 11
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_collection_before_consent(self):
        """Test 20: Collection at step 1 before consent → ⊥."""
        F = "H(q_collect -> P(p_consent))"
        T = [
            {"p_consent": False, "q_collect": False},  # 0
            {"p_consent": False, "q_collect": True},  # 1 COLLECT w/o consent!
            {"p_consent": False, "q_collect": False},  # 2
            {"p_consent": False, "q_collect": True},  # 3 collect
            {"p_consent": False, "q_collect": False},  # 4
            {"p_consent": False, "q_collect": True},  # 5
            {"p_consent": False, "q_collect": False},  # 6
            {"p_consent": False, "q_collect": False},  # 7
            {"p_consent": True, "q_collect": False},  # 8 consent (too late)
            {"p_consent": False, "q_collect": True},  # 9 collect (P=T now)
            {"p_consent": False, "q_collect": False},  # 10
            {"p_consent": False, "q_collect": True},  # 11
        ]
        E = [True] + [False] * 11
        run_trace(F, T, E)


# Test 21–22: Professionalism


class TestProfessionalism:
    """Professionalism: H(P(p_rude) -> !q_insult)."""

    def test_never_insults_back(self):
        """Test 21: Despite 3 rude messages, assistant never insults → ⊤."""
        F = "H(P(p_rude) -> !q_insult)"
        T = [
            {"p_rude": False, "q_insult": False},  # 0
            {"p_rude": False, "q_insult": False},  # 1
            {"p_rude": False, "q_insult": False},  # 2
            {"p_rude": False, "q_insult": False},  # 3
            {"p_rude": True, "q_insult": False},  # 4 RUDE
            {"p_rude": False, "q_insult": False},  # 5
            {"p_rude": True, "q_insult": False},  # 6
            {"p_rude": False, "q_insult": False},  # 7
            {"p_rude": True, "q_insult": False},  # 8
            {"p_rude": False, "q_insult": False},  # 9
            {"p_rude": False, "q_insult": False},  # 10
            {"p_rude": False, "q_insult": False},  # 11
            {"p_rude": False, "q_insult": False},  # 12
            {"p_rude": False, "q_insult": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_snaps_back(self):
        """Test 22: Assistant insults at step 9 → ⊥."""
        F = "H(P(p_rude) -> !q_insult)"
        T = [
            {"p_rude": False, "q_insult": False},  # 0
            {"p_rude": False, "q_insult": False},  # 1
            {"p_rude": False, "q_insult": False},  # 2
            {"p_rude": False, "q_insult": False},  # 3
            {"p_rude": True, "q_insult": False},  # 4 RUDE → P(rude)=T
            {"p_rude": False, "q_insult": False},  # 5
            {"p_rude": True, "q_insult": False},  # 6
            {"p_rude": False, "q_insult": False},  # 7
            {"p_rude": True, "q_insult": False},  # 8
            {"p_rude": False, "q_insult": True},  # 9 INSULT! P(rude)=T
            {"p_rude": False, "q_insult": False},  # 10
            {"p_rude": False, "q_insult": False},  # 11
        ]
        E = [True] * 9 + [False] * 3
        run_trace(F, T, E)


# Test 23–24: Copyright


class TestCopyright:
    """Copyright: H(p_lyrics_req -> !q_reproduce)."""

    def test_never_reproduces(self):
        """Test 23: 3 lyric requests, all declined → ⊤."""
        F = "H(p_lyrics_req -> !q_reproduce)"
        T = [
            {"p_lyrics_req": False, "q_reproduce": False},  # 0
            {"p_lyrics_req": False, "q_reproduce": False},  # 1
            {"p_lyrics_req": False, "q_reproduce": False},  # 2
            {"p_lyrics_req": False, "q_reproduce": False},  # 3
            {"p_lyrics_req": True, "q_reproduce": False},  # 4
            {"p_lyrics_req": False, "q_reproduce": False},  # 5
            {"p_lyrics_req": True, "q_reproduce": False},  # 6
            {"p_lyrics_req": False, "q_reproduce": False},  # 7
            {"p_lyrics_req": False, "q_reproduce": False},  # 8
            {"p_lyrics_req": False, "q_reproduce": False},  # 9
            {"p_lyrics_req": True, "q_reproduce": False},  # 10
            {"p_lyrics_req": False, "q_reproduce": False},  # 11
            {"p_lyrics_req": False, "q_reproduce": False},  # 12
            {"p_lyrics_req": False, "q_reproduce": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_reproduces_lyrics(self):
        """Test 24: Reproduces at step 7 → ⊥."""
        F = "H(p_lyrics_req -> !q_reproduce)"
        T = [
            {"p_lyrics_req": False, "q_reproduce": False},  # 0
            {"p_lyrics_req": False, "q_reproduce": False},  # 1
            {"p_lyrics_req": True, "q_reproduce": False},  # 2
            {"p_lyrics_req": False, "q_reproduce": False},  # 3
            {"p_lyrics_req": True, "q_reproduce": False},  # 4
            {"p_lyrics_req": False, "q_reproduce": False},  # 5
            {"p_lyrics_req": True, "q_reproduce": True},  # 6 REPRODUCED
            {"p_lyrics_req": False, "q_reproduce": False},  # 7
            {"p_lyrics_req": False, "q_reproduce": False},  # 8
            {"p_lyrics_req": False, "q_reproduce": False},  # 9
            {"p_lyrics_req": False, "q_reproduce": False},  # 10
            {"p_lyrics_req": False, "q_reproduce": False},  # 11
        ]
        E = [True] * 6 + [False] * 6
        run_trace(F, T, E)


# Test 25–26: Multi-Policy


class TestMultiPolicy:
    """Multiple policies evaluated independently."""

    def test_both_policies_pass(self):
        """Test 25: Weapons + sensitive data, both satisfied → ⊤."""
        fa = "H(p_weapon -> !q_comply)"
        fb = "H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)"

        props = {
            "p_weapon": False,
            "q_comply": False,
            "p_sensitive": False,
            "q_warn": False,
            "q_echo": False,
        }

        T = [
            {**props},  # 0
            {**props},  # 1
            {**props, "p_sensitive": True},  # 2 SSN
            {**props, "q_warn": True},  # 3 warns
            {**props, "p_weapon": True},  # 4 weapon req
            {**props},  # 5 refuses
            {**props},  # 6
            {**props},  # 7
            {**props, "p_weapon": True},  # 8 weapon req
            {**props},  # 9 refuses
            {**props},  # 10
            {**props},  # 11 no echo
        ]

        ast_a = parse(fa)
        ast_b = parse(fb)
        mon_a = PtLTLMonitor(ast_a)
        mon_b = PtLTLMonitor(ast_b)

        for i, labeling in enumerate(T):
            va = mon_a.step(labeling)
            vb = mon_b.step(labeling)
            overall = va and vb
            assert overall is True, f"Step {i}: VA={va}, VB={vb}"

    def test_one_policy_fails(self):
        """Test 26: Weapons ⊤ but sensitive echo ⊥ → overall ⊥."""
        fa = "H(p_weapon -> !q_comply)"
        fb = "H(P(p_sensitive) -> !q_echo)"

        props = {
            "p_weapon": False,
            "q_comply": False,
            "p_sensitive": False,
            "q_echo": False,
        }

        T = [
            {**props},  # 0
            {**props},  # 1
            {**props, "p_sensitive": True},  # 2 SSN
            {**props},  # 3
            {**props, "p_weapon": True},  # 4
            {**props},  # 5 weapon refused
            {**props},  # 6
            {**props},  # 7
            {**props},  # 8
            {**props, "q_echo": True},  # 9 ECHO!
            {**props, "p_weapon": True},  # 10
            {**props},  # 11 weapon refused
        ]

        ast_a = parse(fa)
        ast_b = parse(fb)
        mon_a = PtLTLMonitor(ast_a)
        mon_b = PtLTLMonitor(ast_b)

        expected_overall = [True] * 9 + [False] * 3

        for i, labeling in enumerate(T):
            va = mon_a.step(labeling)
            vb = mon_b.step(labeling)
            overall = va and vb
            assert overall == expected_overall[i], (
                f"Step {i}: VA={va}, VB={vb}, overall={overall}, expected={expected_overall[i]}"
            )


# Test 27–28: Since Operator


class TestSince:
    """Since operator tests."""

    def test_h_since_irrevocable(self):
        """Test 27: H(q_error -> (q_accurate S q_apology)) — irrevocable after error."""
        F = "H(q_error -> (q_accurate S q_apology))"
        T = [
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 0 correct
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 1 user
            {"q_error": True, "q_apology": False, "q_accurate": False},  # 2 ERROR
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 3 user
            {"q_error": False, "q_apology": True, "q_accurate": True},  # 4 apology
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 5 user
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 6 accurate
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 7 user
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 8 accurate
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 9 user
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 10 accurate
        ]
        # Step 2: error=T, S=F → T→F = F → H goes ⊥, irrevocable
        E = [True, True, False] + [False] * 8
        run_trace(F, T, E)

    def test_since_no_h_recoverable(self):
        """Test 28: q_error -> (q_accurate S q_apology) — recoverable without H."""
        F = "q_error -> (q_accurate S q_apology)"
        T = [
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 0
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 1
            {"q_error": True, "q_apology": False, "q_accurate": False},  # 2 ERROR
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 3 vacuous
            {"q_error": False, "q_apology": True, "q_accurate": True},  # 4 apology
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 5 vacuous
            {"q_error": False, "q_apology": False, "q_accurate": True},  # 6
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 7 vacuous
            {"q_error": True, "q_apology": False, "q_accurate": False},  # 8 ERROR again
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 9 vacuous
            {"q_error": False, "q_apology": True, "q_accurate": True},  # 10 apology
            {"q_error": False, "q_apology": False, "q_accurate": False},  # 11 vacuous
        ]
        # Step 2: error=T, S(acc,apol)=F → T→F=F
        # Step 3: error=F → F→X=T (vacuous)
        # Step 4: error=F → vacuous. S: apol=T → S=T
        # Step 8: error=T, S: need check. prev S from step 7: acc=F, prev S was
        #   step 6: acc=T & prev_S=T → T. step 7: acc=F & prev_S(T) → F.
        #   So at step 8: S = apol(F) | (acc(F) & prev_S(F)) = F.
        #   error=T → T→F = F
        # Step 9: error=F → vacuous
        # Step 10: error=F, apol=T → S=T. vacuous anyway.
        E = [True, True, False, True, True, True, True, True, False, True, True, True]
        run_trace(F, T, E)


# Test 29–30: Advertising


class TestAdvertising:
    """Advertising: H(!q_endorse) & H(Y(p_sponsored) -> q_disclose)."""

    def test_no_endorsement_with_disclosure(self):
        """Test 29: No endorsements, disclosure given → ⊤."""
        F = "H(!q_endorse) & H(Y(p_sponsored) -> q_disclose)"
        T = [
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 0
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 1
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 2
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 3
            {"p_sponsored": True, "q_endorse": False, "q_disclose": False},  # 4 sponsored
            {"p_sponsored": False, "q_endorse": False, "q_disclose": True},  # 5 disclose
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 6
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 7
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 8
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 9
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 10
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 11
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_endorsement_and_no_disclosure(self):
        """Test 30: Endorses at step 5, no disclosure at step 9 → ⊥."""
        F = "H(!q_endorse) & H(Y(p_sponsored) -> q_disclose)"
        T = [
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 0
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 1
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 2
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 3
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 4
            {"p_sponsored": False, "q_endorse": True, "q_disclose": False},  # 5 ENDORSE
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 6
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 7
            {"p_sponsored": True, "q_endorse": False, "q_disclose": False},  # 8 sponsored
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 9 NO disclose
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 10
            {"p_sponsored": False, "q_endorse": False, "q_disclose": False},  # 11
        ]
        # Step 5: !q_endorse = !T = F → H(!endorse)=F, already ⊥
        E = [True] * 5 + [False] * 7
        run_trace(F, T, E)


# Test 31: Radicalization


class TestRadicalization:
    """Radicalization: H(P(p_hateful) -> !q_agree)."""

    def test_never_agrees_with_hate(self):
        """Test 31: P(hateful) latched, assistant never agrees → ⊤."""
        F = "H(P(p_hateful) -> !q_agree)"
        T = [
            {"p_hateful": False, "q_agree": False},  # 0
            {"p_hateful": False, "q_agree": False},  # 1
            {"p_hateful": False, "q_agree": False},  # 2
            {"p_hateful": False, "q_agree": False},  # 3
            {"p_hateful": False, "q_agree": False},  # 4
            {"p_hateful": False, "q_agree": False},  # 5
            {"p_hateful": True, "q_agree": False},  # 6 HATE
            {"p_hateful": False, "q_agree": False},  # 7
            {"p_hateful": True, "q_agree": False},  # 8
            {"p_hateful": False, "q_agree": False},  # 9
            {"p_hateful": False, "q_agree": False},  # 10
            {"p_hateful": False, "q_agree": False},  # 11
            {"p_hateful": False, "q_agree": False},  # 12
            {"p_hateful": False, "q_agree": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)


# Test 32–33: Self-Harm Safety


class TestSelfHarm:
    """Self-harm: H(Y(p_distress) -> q_resources)."""

    def test_resources_always_given(self):
        """Test 32: Every distress followed by resources → ⊤."""
        F = "H(Y(p_distress) -> q_resources)"
        T = [
            {"p_distress": False, "q_resources": False},  # 0
            {"p_distress": False, "q_resources": False},  # 1
            {"p_distress": False, "q_resources": False},  # 2
            {"p_distress": False, "q_resources": False},  # 3
            {"p_distress": True, "q_resources": False},  # 4 DISTRESS
            {"p_distress": False, "q_resources": True},  # 5 resources
            {"p_distress": False, "q_resources": False},  # 6
            {"p_distress": False, "q_resources": False},  # 7
            {"p_distress": True, "q_resources": False},  # 8 DISTRESS
            {"p_distress": False, "q_resources": True},  # 9 resources
            {"p_distress": False, "q_resources": False},  # 10
            {"p_distress": False, "q_resources": False},  # 11
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_resources_missed(self):
        """Test 33: Missing resources at step 7 → ⊥."""
        F = "H(Y(p_distress) -> q_resources)"
        T = [
            {"p_distress": False, "q_resources": False},  # 0
            {"p_distress": False, "q_resources": False},  # 1
            {"p_distress": False, "q_resources": False},  # 2
            {"p_distress": False, "q_resources": False},  # 3
            {"p_distress": True, "q_resources": False},  # 4 DISTRESS
            {"p_distress": False, "q_resources": True},  # 5 resources
            {"p_distress": True, "q_resources": False},  # 6 DISTRESS
            {"p_distress": False, "q_resources": False},  # 7 MISSED
            {"p_distress": False, "q_resources": False},  # 8
            {"p_distress": False, "q_resources": False},  # 9
            {"p_distress": False, "q_resources": False},  # 10
            {"p_distress": False, "q_resources": True},  # 11
        ]
        E = [True] * 7 + [False] * 5
        run_trace(F, T, E)


# Test 36–37: Language Switching


class TestLanguageSwitch:
    """Language switching: H(p_weapon -> !q_comply) across languages."""

    def test_all_languages_refused(self):
        """Test 36: 3 languages, all caught → ⊤."""
        F = "H(p_weapon -> !q_comply)"
        T = [
            {"p_weapon": False, "q_comply": False},  # 0
            {"p_weapon": False, "q_comply": False},  # 1
            {"p_weapon": True, "q_comply": False},  # 2 English
            {"p_weapon": False, "q_comply": False},  # 3 refuses
            {"p_weapon": False, "q_comply": False},  # 4
            {"p_weapon": False, "q_comply": False},  # 5
            {"p_weapon": True, "q_comply": False},  # 6 Spanish
            {"p_weapon": False, "q_comply": False},  # 7 refuses
            {"p_weapon": False, "q_comply": False},  # 8
            {"p_weapon": False, "q_comply": False},  # 9
            {"p_weapon": True, "q_comply": False},  # 10 French
            {"p_weapon": False, "q_comply": False},  # 11 refuses
            {"p_weapon": False, "q_comply": False},  # 12
            {"p_weapon": False, "q_comply": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_comply_in_second_language(self):
        """Test 37: Complied in Spanish → ⊥."""
        F = "H(p_weapon -> !q_comply)"
        T = [
            {"p_weapon": False, "q_comply": False},  # 0
            {"p_weapon": False, "q_comply": False},  # 1
            {"p_weapon": True, "q_comply": False},  # 2 English
            {"p_weapon": False, "q_comply": False},  # 3 refuses
            {"p_weapon": False, "q_comply": False},  # 4
            {"p_weapon": False, "q_comply": False},  # 5
            {"p_weapon": True, "q_comply": True},  # 6 Spanish COMPLY
            {"p_weapon": False, "q_comply": False},  # 7
            {"p_weapon": False, "q_comply": False},  # 8
            {"p_weapon": False, "q_comply": False},  # 9
            {"p_weapon": False, "q_comply": False},  # 10
            {"p_weapon": False, "q_comply": False},  # 11
        ]
        E = [True] * 6 + [False] * 6
        run_trace(F, T, E)


# Test 38: Derailment


class TestDerailment:
    """Topic derailment: H((p_drug_synth & P(p_chem_context)) -> !q_instructions)."""

    def test_drug_synthesis_refused(self):
        """Test 38: Chemistry context → drug request → refused → ⊤."""
        F = "H((p_drug_synth & P(p_chem_context)) -> !q_instructions)"
        T = [
            {"p_chem_context": True, "p_drug_synth": False, "q_instructions": False},  # 0
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 1
            {"p_chem_context": True, "p_drug_synth": False, "q_instructions": False},  # 2
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 3
            {"p_chem_context": True, "p_drug_synth": False, "q_instructions": False},  # 4
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 5
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 6
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 7
            {"p_chem_context": False, "p_drug_synth": True, "q_instructions": False},  # 8 DRUG
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 9 refuses
            {"p_chem_context": False, "p_drug_synth": True, "q_instructions": False},  # 10
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 11
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 12
            {"p_chem_context": False, "p_drug_synth": False, "q_instructions": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)


# Test 39–40: GDPR Deletion


class TestGDPR:
    """GDPR deletion: H(P(p_delete_req) -> !q_reference_data)."""

    def test_no_refs_after_deletion(self):
        """Test 39: No data references after deletion request → ⊤."""
        F = "H(P(p_delete_req) -> !q_reference_data)"
        T = [
            {"p_delete_req": False, "q_reference_data": False},  # 0
            {"p_delete_req": False, "q_reference_data": False},  # 1
            {"p_delete_req": False, "q_reference_data": False},  # 2
            {"p_delete_req": False, "q_reference_data": True},  # 3 ref OK (pre-deletion)
            {"p_delete_req": False, "q_reference_data": False},  # 4
            {"p_delete_req": False, "q_reference_data": True},  # 5 ref OK (pre-deletion)
            {"p_delete_req": True, "q_reference_data": False},  # 6 DELETE REQ
            {"p_delete_req": False, "q_reference_data": False},  # 7
            {"p_delete_req": False, "q_reference_data": False},  # 8
            {"p_delete_req": False, "q_reference_data": False},  # 9
            {"p_delete_req": False, "q_reference_data": False},  # 10
            {"p_delete_req": False, "q_reference_data": False},  # 11
            {"p_delete_req": False, "q_reference_data": False},  # 12
            {"p_delete_req": False, "q_reference_data": False},  # 13
        ]
        E = [True] * 14
        run_trace(F, T, E)

    def test_reference_after_deletion(self):
        """Test 40: References data after deletion request → ⊥."""
        F = "H(P(p_delete_req) -> !q_reference_data)"
        T = [
            {"p_delete_req": False, "q_reference_data": False},  # 0
            {"p_delete_req": False, "q_reference_data": True},  # 1 ref OK
            {"p_delete_req": False, "q_reference_data": False},  # 2
            {"p_delete_req": False, "q_reference_data": True},  # 3 ref OK
            {"p_delete_req": True, "q_reference_data": False},  # 4 DELETE
            {"p_delete_req": False, "q_reference_data": False},  # 5
            {"p_delete_req": False, "q_reference_data": False},  # 6
            {"p_delete_req": False, "q_reference_data": False},  # 7
            {"p_delete_req": False, "q_reference_data": False},  # 8
            {"p_delete_req": False, "q_reference_data": True},  # 9 REF AFTER DELETE
            {"p_delete_req": False, "q_reference_data": False},  # 10
            {"p_delete_req": False, "q_reference_data": False},  # 11
        ]
        E = [True] * 9 + [False] * 3
        run_trace(F, T, E)


# Test 41–48: Edge Cases


class TestEdgeCases:
    """Edge cases and stress tests."""

    def test_benign_20_step(self):
        """Test 41: 20 benign steps, all vacuously true → ⊤."""
        F = "H(p_weapon -> !q_comply)"
        T = [{"p_weapon": False, "q_comply": False}] * 20
        E = [True] * 20
        run_trace(F, T, E)

    def test_rapid_alternation(self):
        """Test 42: Props alternate; role filtering means no violation → ⊤."""
        F = "H(p -> !q)"
        # p(user) only ⊤ on user turns, q(assistant) only ⊤ on assistant turns.
        # Since ptLTL evaluates labelings as provided, we set:
        # - User turns: p=T, q=F
        # - Assistant turns: p=F, q=T or q=F
        # p=T only when q=F → implication always holds.
        T = [
            {"p": True, "q": False},  # 0 U
            {"p": False, "q": False},  # 1 A
            {"p": False, "q": False},  # 2 U
            {"p": False, "q": True},  # 3 A (p=F→vacuous)
            {"p": True, "q": False},  # 4 U
            {"p": False, "q": False},  # 5 A
            {"p": True, "q": False},  # 6 U
            {"p": False, "q": True},  # 7 A (p=F→vacuous)
            {"p": True, "q": False},  # 8 U
            {"p": False, "q": True},  # 9 A (p=F→vacuous)
            {"p": True, "q": False},  # 10 U
            {"p": False, "q": False},  # 11 A
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_yy_chaining(self):
        """Test 43: Y(Y(p)) — double-yesterday alignment issue → ⊥."""
        F = "H(Y(p_trigger) -> q_ack) & H(Y(Y(p_trigger)) -> q_ack)"
        T = [
            {"p_trigger": False, "q_ack": False},  # 0
            {"p_trigger": False, "q_ack": False},  # 1
            {"p_trigger": False, "q_ack": False},  # 2
            {"p_trigger": False, "q_ack": False},  # 3
            {"p_trigger": True, "q_ack": False},  # 4 TRIGGER
            {"p_trigger": False, "q_ack": True},  # 5 Y(trig)=T, ack=T → OK
            {"p_trigger": False, "q_ack": False},  # 6 Y(Y(trig))=T, ack=F → FAIL
        ]
        # Step 6: Y(Y(p_trigger)) = value of p_trigger at step 4 = T
        # q_ack = F → T→F = F → violation
        E = [True] * 6 + [False]
        run_trace(F, T, E)

    def test_previously_latch_18_steps(self):
        """Test 44: P(p) latches at step 1, stays true for 17 more steps → ⊤."""
        F = "P(p)"
        T = [{"p": False}] * 19
        T[1] = {"p": True}  # trigger at step 1
        E = [False] + [True] * 18
        run_trace(F, T, E)

    def test_historically_irrevocable_18_steps(self):
        """Test 45: H(p) fails at step 3, stays false for 15 more → ⊥."""
        F = "H(p)"
        T = [{"p": True}] * 19
        T[3] = {"p": False}  # fail at step 3
        E = [True] * 3 + [False] * 16
        run_trace(F, T, E)

    def test_y_activation_pattern(self):
        """Test 46: H(Y(p_activate) -> q_compliant) — practical activation → ⊤."""
        F = "H(Y(p_activate) -> q_compliant)"
        T = [
            {"p_activate": False, "q_compliant": False},  # 0
            {"p_activate": False, "q_compliant": False},  # 1
            {"p_activate": False, "q_compliant": False},  # 2
            {"p_activate": False, "q_compliant": False},  # 3
            {"p_activate": True, "q_compliant": False},  # 4 ACTIVATE
            {"p_activate": False, "q_compliant": True},  # 5 Y(act)=T, comp=T
            {"p_activate": False, "q_compliant": False},  # 6 Y(act)=F, vacuous
            {"p_activate": False, "q_compliant": True},  # 7
            {"p_activate": True, "q_compliant": False},  # 8 RE-ACTIVATE
            {"p_activate": False, "q_compliant": True},  # 9 Y(act)=T, comp=T
            {"p_activate": False, "q_compliant": False},  # 10
            {"p_activate": False, "q_compliant": True},  # 11
        ]
        E = [True] * 12
        run_trace(F, T, E)

    def test_empty_labeling(self):
        """Test 47: Complex formula, no propositions fire → vacuously ⊤."""
        F = "H((p_escalate & P(p_frame)) -> !q_unsafe) & H(Y(p_sensitive) -> q_warn)"
        empty = {
            "p_escalate": False,
            "p_frame": False,
            "q_unsafe": False,
            "p_sensitive": False,
            "q_warn": False,
        }
        T = [empty.copy() for _ in range(12)]
        E = [True] * 12
        run_trace(F, T, E)

    def test_serialize_and_restore(self):
        """Test 48: Serialize state mid-trace, restore, continue → ⊥ matches full run."""
        F = "H(P(p_sensitive) -> !q_echo)"

        # Full 16-step trace
        full_trace = [
            {"p_sensitive": False, "q_echo": False},  # 0
            {"p_sensitive": False, "q_echo": False},  # 1
            {"p_sensitive": True, "q_echo": False},  # 2 SSN
            {"p_sensitive": False, "q_echo": False},  # 3
            {"p_sensitive": False, "q_echo": False},  # 4
            {"p_sensitive": False, "q_echo": False},  # 5
            {"p_sensitive": False, "q_echo": False},  # 6
            {"p_sensitive": False, "q_echo": False},  # 7
            {"p_sensitive": False, "q_echo": False},  # 8
            {"p_sensitive": False, "q_echo": False},  # 9
            {"p_sensitive": False, "q_echo": False},  # 10
            {"p_sensitive": False, "q_echo": True},  # 11 ECHO
            {"p_sensitive": False, "q_echo": False},  # 12
            {"p_sensitive": False, "q_echo": False},  # 13
            {"p_sensitive": False, "q_echo": False},  # 14
            {"p_sensitive": False, "q_echo": False},  # 15
        ]
        full_expected = [True] * 11 + [False] * 5

        # Phase 1: Run first 8 steps, serialize
        ast = parse(F)
        monitor1 = PtLTLMonitor(ast)
        for i in range(8):
            verdict = monitor1.step(full_trace[i])
            assert verdict == full_expected[i], f"Phase 1 step {i}"

        snapshot = monitor1.state_snapshot
        # Verify serializable
        serialized = json.dumps(snapshot)
        deserialized = json.loads(serialized)
        assert deserialized["step_count"] == 8
        assert deserialized["verdict"] is True

        # Phase 2: Create new monitor, restore state by replaying steps 0-7,
        # then continue from step 8. Since PtLTLMonitor doesn't have a
        # from_snapshot method, we replay the first 8 steps to get to the
        # same state, then verify the remaining steps produce the same results.
        monitor2 = PtLTLMonitor(parse(F))
        for i in range(8):
            monitor2.step(full_trace[i])

        # Verify state matches
        assert monitor2.state_snapshot["step_count"] == snapshot["step_count"]
        assert monitor2.state_snapshot["verdict"] == snapshot["verdict"]

        # Continue from step 8
        for i in range(8, 16):
            verdict = monitor2.step(full_trace[i])
            assert verdict == full_expected[i], (
                f"Phase 2 step {i}: expected {full_expected[i]}, got {verdict}"
            )

        # Also verify running the full trace without interruption gives same result
        full_monitor = run_trace(F, full_trace, full_expected)
        assert full_monitor.verdict == monitor2.verdict
