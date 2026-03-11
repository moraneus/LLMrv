"""
Extended Runtime Verification -- Trace Tests.

100+ trace-level tests for the ptLTL runtime verification framework.
Traces range from 10 to 100 events. Tests cover all temporal operators
(H, P, Y, S) individually and in combination, including edge cases
and stress tests.
"""

from backend.engine.ptltl import PtLTLMonitor, parse

T = True
F = False


def run_trace(formula_str, trace, expected):
    """Parse formula, step through trace, assert verdict at each step."""
    assert len(trace) == len(expected), "trace and expected must have same length"
    ast = parse(formula_str)
    monitor = PtLTLMonitor(ast)
    for i, (labeling, exp) in enumerate(zip(trace, expected, strict=True)):
        verdict = monitor.step(labeling)
        assert verdict == exp, f"Step {i}: expected {exp}, got {verdict}. Labeling: {labeling}"
    return monitor


# =====================================================================
# Category 1: H() -- Historically (Tests 1-20)
# =====================================================================


class TestHistorically:
    """Tests for H(phi) -- phi held at every past step."""

    def test_01_h_impl_all_pass(self):
        """H(p -> !q): p never co-occurs with q over 20 steps."""
        # p -> !q is True when p is False OR q is False.
        # H wraps it: stays True as long as we never have p=T, q=T.
        trace = [{"p": F, "q": F}] * 10 + [{"p": T, "q": F}] * 5 + [{"p": F, "q": T}] * 5
        # p->!q: step 0-9: F->T = T; step 10-14: T->T = T; step 15-19: F->F = T (since !q=F but !p=T)
        # Actually p->!q = !p | !q. All combos except p=T,q=T give True.
        expected = [T] * 20
        run_trace("H(p -> !q)", trace, expected)

    def test_02_h_impl_violation_mid(self):
        """H(p -> !q): violation at step 10 when p=T, q=T."""
        trace = [{"p": F, "q": F}] * 10 + [{"p": T, "q": T}] + [{"p": F, "q": F}] * 9
        # Steps 0-9: H=T. Step 10: p->!q = !T|!T = F, so H becomes F. Steps 11-19: H stays F.
        expected = [T] * 10 + [F] * 10
        run_trace("H(p -> !q)", trace, expected)

    def test_03_h_impl_violation_step0(self):
        """H(p -> !q): violation at step 0."""
        trace = [{"p": T, "q": T}] + [{"p": F, "q": F}] * 14
        expected = [F] * 15
        run_trace("H(p -> !q)", trace, expected)

    def test_04_h_impl_violation_last_step(self):
        """H(p -> !q): violation only at the very last step of 50."""
        trace = [{"p": F, "q": F}] * 49 + [{"p": T, "q": T}]
        expected = [T] * 49 + [F]
        run_trace("H(p -> !q)", trace, expected)

    def test_05_h_p_implies_q_all_pass(self):
        """H(p -> q): whenever p is true, q must also be true."""
        trace = (
            [{"p": F, "q": F}] * 5
            + [{"p": T, "q": T}] * 5
            + [{"p": F, "q": T}] * 5
            + [{"p": T, "q": T}] * 5
        )
        # p->q = !p|q. All of these satisfy that.
        expected = [T] * 20
        run_trace("H(p -> q)", trace, expected)

    def test_06_h_p_implies_q_violation(self):
        """H(p -> q): p=T but q=F at step 7."""
        trace = (
            [{"p": F, "q": F}] * 7
            + [{"p": T, "q": F}]
            + [{"p": T, "q": T}] * 12
        )
        expected = [T] * 7 + [F] * 13
        run_trace("H(p -> q)", trace, expected)

    def test_07_h_not_p_or_q(self):
        """H(!p | q) is equivalent to H(p -> q)."""
        trace = [{"p": T, "q": T}] * 10 + [{"p": T, "q": F}] + [{"p": F, "q": F}] * 9
        expected = [T] * 10 + [F] * 10
        run_trace("H(!p | q)", trace, expected)

    def test_08_h_conjunction_impl(self):
        """H((a & b) -> !c): if both a and b are true, c must be false."""
        # (a & b) -> !c = !(a & b) | !c = !a | !b | !c
        trace = (
            [{"a": T, "b": F, "c": T}] * 5  # a&b=F so impl=T
            + [{"a": T, "b": T, "c": F}] * 5  # a&b=T, !c=T, impl=T
            + [{"a": T, "b": T, "c": T}]       # a&b=T, !c=F, impl=F => violation
            + [{"a": F, "b": F, "c": F}] * 4
        )
        expected = [T] * 10 + [F] * 5
        run_trace("H((a & b) -> !c)", trace, expected)

    def test_09_h_nested_h(self):
        """H(H(p) -> q): H inside H. Inner H tracks p historically.
        H(p)->q = !H(p)|q. Outer H wraps that.
        Step 0: inner H(p)=p[0]. If p=T then H(p)=T, need q=T.
        """
        # Step 0: p=T => H(p)=T, q=T => T->T=T, outer H=T
        # Step 1: p=T => H(p)=T, q=T => T, outer H=T
        # Step 2: p=F => H(p)=F, q=F => F->F = !F|F = T, outer H=T
        # Step 3: p=T => H(p)=F (already broken), q=F => F->F=T, outer H=T
        # Steps 4-9: p=F,q=F => H(p)=F, F->F=T
        trace = (
            [{"p": T, "q": T}] * 2
            + [{"p": F, "q": F}]
            + [{"p": T, "q": F}]
            + [{"p": F, "q": F}] * 6
        )
        expected = [T] * 10
        run_trace("H(H(p) -> q)", trace, expected)

    def test_10_h_long_trace_pass_100(self):
        """H(p -> !q): 100 events, never violated."""
        trace = [{"p": T, "q": F}] * 50 + [{"p": F, "q": T}] * 50
        expected = [T] * 100
        run_trace("H(p -> !q)", trace, expected)

    def test_11_h_long_trace_violation_at_99(self):
        """H(p -> !q): 100 events, violation at last step (99)."""
        trace = [{"p": F, "q": F}] * 99 + [{"p": T, "q": T}]
        expected = [T] * 99 + [F]
        run_trace("H(p -> !q)", trace, expected)

    def test_12_h_alternating_pass(self):
        """H(p -> !q): alternating p and q but never simultaneously true."""
        # Even steps: p=T, q=F; Odd steps: p=F, q=T
        trace = [{"p": (i % 2 == 0), "q": (i % 2 == 1)} for i in range(20)]
        # p->!q = !p|!q. When p=T,q=F: !q=T => T. When p=F,q=T: !p=T => T.
        expected = [T] * 20
        run_trace("H(p -> !q)", trace, expected)

    def test_13_h_alternating_violation(self):
        """H(p -> !q): alternating, then both true at step 16."""
        trace = [{"p": (i % 2 == 0), "q": (i % 2 == 1)} for i in range(16)]
        trace += [{"p": T, "q": T}]
        trace += [{"p": F, "q": F}] * 3
        expected = [T] * 16 + [F] * 4
        run_trace("H(p -> !q)", trace, expected)

    def test_14_h_true_inner(self):
        """H(true): historically true is always true."""
        trace = [{}] * 20
        expected = [T] * 20
        run_trace("H(true)", trace, expected)

    def test_15_h_false_inner(self):
        """H(false): historically false is false from step 0."""
        trace = [{}] * 15
        expected = [F] * 15
        run_trace("H(false)", trace, expected)

    def test_16_h_pure_prop_all_true(self):
        """H(p): p must be true at every step."""
        trace = [{"p": T}] * 20
        expected = [T] * 20
        run_trace("H(p)", trace, expected)

    def test_17_h_pure_prop_fail_step5(self):
        """H(p): p false at step 5, H becomes irrevocably false."""
        trace = [{"p": T}] * 5 + [{"p": F}] + [{"p": T}] * 14
        expected = [T] * 5 + [F] * 15
        run_trace("H(p)", trace, expected)

    def test_18_h_negation(self):
        """H(!p): p must always be false."""
        trace = [{"p": F}] * 10 + [{"p": T}] + [{"p": F}] * 9
        expected = [T] * 10 + [F] * 10
        run_trace("H(!p)", trace, expected)

    def test_19_h_disjunction(self):
        """H(p | q): at every step, at least one of p,q must be true."""
        trace = (
            [{"p": T, "q": F}] * 5
            + [{"p": F, "q": T}] * 5
            + [{"p": F, "q": F}]  # violation
            + [{"p": T, "q": T}] * 4
        )
        expected = [T] * 10 + [F] * 5
        run_trace("H(p | q)", trace, expected)

    def test_20_h_complex_3props_50steps(self):
        """H((a | b) -> c): if a or b, then c. 50-step trace."""
        # (a|b)->c = !(a|b)|c = (!a & !b) | c
        # Violation when (a|b)=T and c=F, i.e., at least one of a,b true and c false.
        trace = (
            [{"a": F, "b": F, "c": F}] * 20  # a|b=F => impl=T
            + [{"a": T, "b": F, "c": T}] * 15  # a|b=T, c=T => T
            + [{"a": F, "b": T, "c": T}] * 10  # a|b=T, c=T => T
            + [{"a": T, "b": T, "c": F}]        # violation
            + [{"a": F, "b": F, "c": T}] * 4
        )
        expected = [T] * 45 + [F] * 5
        run_trace("H((a | b) -> c)", trace, expected)


# =====================================================================
# Category 2: P() -- Previously (Tests 21-35)
# =====================================================================


class TestPreviously:
    """Tests for P(phi) -- phi held at some past step or now."""

    def test_21_p_latch_basic(self):
        """P(p): false until p becomes true, then latches true."""
        trace = [{"p": F}] * 5 + [{"p": T}] + [{"p": F}] * 9
        # P(p) = p OR prev_P. prev_P starts False.
        # Steps 0-4: F. Step 5: T (p=T). Steps 6-14: T (latched).
        expected = [F] * 5 + [T] * 10
        run_trace("P(p)", trace, expected)

    def test_22_p_true_from_start(self):
        """P(p): p true at step 0 => P(p) true from step 0."""
        trace = [{"p": T}] + [{"p": F}] * 14
        expected = [T] * 15
        run_trace("P(p)", trace, expected)

    def test_23_p_never_true(self):
        """P(p): p never true => P(p) always false."""
        trace = [{"p": F}] * 20
        expected = [F] * 20
        run_trace("P(p)", trace, expected)

    def test_24_h_p_implies_q(self):
        """H(P(p) -> q): once p has occurred, q must always hold after.
        P(p)->q = !P(p)|q.
        Before p occurs: P(p)=F so impl=T.
        After p occurs: P(p)=T so need q=T.
        H wraps: once the implication fails, stays false.
        """
        trace = (
            [{"p": F, "q": F}] * 5   # P(p)=F, impl=T, H=T
            + [{"p": T, "q": T}]      # P(p)=T, q=T, impl=T, H=T
            + [{"p": F, "q": T}] * 4  # P(p)=T, q=T, impl=T, H=T
            + [{"p": F, "q": F}]      # P(p)=T, q=F, impl=F, H=F
            + [{"p": F, "q": T}] * 4  # H stays F
        )
        expected = [T] * 10 + [F] * 5
        run_trace("H(P(p) -> q)", trace, expected)

    def test_25_h_p_implies_q_never_triggered(self):
        """H(P(p) -> q): p never occurs, so P(p) always false, impl always true."""
        trace = [{"p": F, "q": F}] * 20
        expected = [T] * 20
        run_trace("H(P(p) -> q)", trace, expected)

    def test_26_p_p_and_p_q(self):
        """P(p) & P(q): both must have occurred at some point."""
        trace = (
            [{"p": F, "q": F}] * 3
            + [{"p": T, "q": F}]       # P(p)=T, P(q)=F => F
            + [{"p": F, "q": F}] * 3
            + [{"p": F, "q": T}]       # P(p)=T, P(q)=T => T
            + [{"p": F, "q": F}] * 7   # both latched T => T
        )
        expected = [F] * 7 + [T] * 8
        run_trace("P(p) & P(q)", trace, expected)

    def test_27_p_negation(self):
        """P(!p): true once p has been false at some step.
        !p is True when p is False. P(!p) latches once that happens.
        """
        trace = [{"p": T}] * 5 + [{"p": F}] + [{"p": T}] * 9
        # !p: F,F,F,F,F,T,F,...  P(!p): F,F,F,F,F,T,T,...
        expected = [F] * 5 + [T] * 10
        run_trace("P(!p)", trace, expected)

    def test_28_p_latch_permanence_100_steps(self):
        """P(p): p true at step 3, verify latch holds for 100 steps."""
        trace = [{"p": F}] * 3 + [{"p": T}] + [{"p": F}] * 96
        expected = [F] * 3 + [T] * 97
        run_trace("P(p)", trace, expected)

    def test_29_p_disjunction(self):
        """P(p | q): true once either p or q has been true."""
        trace = (
            [{"p": F, "q": F}] * 4
            + [{"p": F, "q": T}]
            + [{"p": F, "q": F}] * 10
        )
        # p|q: F,F,F,F,T,F,...  P(p|q): F,F,F,F,T,T,...
        expected = [F] * 4 + [T] * 11
        run_trace("P(p | q)", trace, expected)

    def test_30_p_conjunction(self):
        """P(p & q): true once p and q are simultaneously true."""
        trace = (
            [{"p": T, "q": F}] * 3
            + [{"p": F, "q": T}] * 3
            + [{"p": T, "q": T}]
            + [{"p": F, "q": F}] * 8
        )
        # p&q: F,F,F,F,F,F,T,F,...  P(p&q): F,...,F,T,T,...
        expected = [F] * 6 + [T] * 9
        run_trace("P(p & q)", trace, expected)

    def test_31_h_p_p_implies_not_q_long(self):
        """H(P(p) -> !q): once p occurred, q must never be true. 50 steps."""
        trace = (
            [{"p": F, "q": T}] * 10  # P(p)=F, impl=T, H=T
            + [{"p": T, "q": F}]      # P(p)=T, !q=T, impl=T, H=T
            + [{"p": F, "q": F}] * 20 # P(p)=T, !q=T, H=T
            + [{"p": F, "q": T}]      # P(p)=T, !q=F, impl=F, H=F
            + [{"p": F, "q": F}] * 18
        )
        expected = [T] * 31 + [F] * 19
        run_trace("H(P(p) -> !q)", trace, expected)

    def test_32_p_implication(self):
        """P(p -> q): true once there was a step where p->q held (i.e., !p|q)."""
        # p->q = !p|q. At step 0: p=T, q=F => !T|F=F. So P starts F.
        # Step 1: p=T, q=F => F again. Step 2: p=F, q=F => !F|F=T => P latches.
        trace = (
            [{"p": T, "q": F}] * 2
            + [{"p": F, "q": F}]
            + [{"p": T, "q": F}] * 7
        )
        expected = [F, F, T, T, T, T, T, T, T, T]
        run_trace("P(p -> q)", trace, expected)

    def test_33_p_true_literal(self):
        """P(true): true from step 0 since 'true' is always true."""
        trace = [{}] * 15
        expected = [T] * 15
        run_trace("P(true)", trace, expected)

    def test_34_p_false_literal(self):
        """P(false): false always since 'false' is never true."""
        trace = [{}] * 15
        expected = [F] * 15
        run_trace("P(false)", trace, expected)

    def test_35_p_p_at_last_step(self):
        """P(p): p only becomes true at the very last step of 20."""
        trace = [{"p": F}] * 19 + [{"p": T}]
        expected = [F] * 19 + [T]
        run_trace("P(p)", trace, expected)


# =====================================================================
# Category 3: Y() -- Yesterday (Tests 36-50)
# =====================================================================


class TestYesterday:
    """Tests for Y(phi) -- phi held at the previous step."""

    def test_36_y_false_at_step0(self):
        """Y(p): always false at step 0 regardless of p."""
        trace = [{"p": T}] + [{"p": F}] * 9
        # Y(p): step0=F (no previous), step1=prev(p)=T, step2=prev(p)=F,...
        expected = [F, T, F, F, F, F, F, F, F, F]
        run_trace("Y(p)", trace, expected)

    def test_37_y_tracks_previous(self):
        """Y(p): output at step i is p's value at step i-1."""
        trace = [{"p": T}, {"p": F}, {"p": T}, {"p": T}, {"p": F},
                 {"p": F}, {"p": T}, {"p": F}, {"p": T}, {"p": F}]
        # Y(p) = [F, T, F, T, T, F, F, T, F, T]
        expected = [F, T, F, T, T, F, F, T, F, T]
        run_trace("Y(p)", trace, expected)

    def test_38_h_y_implies_q(self):
        """H(Y(p) -> q): if p was true at the previous step, q must be true now.
        At step 0: Y(p)=F, so impl=T.
        At step i>0: Y(p)=p[i-1]. If p[i-1]=T, need q[i]=T.
        """
        trace = [
            {"p": T, "q": T},   # step 0: Y(p)=F, impl=T, H=T
            {"p": F, "q": T},   # step 1: Y(p)=T, q=T, impl=T, H=T
            {"p": T, "q": F},   # step 2: Y(p)=F, impl=T, H=T
            {"p": F, "q": F},   # step 3: Y(p)=T, q=F, impl=F, H=F
            {"p": F, "q": T},   # step 4: H stays F
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("H(Y(p) -> q)", trace, expected)

    def test_39_h_y_implies_q_all_pass(self):
        """H(Y(p) -> q): every time p was true at prev step, q is true now."""
        trace = [
            {"p": T, "q": T},   # step 0: Y(p)=F, impl=T
            {"p": F, "q": T},   # step 1: Y(p)=T, q=T => T
            {"p": T, "q": T},   # step 2: Y(p)=F, impl=T
            {"p": F, "q": T},   # step 3: Y(p)=T, q=T => T
            {"p": T, "q": T},   # step 4: Y(p)=F, impl=T
            {"p": F, "q": T},   # step 5: Y(p)=T, q=T => T
            {"p": F, "q": F},   # step 6: Y(p)=F, impl=T
            {"p": F, "q": F},   # step 7: Y(p)=F, impl=T
            {"p": F, "q": F},   # step 8: Y(p)=F, impl=T
            {"p": F, "q": F},   # step 9: Y(p)=F, impl=T
        ]
        expected = [T] * 10
        run_trace("H(Y(p) -> q)", trace, expected)

    def test_40_y_y_nested(self):
        """Y(Y(p)): two steps ago. False at steps 0 and 1."""
        trace = [{"p": T}, {"p": F}, {"p": T}, {"p": T}, {"p": F},
                 {"p": T}, {"p": F}, {"p": T}, {"p": F}, {"p": T}]
        # Y(p) =  [F,    T,     F,     T,     T,     F,     T,     F,     T,     F]
        # Y(Y(p))=[F,    F,     T,     F,     T,     T,     F,     T,     F,     T]
        expected = [F, F, T, F, T, T, F, T, F, T]
        run_trace("Y(Y(p))", trace, expected)

    def test_41_y_y_y_nested(self):
        """Y(Y(Y(p))): three steps ago. False at steps 0, 1, 2."""
        trace = [{"p": T}, {"p": F}, {"p": T}, {"p": T}, {"p": F},
                 {"p": T}, {"p": F}, {"p": T}, {"p": F}, {"p": T},
                 {"p": T}, {"p": F}]
        # p    = [T, F, T, T, F, T, F, T, F, T, T, F]
        # Y(p) = [F, T, F, T, T, F, T, F, T, F, T, T]
        # YY   = [F, F, T, F, T, T, F, T, F, T, F, T]
        # YYY  = [F, F, F, T, F, T, T, F, T, F, T, F]
        expected = [F, F, F, T, F, T, T, F, T, F, T, F]
        run_trace("Y(Y(Y(p)))", trace, expected)

    def test_42_h_p_implies_y_q(self):
        """H(p -> Y(q)): p requires q to have been true at the previous step.
        At step 0: Y(q)=F. If p=T, then p->F = F => H=F immediately.
        """
        trace = [
            {"p": F, "q": T},  # step 0: p=F, impl=T (vacuous), H=T
            {"p": T, "q": T},  # step 1: p=T, Y(q)=T(q[0]=T), impl=T, H=T
            {"p": T, "q": F},  # step 2: p=T, Y(q)=T(q[1]=T), impl=T, H=T
            {"p": T, "q": T},  # step 3: p=T, Y(q)=F(q[2]=F), impl=F, H=F
            {"p": F, "q": F},  # step 4: H stays F
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("H(p -> Y(q))", trace, expected)

    def test_43_y_constant_true(self):
        """Y(true): false at step 0, true thereafter (since true held yesterday)."""
        trace = [{}] * 15
        expected = [F] + [T] * 14
        run_trace("Y(true)", trace, expected)

    def test_44_y_constant_false(self):
        """Y(false): always false (false never held yesterday)."""
        trace = [{}] * 15
        expected = [F] * 15
        run_trace("Y(false)", trace, expected)

    def test_45_y_negation(self):
        """Y(!p): was p false at the previous step?"""
        trace = [{"p": T}, {"p": T}, {"p": F}, {"p": F}, {"p": T},
                 {"p": F}, {"p": T}, {"p": F}, {"p": T}, {"p": T}]
        # !p   = [F, F, T, T, F, T, F, T, F, F]
        # Y(!p)= [F, F, F, T, T, F, T, F, T, F]
        expected = [F, F, F, T, T, F, T, F, T, F]
        run_trace("Y(!p)", trace, expected)

    def test_46_y_conjunction(self):
        """Y(p & q): were both p and q true at the previous step?"""
        trace = [
            {"p": T, "q": T},  # step 0
            {"p": T, "q": F},  # step 1
            {"p": F, "q": T},  # step 2
            {"p": T, "q": T},  # step 3
            {"p": T, "q": T},  # step 4
            {"p": F, "q": F},  # step 5
            {"p": T, "q": T},  # step 6
            {"p": T, "q": T},  # step 7
            {"p": F, "q": F},  # step 8
            {"p": F, "q": F},  # step 9
        ]
        # p&q = [T, F, F, T, T, F, T, T, F, F]
        # Y(p&q)=[F, T, F, F, T, T, F, T, T, F]
        expected = [F, T, F, F, T, T, F, T, T, F]
        run_trace("Y(p & q)", trace, expected)

    def test_47_h_y_p_implies_y_q(self):
        """H(Y(p) -> Y(q)): if p was true yesterday, q was also true yesterday.
        Equivalently, at every step, if p[i-1]=T then q[i-1]=T.
        """
        trace = [
            {"p": T, "q": T},  # step 0: Y(p)=F, Y(q)=F, F->F=T, H=T
            {"p": F, "q": T},  # step 1: Y(p)=T, Y(q)=T, T->T=T, H=T
            {"p": T, "q": F},  # step 2: Y(p)=F, Y(q)=T, F->T=T, H=T
            {"p": F, "q": T},  # step 3: Y(p)=T, Y(q)=F, T->F=F, H=F
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
            {"p": F, "q": F},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("H(Y(p) -> Y(q))", trace, expected)

    def test_48_y_oscillating(self):
        """Y(p) with p oscillating T/F over 20 steps."""
        trace = [{"p": (i % 2 == 0)} for i in range(20)]
        # p   = [T, F, T, F, T, F, T, F, T, F, T, F, T, F, T, F, T, F, T, F]
        # Y(p)= [F, T, F, T, F, T, F, T, F, T, F, T, F, T, F, T, F, T, F, T]
        expected = [F] + [(i % 2 == 0) for i in range(19)]
        # i=0: F, i=1: T(p[0]=T), i=2: F(p[1]=F), i=3: T(p[2]=T),...
        run_trace("Y(p)", trace, expected)

    def test_49_y_all_false(self):
        """Y(p) where p is always false => Y(p) always false."""
        trace = [{"p": F}] * 15
        expected = [F] * 15
        run_trace("Y(p)", trace, expected)

    def test_50_y_all_true(self):
        """Y(p) where p is always true => Y(p) is F at step 0, then T."""
        trace = [{"p": T}] * 15
        expected = [F] + [T] * 14
        run_trace("Y(p)", trace, expected)


# =====================================================================
# Category 4: S -- Since (Tests 51-65)
# =====================================================================


class TestSince:
    """Tests for phi S psi -- psi triggered and phi held since."""

    def test_51_since_basic_trigger(self):
        """p S q: q triggers at step 3, p holds after.
        now(p S q) = q OR (p AND prev(p S q)). prev starts F.
        """
        trace = [
            {"p": T, "q": F},  # step 0: F OR (T AND F) = F
            {"p": T, "q": F},  # step 1: F OR (T AND F) = F
            {"p": F, "q": F},  # step 2: F OR (F AND F) = F
            {"p": T, "q": T},  # step 3: T OR (T AND F) = T
            {"p": T, "q": F},  # step 4: F OR (T AND T) = T
            {"p": T, "q": F},  # step 5: F OR (T AND T) = T
            {"p": F, "q": F},  # step 6: F OR (F AND T) = F  (p broke the chain)
            {"p": T, "q": F},  # step 7: F OR (T AND F) = F
            {"p": T, "q": F},  # step 8: F OR (T AND F) = F
            {"p": T, "q": F},  # step 9: F OR (T AND F) = F
        ]
        expected = [F, F, F, T, T, T, F, F, F, F]
        run_trace("p S q", trace, expected)

    def test_52_since_trigger_at_step0(self):
        """p S q: q triggers at step 0."""
        trace = [
            {"p": T, "q": T},  # step 0: T OR (T AND F) = T
            {"p": T, "q": F},  # step 1: F OR (T AND T) = T
            {"p": T, "q": F},  # step 2: F OR (T AND T) = T
            {"p": F, "q": F},  # step 3: F OR (F AND T) = F
            {"p": T, "q": F},  # step 4: F OR (T AND F) = F
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("p S q", trace, expected)

    def test_53_since_never_triggered(self):
        """p S q: q never true => since never true."""
        trace = [{"p": T, "q": F}] * 15
        expected = [F] * 15
        run_trace("p S q", trace, expected)

    def test_54_since_retrigger(self):
        """p S q: q triggers, chain breaks, q triggers again."""
        trace = [
            {"p": T, "q": T},  # step 0: T
            {"p": T, "q": F},  # step 1: F|(T&T)=T
            {"p": F, "q": F},  # step 2: F|(F&T)=F  chain broken
            {"p": T, "q": F},  # step 3: F|(T&F)=F
            {"p": T, "q": T},  # step 4: T  retrigger
            {"p": T, "q": F},  # step 5: F|(T&T)=T
            {"p": T, "q": F},  # step 6: F|(T&T)=T
            {"p": T, "q": F},  # step 7: F|(T&T)=T
            {"p": T, "q": F},  # step 8: F|(T&T)=T
            {"p": T, "q": F},  # step 9: F|(T&T)=T
        ]
        expected = [T, T, F, F, T, T, T, T, T, T]
        run_trace("p S q", trace, expected)

    def test_55_since_q_always_true(self):
        """p S q: q always true => since always true (q fires every step)."""
        trace = [{"p": F, "q": T}] * 12
        expected = [T] * 12
        run_trace("p S q", trace, expected)

    def test_56_since_with_negation(self):
        """(!p) S q: after q triggers, p must stay false."""
        trace = [
            {"p": F, "q": F},  # step 0: F OR (!F AND F) = F OR (T AND F) = F
            {"p": F, "q": T},  # step 1: T
            {"p": F, "q": F},  # step 2: F OR (T AND T) = T
            {"p": T, "q": F},  # step 3: F OR (F AND T) = F  (p broke it, !p=F)
            {"p": F, "q": F},  # step 4: F OR (T AND F) = F
            {"p": F, "q": T},  # step 5: T  retrigger
            {"p": F, "q": F},  # step 6: F OR (T AND T) = T
            {"p": F, "q": F},  # step 7: F OR (T AND T) = T
            {"p": F, "q": F},  # step 8: F OR (T AND T) = T
            {"p": F, "q": F},  # step 9: F OR (T AND T) = T
        ]
        expected = [F, T, T, F, F, T, T, T, T, T]
        run_trace("(!p) S q", trace, expected)

    def test_57_h_since_implies_r(self):
        """H((p S q) -> r): whenever since-condition holds, r must be true."""
        trace = [
            {"p": T, "q": F, "r": T},   # step 0: pSq=F, impl=T, H=T
            {"p": T, "q": T, "r": T},   # step 1: pSq=T, r=T, impl=T, H=T
            {"p": T, "q": F, "r": T},   # step 2: pSq=T, r=T, impl=T, H=T
            {"p": T, "q": F, "r": F},   # step 3: pSq=T, r=F, impl=F, H=F
            {"p": F, "q": F, "r": T},   # step 4: H stays F
            {"p": F, "q": F, "r": T},
            {"p": F, "q": F, "r": T},
            {"p": F, "q": F, "r": T},
            {"p": F, "q": F, "r": T},
            {"p": F, "q": F, "r": T},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("H((p S q) -> r)", trace, expected)

    def test_58_since_immediate_break(self):
        """p S q: q triggers but p is immediately false at same step?
        Since = q OR (p AND prev). If q=T, since=T regardless of p.
        """
        trace = [
            {"p": F, "q": T},  # step 0: T (q fires)
            {"p": F, "q": F},  # step 1: F|(F&T)=F  (p=F breaks)
            {"p": T, "q": F},  # step 2: F|(T&F)=F
            {"p": T, "q": F},  # step 3: F|(T&F)=F
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
            {"p": T, "q": F},
        ]
        expected = [T, F, F, F, F, F, F, F, F, F]
        run_trace("p S q", trace, expected)

    def test_59_since_long_chain_50(self):
        """p S q: q at step 0, p holds for 49 more steps."""
        trace = [{"p": T, "q": T}] + [{"p": T, "q": F}] * 49
        expected = [T] * 50
        run_trace("p S q", trace, expected)

    def test_60_since_long_chain_break_at_end(self):
        """p S q: q at step 0, p holds for 48 steps, breaks at step 49."""
        trace = [{"p": T, "q": T}] + [{"p": T, "q": F}] * 48 + [{"p": F, "q": F}]
        expected = [T] * 49 + [F]
        run_trace("p S q", trace, expected)

    def test_61_since_multiple_triggers(self):
        """p S q: q triggers repeatedly, keeping since alive."""
        trace = [
            {"p": F, "q": T},  # T
            {"p": F, "q": T},  # T
            {"p": F, "q": T},  # T
            {"p": F, "q": F},  # F|(F&T) = F
            {"p": F, "q": F},  # F
            {"p": F, "q": T},  # T
            {"p": T, "q": F},  # F|(T&T) = T
            {"p": T, "q": F},  # F|(T&T) = T
            {"p": T, "q": F},  # F|(T&T) = T
            {"p": F, "q": F},  # F|(F&T) = F
        ]
        expected = [T, T, T, F, F, T, T, T, T, F]
        run_trace("p S q", trace, expected)

    def test_62_since_p_and_q_same(self):
        """p S p: equivalent to P(p) -- p has occurred at some point and p holds continuously since.
        Actually, p S p = p OR (p AND prev(p S p)).
        If p=T: T. If p=F: F OR (F AND prev) = F.
        So p S p = p. Wait, let me think more carefully.
        step 0: p=T => T||(T&&F) = T. p=F => F||(F&&F) = F.
        step 1: if p[0]=T, p[1]=T => T. p[1]=F => F||(F&&T)=F.
        step 1: if p[0]=T, p[1]=F => F. So p S p = p.
        """
        trace = [{"p": T}, {"p": F}, {"p": T}, {"p": F}, {"p": T},
                 {"p": T}, {"p": F}, {"p": T}, {"p": T}, {"p": T}]
        expected = [T, F, T, F, T, T, F, T, T, T]
        run_trace("p S p", trace, expected)

    def test_63_since_with_disjunction_trigger(self):
        """p S (q | r): triggered by either q or r."""
        trace = [
            {"p": T, "q": F, "r": F},  # step 0: q|r=F, F|(T&F)=F
            {"p": T, "q": F, "r": T},  # step 1: q|r=T => T
            {"p": T, "q": F, "r": F},  # step 2: F|(T&T)=T
            {"p": T, "q": F, "r": F},  # step 3: F|(T&T)=T
            {"p": F, "q": F, "r": F},  # step 4: F|(F&T)=F
            {"p": T, "q": T, "r": F},  # step 5: q|r=T => T
            {"p": T, "q": F, "r": F},  # step 6: F|(T&T)=T
            {"p": T, "q": F, "r": F},  # step 7: F|(T&T)=T
            {"p": T, "q": F, "r": F},  # step 8: F|(T&T)=T
            {"p": T, "q": F, "r": F},  # step 9: F|(T&T)=T
        ]
        expected = [F, T, T, T, F, T, T, T, T, T]
        run_trace("p S (q | r)", trace, expected)

    def test_64_since_false_trigger(self):
        """p S false: false is never true, so since is always false."""
        trace = [{"p": T}] * 15
        expected = [F] * 15
        run_trace("p S false", trace, expected)

    def test_65_since_true_trigger(self):
        """p S true: true fires every step, so since = true OR (p AND prev).
        Since true fires every step, result is always true.
        """
        trace = [{"p": F}] * 15
        expected = [T] * 15
        run_trace("p S true", trace, expected)


# =====================================================================
# Category 5: Combined operators (Tests 66-85)
# =====================================================================


class TestCombined:
    """Tests combining multiple temporal operators."""

    def test_66_h_y_and_h_p(self):
        """H(Y(p) -> q) & H(P(r) -> !s): conjunction of two policies.
        Policy A: if p was true yesterday, q must be true.
        Policy B: once r occurred, s must never be true.
        """
        trace = [
            {"p": F, "q": F, "r": F, "s": F},  # A: Y(p)=F,impl=T,H=T. B: P(r)=F,impl=T,H=T. T&T=T
            {"p": T, "q": T, "r": F, "s": F},  # A: Y(p)=F,impl=T,H=T. B: P(r)=F,impl=T,H=T. T
            {"p": F, "q": T, "r": T, "s": F},  # A: Y(p)=T,q=T,T,H=T. B: P(r)=T,!s=T,T,H=T. T
            {"p": F, "q": F, "r": F, "s": F},  # A: Y(p)=F,impl=T,H=T. B: P(r)=T,!s=T,T,H=T. T
            {"p": F, "q": F, "r": F, "s": T},  # A: Y(p)=F,impl=T,H=T. B: P(r)=T,!s=F,F,H=F. T&F=F
            {"p": F, "q": F, "r": F, "s": F},  # B stays F. F
            {"p": F, "q": F, "r": F, "s": F},
            {"p": F, "q": F, "r": F, "s": F},
            {"p": F, "q": F, "r": F, "s": F},
            {"p": F, "q": F, "r": F, "s": F},
        ]
        expected = [T, T, T, T, F, F, F, F, F, F]
        run_trace("H(Y(p) -> q) & H(P(r) -> !s)", trace, expected)

    def test_67_h_y_and_h_p_first_fails(self):
        """H(Y(p) -> q) & H(P(r) -> !s): first policy fails."""
        trace = [
            {"p": T, "q": T, "r": F, "s": F},  # A: Y(p)=F,T. B: P(r)=F,T. T
            {"p": F, "q": F, "r": F, "s": F},  # A: Y(p)=T,q=F => F, H=F. B: T. F&T=F
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
            {"p": F, "q": T, "r": F, "s": F},
        ]
        expected = [T, F, F, F, F, F, F, F, F, F]
        run_trace("H(Y(p) -> q) & H(P(r) -> !s)", trace, expected)

    def test_68_jailbreak_pattern(self):
        """H((p & P(frame)) -> !unsafe): if p occurs and frame was seen before, unsafe must not happen.
        This models the jailbreak detection pattern from the paper.
        """
        trace = [
            {"p": F, "frame": F, "unsafe": F},   # P(frame)=F, p&P(f)=F, impl=T, H=T
            {"p": T, "frame": F, "unsafe": T},    # P(frame)=F, p&F=F, impl=T, H=T
            {"p": F, "frame": T, "unsafe": F},    # P(frame)=T, p&T=F, impl=T, H=T
            {"p": F, "frame": F, "unsafe": F},    # P(frame)=T, p=F, F, impl=T, H=T
            {"p": T, "frame": F, "unsafe": F},    # P(frame)=T, p&T=T, !unsafe=T, impl=T, H=T
            {"p": T, "frame": F, "unsafe": T},    # P(frame)=T, p&T=T, !unsafe=F, impl=F, H=F
            {"p": F, "frame": F, "unsafe": F},    # H stays F
            {"p": F, "frame": F, "unsafe": F},
            {"p": F, "frame": F, "unsafe": F},
            {"p": F, "frame": F, "unsafe": F},
        ]
        expected = [T, T, T, T, T, F, F, F, F, F]
        run_trace("H((p & P(frame)) -> !unsafe)", trace, expected)

    def test_69_jailbreak_no_frame(self):
        """H((p & P(frame)) -> !unsafe): frame never occurs so antecedent is always false."""
        trace = [
            {"p": T, "frame": F, "unsafe": T},
            {"p": T, "frame": F, "unsafe": T},
            {"p": T, "frame": F, "unsafe": T},
        ] * 5  # 15 steps
        # P(frame) never becomes true, so p & P(frame) = F, impl = T always.
        expected = [T] * 15
        run_trace("H((p & P(frame)) -> !unsafe)", trace, expected)

    def test_70_sensitive_data_pattern(self):
        """H(P(sensitive) -> !echo) & H(Y(sensitive) -> warn):
        Once sensitive data appears, echo must never happen.
        Immediately after sensitive data, a warning must follow.
        """
        trace = [
            {"sensitive": F, "echo": F, "warn": F},  # P(s)=F,T; Y(s)=F,T; T&T=T
            {"sensitive": T, "echo": F, "warn": F},   # P(s)=T,!echo=T,T; Y(s)=F,T; T&T=T
            {"sensitive": F, "echo": F, "warn": T},   # P(s)=T,!echo=T,T; Y(s)=T,warn=T,T; T&T=T
            {"sensitive": F, "echo": F, "warn": F},   # P(s)=T,T; Y(s)=F,T; T&T=T
            {"sensitive": F, "echo": T, "warn": F},   # P(s)=T,!echo=F => H1=F; Y(s)=F,T; F&T=F
            {"sensitive": F, "echo": F, "warn": F},
            {"sensitive": F, "echo": F, "warn": F},
            {"sensitive": F, "echo": F, "warn": F},
            {"sensitive": F, "echo": F, "warn": F},
            {"sensitive": F, "echo": F, "warn": F},
        ]
        expected = [T, T, T, T, F, F, F, F, F, F]
        run_trace("H(P(sensitive) -> !echo) & H(Y(sensitive) -> warn)", trace, expected)

    def test_71_sensitive_warn_failure(self):
        """H(P(sensitive) -> !echo) & H(Y(sensitive) -> warn): warn missing after sensitive."""
        trace = [
            {"sensitive": T, "echo": F, "warn": F},  # P(s)=T,!echo=T,H1=T; Y(s)=F,impl=T,H2=T; T
            {"sensitive": F, "echo": F, "warn": F},   # P(s)=T,!echo=T,H1=T; Y(s)=T,warn=F,F,H2=F; T&F=F
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
            {"sensitive": F, "echo": F, "warn": T},
        ]
        expected = [T, F, F, F, F, F, F, F, F, F]
        run_trace("H(P(sensitive) -> !echo) & H(Y(sensitive) -> warn)", trace, expected)

    def test_72_combined_y_and_p(self):
        """Y(p) & P(q): yesterday p was true AND q has occurred at some point."""
        trace = [
            {"p": T, "q": F},  # Y(p)=F, P(q)=F, F&F=F
            {"p": F, "q": T},  # Y(p)=T, P(q)=T, T&T=T
            {"p": T, "q": F},  # Y(p)=F, P(q)=T, F&T=F
            {"p": T, "q": F},  # Y(p)=T, P(q)=T, T&T=T
            {"p": F, "q": F},  # Y(p)=T, P(q)=T, T&T=T
            {"p": F, "q": F},  # Y(p)=F, P(q)=T, F&T=F
            {"p": T, "q": F},  # Y(p)=F, P(q)=T, F&T=F
            {"p": F, "q": F},  # Y(p)=T, P(q)=T, T&T=T
            {"p": F, "q": F},  # Y(p)=F, P(q)=T, F&T=F
            {"p": F, "q": F},  # Y(p)=F, P(q)=T, F&T=F
        ]
        expected = [F, T, F, T, T, F, F, T, F, F]
        run_trace("Y(p) & P(q)", trace, expected)

    def test_73_h_p_or_q(self):
        """H(p | q): at every step at least one must be true. 30 steps."""
        trace = (
            [{"p": T, "q": F}] * 10
            + [{"p": F, "q": T}] * 10
            + [{"p": T, "q": T}] * 10
        )
        expected = [T] * 30
        run_trace("H(p | q)", trace, expected)

    def test_74_combined_since_and_h(self):
        """H(p S q): historically, the since-condition has held.
        H wraps pSq. pSq can oscillate, H will latch false on first F.
        pSq at step 0: q OR (p AND F) = q.
        """
        trace = [
            {"p": T, "q": T},  # pSq=T, H=T
            {"p": T, "q": F},  # pSq=F|(T&T)=T, H=T
            {"p": T, "q": F},  # pSq=F|(T&T)=T, H=T
            {"p": F, "q": F},  # pSq=F|(F&T)=F, H=F
            {"p": T, "q": T},  # H stays F
            {"p": T, "q": T},
            {"p": T, "q": T},
            {"p": T, "q": T},
            {"p": T, "q": T},
            {"p": T, "q": T},
        ]
        expected = [T, T, T, F, F, F, F, F, F, F]
        run_trace("H(p S q)", trace, expected)

    def test_75_p_y_combined(self):
        """P(Y(p)): has p ever been true at a step's previous step?
        P(Y(p)) = Y(p) OR prev_P(Y(p)).
        Y(p) at step 0 = F. So P(Y(p)) at step 0 = F.
        Y(p) at step 1 = p[0]. If p[0]=T, then P(Y(p)) latches T at step 1.
        """
        trace = [
            {"p": F},  # step 0: Y(p)=F, P=F
            {"p": F},  # step 1: Y(p)=F, P=F
            {"p": T},  # step 2: Y(p)=F, P=F
            {"p": F},  # step 3: Y(p)=T, P=T (latches)
            {"p": F},  # step 4: Y(p)=F, P=T
            {"p": F},  # step 5: P=T
            {"p": F},
            {"p": F},
            {"p": F},
            {"p": F},
        ]
        expected = [F, F, F, T, T, T, T, T, T, T]
        run_trace("P(Y(p))", trace, expected)

    def test_76_h_not_p_and_not_q(self):
        """H(!p & !q): both p and q must always be false."""
        trace = [{"p": F, "q": F}] * 12 + [{"p": T, "q": F}] + [{"p": F, "q": F}] * 2
        expected = [T] * 12 + [F] * 3
        run_trace("H(!p & !q)", trace, expected)

    def test_77_p_implies_h(self):
        """P(p) -> H(q): if p ever occurred, then q must have always held.
        !P(p) | H(q). This can oscillate.
        """
        trace = [
            {"p": F, "q": F},  # P(p)=F, H(q)=F(q=F,prev=T => F). !F|F=T
            {"p": F, "q": T},  # P(p)=F, H(q)=F(prev was F). !F|F = T
            {"p": T, "q": T},  # P(p)=T, H(q)=F. !T|F = F
            {"p": F, "q": T},  # P(p)=T, H(q)=F. F
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
            {"p": F, "q": T},
        ]
        expected = [T, T, F, F, F, F, F, F, F, F]
        run_trace("P(p) -> H(q)", trace, expected)

    def test_78_double_h(self):
        """H(H(p)): equivalent to H(p) since H is idempotent.
        Inner H(p) = p AND prev_inner. Outer H = inner AND prev_outer.
        Both start with prev=T.
        Step 0: inner = p AND T = p. outer = inner AND T = p.
        Step 1: inner = p AND prev_inner. outer = inner AND prev_outer.
        So H(H(p)) = H(p).
        """
        trace = [{"p": T}] * 8 + [{"p": F}] + [{"p": T}] * 6
        expected = [T] * 8 + [F] * 7
        run_trace("H(H(p))", trace, expected)

    def test_79_complex_multi_policy_50_steps(self):
        """H(a -> !b) & H(P(c) -> d): two policies, 50 steps.
        Policy 1: if a then not b.
        Policy 2: once c occurred, d must always hold.
        """
        trace = (
            [{"a": F, "b": F, "c": F, "d": F}] * 10   # P1:T, P2:T (P(c)=F)
            + [{"a": T, "b": F, "c": F, "d": F}] * 10  # P1:T, P2:T (P(c)=F)
            + [{"a": F, "b": F, "c": T, "d": T}] * 5   # P1:T, P2:T (P(c)=T, d=T)
            + [{"a": F, "b": F, "c": F, "d": T}] * 5   # P1:T, P2:T
            + [{"a": F, "b": F, "c": F, "d": T}] * 10  # P1:T, P2:T
            + [{"a": T, "b": T, "c": F, "d": T}]        # P1:F (a&b both T), P2:T. F&T=F
            + [{"a": F, "b": F, "c": F, "d": T}] * 9
        )
        expected = [T] * 40 + [F] * 10
        run_trace("H(a -> !b) & H(P(c) -> d)", trace, expected)

    def test_80_y_since_combined(self):
        """Y(p S q): the since-condition one step ago."""
        trace = [
            {"p": T, "q": T},  # pSq=T
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T (prev pSq=T)
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T
            {"p": F, "q": F},  # pSq=F; Y(pSq)=T (prev pSq was T at step 2)
            {"p": T, "q": F},  # pSq=F; Y(pSq)=F (prev pSq was F at step 3)
            {"p": T, "q": T},  # pSq=T; Y(pSq)=F (prev pSq was F at step 4)
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T (prev pSq was T at step 5)
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T
            {"p": T, "q": F},  # pSq=T; Y(pSq)=T
        ]
        # Y(pSq): step 0 = F (no prev), then tracks previous pSq value.
        # pSq: [T, T, T, F, F, T, T, T, T, T]
        # Y:   [F, T, T, T, F, F, T, T, T, T]
        expected = [F, T, T, T, F, F, T, T, T, T]
        run_trace("Y(p S q)", trace, expected)

    def test_81_h_p_y_combined(self):
        """H(P(Y(p) -> !q)): deeply combined. P(Y(p)->!q) latches once (Y(p)->!q) is true.
        Y(p)->!q = !Y(p)|!q.
        At step 0: Y(p)=F, so !Y(p)=T, impl=T. P latches T.
        So P(Y(p)->!q) = T from step 0 onward.
        H(T) = T always.
        """
        trace = [{"p": T, "q": T}] * 15
        # Step 0: Y(p)=F, !Y(p)=T, !q=F, !Y(p)|!q=T. P=T. H=T.
        # Step 1: Y(p)=T, !Y(p)=F, !q=F, F|F=F. But P=T|T=T. H=T&T=T.
        # From step 0 onward, P latched to T, so H stays T.
        expected = [T] * 15
        run_trace("H(P(Y(p) -> !q))", trace, expected)

    def test_82_three_policy_conjunction(self):
        """H(!a) & H(!b) & H(!c): none of a, b, c must ever be true.
        Note: parser should handle this as (H(!a) & H(!b)) & H(!c).
        """
        trace = (
            [{"a": F, "b": F, "c": F}] * 10
            + [{"a": F, "b": T, "c": F}]  # b=T => H(!b)=F
            + [{"a": F, "b": F, "c": F}] * 4
        )
        expected = [T] * 10 + [F] * 5
        run_trace("H(!a) & H(!b) & H(!c)", trace, expected)

    def test_83_implication_chain(self):
        """H(p -> (q -> r)): if p then (if q then r).
        p->(q->r) = !p | (!q | r) = !p | !q | r.
        Violation only when p=T, q=T, r=F.
        """
        trace = (
            [{"p": T, "q": T, "r": T}] * 5
            + [{"p": T, "q": F, "r": F}] * 5  # !q=T => T
            + [{"p": F, "q": T, "r": F}] * 5  # !p=T => T
            + [{"p": T, "q": T, "r": F}]       # violation
            + [{"p": F, "q": F, "r": F}] * 4
        )
        expected = [T] * 15 + [F] * 5
        run_trace("H(p -> (q -> r))", trace, expected)

    def test_84_since_with_h_guard(self):
        """H((p S q) -> H(r)): if since-condition holds, r must have always been true.
        This is complex: H(r) is inner H, evaluated per step.
        """
        # Keep it simple: r always true for first 10 steps.
        # pSq becomes true at step 2. H(r) still true. So outer H = T.
        # At step 10, r becomes false. H(r) becomes F. If pSq is still true, outer impl is F.
        trace = (
            [{"p": T, "q": F, "r": T}] * 2   # pSq=F, impl=T, outerH=T
            + [{"p": T, "q": T, "r": T}]       # pSq=T, H(r)=T, impl=T, outerH=T
            + [{"p": T, "q": F, "r": T}] * 7   # pSq=T, H(r)=T, outerH=T
            + [{"p": T, "q": F, "r": F}]        # pSq=T, H(r)=F(r=F), impl=F, outerH=F
            + [{"p": F, "q": F, "r": T}] * 4
        )
        expected = [T] * 10 + [F] * 5
        run_trace("H((p S q) -> H(r))", trace, expected)

    def test_85_realistic_multi_turn_30_steps(self):
        """H(Y(request) -> response) & H(P(auth) -> !leak):
        After any request, a response must follow.
        Once authenticated, no leak may occur.
        """
        trace = [
            {"request": F, "response": F, "auth": F, "leak": F},  # T&T=T
            {"request": T, "response": F, "auth": F, "leak": F},  # Y(req)=F,T; P(auth)=F,T; T
            {"request": F, "response": T, "auth": F, "leak": F},  # Y(req)=T,resp=T,T; P(auth)=F,T; T
            {"request": F, "response": F, "auth": T, "leak": F},  # Y(req)=F,T; P(auth)=T,!leak=T,T; T
            {"request": T, "response": F, "auth": F, "leak": F},  # Y(req)=F,T; P(auth)=T,!leak=T,T; T
            {"request": F, "response": T, "auth": F, "leak": F},  # Y(req)=T,resp=T,T; P(auth)=T,T; T
            {"request": F, "response": F, "auth": F, "leak": F},  # Y(req)=F,T; T; T
            {"request": F, "response": F, "auth": F, "leak": F},  # T; T; T
            {"request": T, "response": F, "auth": F, "leak": F},  # Y(req)=F,T; T; T
            {"request": F, "response": F, "auth": F, "leak": F},  # Y(req)=T,resp=F => H1=F; T; F&T=F
        ] + [{"request": F, "response": F, "auth": F, "leak": F}] * 5
        expected = [T, T, T, T, T, T, T, T, T, F, F, F, F, F, F]
        run_trace("H(Y(request) -> response) & H(P(auth) -> !leak)", trace, expected)


# =====================================================================
# Category 6: Edge cases and stress tests (Tests 86-100+)
# =====================================================================


class TestEdgeCases:
    """Edge cases, boundary conditions, and stress tests."""

    def test_86_all_false_100_events(self):
        """H(p -> !q) with all propositions false for 100 events."""
        trace = [{}] * 100
        # Missing props default to False. p=F => p->!q = T. H stays T.
        expected = [T] * 100
        run_trace("H(p -> !q)", trace, expected)

    def test_87_all_true_100_events(self):
        """H(p -> q) with p=T, q=T for 100 events."""
        trace = [{"p": T, "q": T}] * 100
        expected = [T] * 100
        run_trace("H(p -> q)", trace, expected)

    def test_88_alternating_100_events(self):
        """Y(p) with alternating p over 100 events."""
        trace = [{"p": (i % 2 == 0)} for i in range(100)]
        # Y(p): step 0=F, step 1=p[0]=T, step 2=p[1]=F, step 3=p[2]=T,...
        expected = [F] + [(i % 2 == 0) for i in range(99)]
        run_trace("Y(p)", trace, expected)

    def test_89_single_prop_10_steps(self):
        """H(p): 10 steps, p true at all steps."""
        trace = [{"p": T}] * 10
        expected = [T] * 10
        run_trace("H(p)", trace, expected)

    def test_90_five_propositions(self):
        """H((a & b & c) -> (d | e)): 5 propositions in one formula.
        (a&b&c) -> (d|e) = !(a&b&c) | d | e.
        """
        trace = (
            [{"a": T, "b": T, "c": T, "d": T, "e": F}] * 5  # d=T => T
            + [{"a": T, "b": T, "c": T, "d": F, "e": T}] * 5  # e=T => T
            + [{"a": T, "b": T, "c": F, "d": F, "e": F}] * 5  # c=F => ant=F => T
            + [{"a": T, "b": T, "c": T, "d": F, "e": F}]       # violation
            + [{"a": F, "b": F, "c": F, "d": F, "e": F}] * 4
        )
        expected = [T] * 15 + [F] * 5
        run_trace("H((a & b & c) -> (d | e))", trace, expected)

    def test_91_deeply_nested(self):
        """H(P(Y(p -> !q))): deeply nested formula.
        Inner: p->!q = !p|!q.
        Y(inner): prev value of inner.
        P(Y(inner)): latches once Y(inner) is true.
        H(P(Y(inner))): once P latches, it's always T, so H tracks the first latch.

        Step 0: inner=!p|!q. Y(inner)=F. P(Y(inner))=F. H=F (since F AND prev=T = F).
        Step 1: inner=!p|!q. Y(inner)=inner[0]. If inner[0]=T, P latches T, H=T?
        Wait, H(phi) = phi AND prev_H. prev_H starts T.
        Step 0: P(Y(inner))=F. H = F AND T = F. Prev_H becomes F. H is now irrevocably F.

        So H(P(Y(...))) always starts F because P(Y(...)) is F at step 0.
        """
        trace = [{"p": F, "q": F}] * 15
        # Step 0: inner=T, Y(inner)=F, P=F, H=F AND T = F.
        # Once H is F at step 0, it stays F forever.
        expected = [F] * 15
        run_trace("H(P(Y(p -> !q)))", trace, expected)

    def test_92_reset_behavior(self):
        """Run trace, reset, run again -- monitor should give same results."""
        formula = "H(p -> !q)"
        trace = [{"p": T, "q": F}] * 5 + [{"p": T, "q": T}] + [{"p": F, "q": F}] * 4
        expected = [T] * 5 + [F] * 5

        ast = parse(formula)
        monitor = PtLTLMonitor(ast)
        # First run
        for i, (lab, exp) in enumerate(zip(trace, expected)):
            v = monitor.step(lab)
            assert v == exp, f"First run step {i}: expected {exp}, got {v}"

        # Reset
        monitor.reset()

        # Second run -- same results
        for i, (lab, exp) in enumerate(zip(trace, expected)):
            v = monitor.step(lab)
            assert v == exp, f"Second run step {i}: expected {exp}, got {v}"

    def test_93_state_snapshot(self):
        """Verify state_snapshot returns correct verdict after trace."""
        formula = "H(p -> !q)"
        ast = parse(formula)
        monitor = PtLTLMonitor(ast)
        monitor.step({"p": T, "q": F})  # T
        monitor.step({"p": T, "q": T})  # F
        snapshot = monitor.state_snapshot
        assert snapshot["verdict"] is False
        assert snapshot["step_count"] == 2

    def test_94_true_literal_formula(self):
        """Formula 'true' is always true."""
        trace = [{}] * 20
        expected = [T] * 20
        run_trace("true", trace, expected)

    def test_95_false_literal_formula(self):
        """Formula 'false' is always false."""
        trace = [{}] * 20
        expected = [F] * 20
        run_trace("false", trace, expected)

    def test_96_empty_labeling_every_step(self):
        """H(p) with empty labeling: p defaults to False, so H(p) is F from step 0."""
        trace = [{}] * 20
        expected = [F] * 20
        run_trace("H(p)", trace, expected)

    def test_97_empty_labeling_h_not_p(self):
        """H(!p) with empty labeling: p defaults to False, !p=T, H(!p) stays T."""
        trace = [{}] * 20
        expected = [T] * 20
        run_trace("H(!p)", trace, expected)

    def test_98_violation_at_step_99(self):
        """H(!p): 100-step trace, p becomes true only at step 99."""
        trace = [{"p": F}] * 99 + [{"p": T}]
        expected = [T] * 99 + [F]
        run_trace("H(!p)", trace, expected)

    def test_99_h_stays_false_after_recovery(self):
        """H(p): after violation, even if p becomes true again, H stays false."""
        trace = (
            [{"p": T}] * 30
            + [{"p": F}]        # violation at step 30
            + [{"p": T}] * 69  # recovery attempt
        )
        expected = [T] * 30 + [F] * 70
        run_trace("H(p)", trace, expected)

    def test_100_p_double_latch(self):
        """P(p) | P(q): either p or q has occurred."""
        trace = [{"p": F, "q": F}] * 10
        # Neither occurs, both P(p) and P(q) stay F.
        expected = [F] * 10
        run_trace("P(p) | P(q)", trace, expected)


# =====================================================================
# Category 7: Long stress traces (Tests 101-110)
# =====================================================================


class TestLongTraces:
    """Long traces with 50-100 events and various patterns."""

    def test_101_100_step_pattern_h_impl(self):
        """H(p -> q): 100 steps with deterministic pattern, violation at step 75."""
        trace = []
        for i in range(100):
            if i < 75:
                # p true every 5 steps, q always true
                trace.append({"p": (i % 5 == 0), "q": T})
            elif i == 75:
                trace.append({"p": T, "q": F})  # violation
            else:
                trace.append({"p": F, "q": T})
        expected = [T] * 75 + [F] * 25
        run_trace("H(p -> q)", trace, expected)

    def test_102_100_step_p_latch_at_50(self):
        """P(p): p becomes true at step 50 in a 100-step trace."""
        trace = [{"p": F}] * 50 + [{"p": T}] + [{"p": F}] * 49
        expected = [F] * 50 + [T] * 50
        run_trace("P(p)", trace, expected)

    def test_103_100_step_since_chain(self):
        """p S q: q triggers at step 10, p holds until step 90 then breaks."""
        trace = (
            [{"p": T, "q": F}] * 10
            + [{"p": T, "q": T}]         # trigger
            + [{"p": T, "q": F}] * 79   # p holds (steps 11-89)
            + [{"p": F, "q": F}]         # break at step 90
            + [{"p": T, "q": F}] * 9
        )
        expected = [F] * 10 + [T] * 80 + [F] * 10
        run_trace("p S q", trace, expected)

    def test_104_100_step_y_chain(self):
        """Y(Y(Y(Y(p)))): four steps ago. 100-step trace."""
        # Build a pattern: p true every 7 steps
        trace = [{"p": (i % 7 == 0)} for i in range(100)]
        # p values: i%7==0 => T at 0,7,14,21,28,...
        # Y(p): shifted by 1. Y^2: by 2. Y^3: by 3. Y^4: by 4.
        # Y^4(p) at step i = p[i-4] for i>=4, F for i<4.
        expected = []
        p_vals = [(i % 7 == 0) for i in range(100)]
        for i in range(100):
            if i < 4:
                expected.append(F)
            else:
                expected.append(p_vals[i - 4])
        run_trace("Y(Y(Y(Y(p))))", trace, expected)

    def test_105_100_step_combined_policies(self):
        """H(p -> !q) & H(r -> s): two policies, 100 steps, second fails at 80."""
        trace = []
        for i in range(100):
            if i < 80:
                trace.append({"p": F, "q": F, "r": (i % 10 == 0), "s": T})
            elif i == 80:
                trace.append({"p": F, "q": F, "r": T, "s": F})  # second policy fails
            else:
                trace.append({"p": F, "q": F, "r": F, "s": T})
        expected = [T] * 80 + [F] * 20
        run_trace("H(p -> !q) & H(r -> s)", trace, expected)

    def test_106_50_step_h_since_combined(self):
        """H((p S q) -> r): 50 steps."""
        trace = (
            [{"p": T, "q": F, "r": T}] * 5     # pSq=F, impl=T
            + [{"p": T, "q": T, "r": T}]         # pSq=T, r=T, impl=T
            + [{"p": T, "q": F, "r": T}] * 20   # pSq=T (chain), r=T, impl=T
            + [{"p": T, "q": F, "r": F}]          # pSq=T, r=F, impl=F, H=F
            + [{"p": F, "q": F, "r": T}] * 23
        )
        expected = [T] * 26 + [F] * 24
        run_trace("H((p S q) -> r)", trace, expected)

    def test_107_100_events_all_true_h_p_and_q(self):
        """H(p & q): both always true for 100 events."""
        trace = [{"p": T, "q": T}] * 100
        expected = [T] * 100
        run_trace("H(p & q)", trace, expected)

    def test_108_100_events_alternating_h_or(self):
        """H(p | q): alternating who is true, always at least one. 100 events."""
        trace = [{"p": (i % 2 == 0), "q": (i % 2 == 1)} for i in range(100)]
        expected = [T] * 100
        run_trace("H(p | q)", trace, expected)

    def test_109_100_step_p_never_latches(self):
        """P(p & q): p and q are never simultaneously true for 100 steps."""
        trace = [{"p": (i % 2 == 0), "q": (i % 2 == 1)} for i in range(100)]
        # p&q is always F since they alternate.
        expected = [F] * 100
        run_trace("P(p & q)", trace, expected)

    def test_110_100_step_complex_scenario(self):
        """H(Y(a) -> b) & H(P(c) -> !d) & H(e | f): three policies, 100 steps.
        All pass for 60 steps, third fails at step 60 (neither e nor f true).
        """
        trace = []
        for i in range(100):
            if i < 60:
                # a alternates, b always true when needed.
                # c never true. e always true.
                a_val = (i % 3 == 0)
                trace.append({"a": a_val, "b": T, "c": F, "d": F, "e": T, "f": F})
            elif i == 60:
                trace.append({"a": F, "b": T, "c": F, "d": F, "e": F, "f": F})  # e|f=F => H3=F
            else:
                trace.append({"a": F, "b": T, "c": F, "d": F, "e": T, "f": T})
        expected = [T] * 60 + [F] * 40
        run_trace("H(Y(a) -> b) & H(P(c) -> !d) & H(e | f)", trace, expected)
