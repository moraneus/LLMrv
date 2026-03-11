"""
Comprehensive tests for the ptLTL parser and incremental monitor.

~107 tests covering every code path, edge case, boundary condition, and example.
Written FIRST (TDD philosophy) — implementation must satisfy every test.

Examples tested:
  - Example 1: Fraud Prevention — H(p_fraud -> !q_comply)
  - Example 2: Sensitive Data Handling — H(Y(ps) -> qw) & H(P(ps) -> !qe)
  - Example 3: Jailbreak Detection — H((pe & P(pf)) -> !qu)
"""

import json

import pytest

from backend.engine.ptltl import (
    ASTNode,
    BinOpNode,
    BoolNode,
    HistoricallyNode,
    NotNode,
    ParseError,
    PreviouslyNode,
    PropNode,
    PtLTLMonitor,
    SinceNode,
    YesterdayNode,
    parse,
)

# PARSER TESTS


class TestParserAtomicAndLiterals:
    """Parser — Atomic propositions and boolean literals."""

    def test_parse_atomic_simple(self):
        """Single-letter proposition."""
        ast = parse("p")
        assert isinstance(ast, PropNode)
        assert ast.prop_id == "p"

    def test_parse_atomic_underscore(self):
        """Proposition with underscores."""
        ast = parse("p_fraud")
        assert isinstance(ast, PropNode)
        assert ast.prop_id == "p_fraud"

    def test_parse_atomic_alphanumeric(self):
        """Proposition with digits."""
        ast = parse("prop123")
        assert isinstance(ast, PropNode)
        assert ast.prop_id == "prop123"

    def test_parse_atomic_single_char(self):
        """Single character proposition."""
        ast = parse("x")
        assert isinstance(ast, PropNode)
        assert ast.prop_id == "x"

    def test_parse_true_literal(self):
        """Boolean literal true."""
        ast = parse("true")
        assert isinstance(ast, BoolNode)
        assert ast.value is True

    def test_parse_false_literal(self):
        """Boolean literal false."""
        ast = parse("false")
        assert isinstance(ast, BoolNode)
        assert ast.value is False

    def test_parse_whitespace_ignored(self):
        """Leading/trailing/internal whitespace is ignored."""
        ast = parse("  p  ")
        assert isinstance(ast, PropNode)
        assert ast.prop_id == "p"

    def test_parse_case_sensitive(self):
        """Uppercase and lowercase are different propositions."""
        ast_upper = parse("Q")
        ast_lower = parse("q")
        # Both are PropNodes but with different prop_ids
        assert isinstance(ast_upper, PropNode)
        assert isinstance(ast_lower, PropNode)
        assert ast_upper.prop_id != ast_lower.prop_id
        assert ast_upper.prop_id == "Q"
        assert ast_lower.prop_id == "q"


class TestParserUnaryOperators:
    """Parser — Negation and temporal unary operators."""

    def test_parse_negation(self):
        """Parse !p."""
        ast = parse("!p")
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, PropNode)
        assert ast.child.prop_id == "p"

    def test_parse_double_negation(self):
        """Parse !!p — negation of negation."""
        ast = parse("!!p")
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, NotNode)
        assert isinstance(ast.child.child, PropNode)
        assert ast.child.child.prop_id == "p"

    def test_parse_negation_of_complex(self):
        """Parse !(p & q) — negation of conjunction."""
        ast = parse("!(p & q)")
        assert isinstance(ast, NotNode)
        assert isinstance(ast.child, BinOpNode)
        assert ast.child.op == "&"

    def test_parse_yesterday(self):
        """Parse Y(p)."""
        ast = parse("Y(p)")
        assert isinstance(ast, YesterdayNode)
        assert isinstance(ast.child, PropNode)
        assert ast.child.prop_id == "p"

    def test_parse_once(self):
        """Parse P(p)."""
        ast = parse("P(p)")
        assert isinstance(ast, PreviouslyNode)
        assert isinstance(ast.child, PropNode)
        assert ast.child.prop_id == "p"

    def test_parse_historically(self):
        """Parse H(p)."""
        ast = parse("H(p)")
        assert isinstance(ast, HistoricallyNode)
        assert isinstance(ast.child, PropNode)
        assert ast.child.prop_id == "p"

    def test_parse_nested_temporal(self):
        """Parse H(Y(p)) — historically of yesterday."""
        ast = parse("H(Y(p))")
        assert isinstance(ast, HistoricallyNode)
        assert isinstance(ast.child, YesterdayNode)
        assert isinstance(ast.child.child, PropNode)
        assert ast.child.child.prop_id == "p"

    def test_parse_negation_inside_temporal(self):
        """Parse H(!p) — historically not p."""
        ast = parse("H(!p)")
        assert isinstance(ast, HistoricallyNode)
        assert isinstance(ast.child, NotNode)
        assert isinstance(ast.child.child, PropNode)

    def test_parse_temporal_of_conjunction(self):
        """Parse P(p & q) — previously of conjunction."""
        ast = parse("P(p & q)")
        assert isinstance(ast, PreviouslyNode)
        assert isinstance(ast.child, BinOpNode)
        assert ast.child.op == "&"

    def test_parse_historically_of_implication(self):
        """Parse H(p -> q)."""
        ast = parse("H(p -> q)")
        assert isinstance(ast, HistoricallyNode)
        assert isinstance(ast.child, BinOpNode)
        assert ast.child.op == "->"


class TestParserBinaryOperators:
    """Parser — Binary operators and associativity."""

    def test_parse_conjunction(self):
        """Parse p & q."""
        ast = parse("p & q")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        assert isinstance(ast.left, PropNode)
        assert ast.left.prop_id == "p"
        assert isinstance(ast.right, PropNode)
        assert ast.right.prop_id == "q"

    def test_parse_disjunction(self):
        """Parse p | q."""
        ast = parse("p | q")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "|"
        assert isinstance(ast.left, PropNode)
        assert isinstance(ast.right, PropNode)

    def test_parse_implication(self):
        """Parse p -> q."""
        ast = parse("p -> q")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "->"
        assert isinstance(ast.left, PropNode)
        assert isinstance(ast.right, PropNode)

    def test_parse_since(self):
        """Parse p S q."""
        ast = parse("p S q")
        assert isinstance(ast, SinceNode)
        assert isinstance(ast.left, PropNode)
        assert ast.left.prop_id == "p"
        assert isinstance(ast.right, PropNode)
        assert ast.right.prop_id == "q"

    def test_parse_conjunction_chain(self):
        """Parse p & q & r — left-associative: (p & q) & r."""
        ast = parse("p & q & r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        # Left should be (p & q)
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"
        assert ast.left.left.prop_id == "p"
        assert ast.left.right.prop_id == "q"
        # Right should be r
        assert isinstance(ast.right, PropNode)
        assert ast.right.prop_id == "r"

    def test_parse_disjunction_chain(self):
        """Parse p | q | r — left-associative: (p | q) | r."""
        ast = parse("p | q | r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "|"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "|"
        assert ast.left.left.prop_id == "p"
        assert ast.left.right.prop_id == "q"
        assert ast.right.prop_id == "r"

    def test_parse_implication_right_associative(self):
        """Parse p -> q -> r — right-associative: p -> (q -> r)."""
        ast = parse("p -> q -> r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "->"
        assert isinstance(ast.left, PropNode)
        assert ast.left.prop_id == "p"
        # Right should be (q -> r)
        assert isinstance(ast.right, BinOpNode)
        assert ast.right.op == "->"
        assert ast.right.left.prop_id == "q"
        assert ast.right.right.prop_id == "r"

    def test_parse_mixed_and_or(self):
        """Parse p & q | r — precedence: (p & q) | r."""
        ast = parse("p & q | r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "|"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"

    def test_parse_mixed_or_implies(self):
        """Parse p | q -> r — precedence: (p | q) -> r."""
        ast = parse("p | q -> r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "->"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "|"

    def test_parse_explicit_parens_override(self):
        """Parse p & (q | r) — parens override default precedence."""
        ast = parse("p & (q | r)")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        assert isinstance(ast.left, PropNode)
        assert isinstance(ast.right, BinOpNode)
        assert ast.right.op == "|"

    def test_parse_since_lowest_precedence(self):
        """Parse p & q S r | s — S has lowest precedence: (p & q) S (r | s)."""
        ast = parse("p & q S r | s")
        assert isinstance(ast, SinceNode)
        # Left of S: p & q
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"
        # Right of S: r | s
        assert isinstance(ast.right, BinOpNode)
        assert ast.right.op == "|"

    def test_parse_since_with_complex_operands(self):
        """Parse (p & q) S (r | s) — complex operands on both sides."""
        ast = parse("(p & q) S (r | s)")
        assert isinstance(ast, SinceNode)
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"
        assert isinstance(ast.right, BinOpNode)
        assert ast.right.op == "|"


class TestParserFormulas:
    """Parser — formula tests."""

    def test_parse_fraud_formula(self):
        """Example 1: H(p_fraud -> !q_comply)."""
        ast = parse("H(p_fraud -> !q_comply)")
        assert isinstance(ast, HistoricallyNode)
        impl = ast.child
        assert isinstance(impl, BinOpNode)
        assert impl.op == "->"
        assert isinstance(impl.left, PropNode)
        assert impl.left.prop_id == "p_fraud"
        assert isinstance(impl.right, NotNode)
        assert isinstance(impl.right.child, PropNode)
        assert impl.right.child.prop_id == "q_comply"

    def test_parse_sensitive_data_formula(self):
        """Example 2: H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)."""
        ast = parse("H(Y(ps) -> qw) & H(P(ps) -> !qe)")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        # Left: H(Y(ps) -> qw)
        left = ast.left
        assert isinstance(left, HistoricallyNode)
        impl1 = left.child
        assert isinstance(impl1, BinOpNode)
        assert impl1.op == "->"
        assert isinstance(impl1.left, YesterdayNode)
        assert isinstance(impl1.left.child, PropNode)
        # Right: H(P(ps) -> !qe)
        right = ast.right
        assert isinstance(right, HistoricallyNode)
        impl2 = right.child
        assert isinstance(impl2, BinOpNode)
        assert impl2.op == "->"
        assert isinstance(impl2.left, PreviouslyNode)
        assert isinstance(impl2.right, NotNode)

    def test_parse_jailbreak_formula(self):
        """Example 3: H((p_escalate & P(p_frame)) -> !q_unsafe)."""
        ast = parse("H((p_escalate & P(p_frame)) -> !q_unsafe)")
        assert isinstance(ast, HistoricallyNode)
        impl = ast.child
        assert isinstance(impl, BinOpNode)
        assert impl.op == "->"
        # Left: p_escalate & P(p_frame)
        conj = impl.left
        assert isinstance(conj, BinOpNode)
        assert conj.op == "&"
        assert isinstance(conj.left, PropNode)
        assert conj.left.prop_id == "p_escalate"
        assert isinstance(conj.right, PreviouslyNode)
        # Right: !q_unsafe
        assert isinstance(impl.right, NotNode)
        assert impl.right.child.prop_id == "q_unsafe"

    def test_parse_deep_nesting(self):
        """Parse H(Y(P(p -> !q))) — 3 levels of temporal nesting."""
        ast = parse("H(Y(P(p -> !q)))")
        assert isinstance(ast, HistoricallyNode)
        assert isinstance(ast.child, YesterdayNode)
        assert isinstance(ast.child.child, PreviouslyNode)
        inner = ast.child.child.child
        assert isinstance(inner, BinOpNode)
        assert inner.op == "->"

    def test_parse_all_operators_combined(self):
        """Formula using every operator: H, O, Y, S, &, |, ->, !."""
        formula = "H(Y(p) -> !q) & P(r | s) S (a -> b)"
        ast = parse(formula)
        # Outermost is S (lowest precedence)
        assert isinstance(ast, SinceNode)
        # Left of S: H(...) & P(...)
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"
        assert isinstance(ast.left.left, HistoricallyNode)
        assert isinstance(ast.left.right, PreviouslyNode)

    def test_parse_complex_real_world(self):
        """Realistic 4-proposition formula with multiple temporal operators."""
        formula = "H((p_harmful & P(p_jailbreak)) -> !(q_comply | q_partial))"
        ast = parse(formula)
        assert isinstance(ast, HistoricallyNode)
        impl = ast.child
        assert isinstance(impl, BinOpNode)
        assert impl.op == "->"
        # Antecedent: p_harmful & P(p_jailbreak)
        ant = impl.left
        assert isinstance(ant, BinOpNode)
        assert ant.op == "&"
        # Consequent: !(q_comply | q_partial)
        cons = impl.right
        assert isinstance(cons, NotNode)
        assert isinstance(cons.child, BinOpNode)
        assert cons.child.op == "|"


class TestParserPrecedence:
    """Parser — Precedence verification without explicit parens."""

    def test_precedence_negation_over_conjunction(self):
        """!p & q means (!p) & q, not !(p & q)."""
        ast = parse("!p & q")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        assert isinstance(ast.left, NotNode)
        assert isinstance(ast.right, PropNode)

    def test_precedence_conjunction_over_disjunction(self):
        """p & q | r means (p & q) | r."""
        ast = parse("p & q | r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "|"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "&"

    def test_precedence_disjunction_over_implication(self):
        """p | q -> r means (p | q) -> r."""
        ast = parse("p | q -> r")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "->"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "|"

    def test_precedence_implication_over_since(self):
        """p -> q S r means (p -> q) S r."""
        ast = parse("p -> q S r")
        assert isinstance(ast, SinceNode)
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "->"
        assert isinstance(ast.right, PropNode)

    def test_precedence_temporal_highest(self):
        """H(p) & q means (H(p)) & q, not H(p & q)."""
        ast = parse("H(p) & q")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        assert isinstance(ast.left, HistoricallyNode)
        assert isinstance(ast.right, PropNode)

    def test_precedence_parentheses_override_all(self):
        """(p | q) & (r -> s) — parentheses respected."""
        ast = parse("(p | q) & (r -> s)")
        assert isinstance(ast, BinOpNode)
        assert ast.op == "&"
        assert isinstance(ast.left, BinOpNode)
        assert ast.left.op == "|"
        assert isinstance(ast.right, BinOpNode)
        assert ast.right.op == "->"


class TestParserErrors:
    """Parser — Error handling."""

    def test_parse_empty_string(self):
        """Empty formula raises ParseError."""
        with pytest.raises(ParseError):
            parse("")

    def test_parse_just_operator(self):
        """Bare operator raises ParseError."""
        with pytest.raises(ParseError):
            parse("&")

    def test_parse_unclosed_paren(self):
        """Unclosed parenthesis raises ParseError with position."""
        with pytest.raises(ParseError):
            parse("(p & q")

    def test_parse_extra_close_paren(self):
        """Extra closing parenthesis raises ParseError."""
        with pytest.raises(ParseError):
            parse("p & q)")

    def test_parse_missing_operand(self):
        """Trailing operator without right operand."""
        with pytest.raises(ParseError):
            parse("p &")

    def test_parse_double_operator(self):
        """Two operators in a row raises ParseError."""
        with pytest.raises(ParseError):
            parse("p && q")

    def test_parse_temporal_missing_paren(self):
        """H without parentheses — H p — raises ParseError."""
        with pytest.raises(ParseError):
            parse("H p")

    def test_parse_temporal_empty_parens(self):
        """H() with nothing inside raises ParseError."""
        with pytest.raises(ParseError):
            parse("H()")

    def test_parse_unknown_token(self):
        """Unknown character @ raises ParseError."""
        with pytest.raises(ParseError):
            parse("p @ q")

    def test_parse_error_includes_position(self):
        """ParseError includes character position."""
        with pytest.raises(ParseError) as exc_info:
            parse("p & & q")
        assert exc_info.value.position is not None

    def test_parse_error_includes_description(self):
        """ParseError message is human-readable."""
        with pytest.raises(ParseError) as exc_info:
            parse("(p")
        assert len(str(exc_info.value)) > 10  # Not a bare error

    @pytest.mark.parametrize(
        "formula",
        [
            "",
            "&",
            "H(",
            "!!!",
            "p -> -> q",
            "S p q",
            "p q",
            "-> p",
            "|",
            "( )",
            "H(p",
            "p &",
        ],
    )
    def test_parse_various_invalid(self, formula: str):
        """Multiple invalid formulas all raise ParseError."""
        with pytest.raises(ParseError):
            parse(formula)


class TestParserRoundtripProperties:
    """Parser — Structural property tests."""

    def test_parse_returns_ast_node(self):
        """Return type is always an ASTNode subclass."""
        for formula in ["p", "!p", "p & q", "H(p)", "P(p)", "Y(p)", "p S q", "true"]:
            ast = parse(formula)
            assert isinstance(ast, ASTNode)

    def test_parse_preserves_proposition_names(self):
        """PropNode.prop_id matches the input identifier exactly."""
        for name in ["p", "p_fraud", "myProp123", "x", "abc_def_ghi"]:
            ast = parse(name)
            assert isinstance(ast, PropNode)
            assert ast.prop_id == name

    def test_parse_different_formulas_different_asts(self):
        """H(p) and P(p) produce different AST node types."""
        h_ast = parse("H(p)")
        p_ast = parse("P(p)")
        assert not isinstance(h_ast, type(p_ast))
        assert isinstance(h_ast, HistoricallyNode)
        assert isinstance(p_ast, PreviouslyNode)

    def test_parse_equivalent_with_parens(self):
        """(p & q) produces same AST structure as p & q."""
        ast1 = parse("p & q")
        ast2 = parse("(p & q)")
        assert isinstance(ast1, type(ast2))
        assert ast1 == ast2  # Dataclass equality


# MONITOR TESTS


class TestMonitorInit:
    """Monitor — Initialization tests."""

    def test_monitor_init_fresh_verdict_true(self):
        """New monitor with no steps → verdict is True (vacuously)."""
        formula = parse("H(p -> !q)")
        monitor = PtLTLMonitor(formula)
        assert monitor.verdict is True

    def test_monitor_init_step_zero(self):
        """No steps processed yet."""
        formula = parse("p")
        monitor = PtLTLMonitor(formula)
        # Verdict should be True before any steps
        assert monitor.verdict is True

    def test_monitor_init_state_snapshot(self):
        """state_snapshot is a dict with expected structure."""
        formula = parse("H(p)")
        monitor = PtLTLMonitor(formula)
        snap = monitor.state_snapshot
        assert isinstance(snap, dict)
        assert "step_count" in snap
        assert "verdict" in snap
        assert "sub_formulas" in snap

    def test_monitor_init_state_serializable(self):
        """state_snapshot can be JSON-serialized."""
        formula = parse("H(p -> !q)")
        monitor = PtLTLMonitor(formula)
        json_str = json.dumps(monitor.state_snapshot)
        assert isinstance(json_str, str)

    def test_monitor_init_various_formulas(self):
        """Monitor can be initialized with any valid formula."""
        formulas = [
            "p",
            "!p",
            "p & q",
            "H(p)",
            "P(p)",
            "Y(p)",
            "p S q",
            "H(p -> !q)",
            "H(Y(ps) -> qw) & H(P(ps) -> !qe)",
        ]
        for f_str in formulas:
            formula = parse(f_str)
            monitor = PtLTLMonitor(formula)
            assert monitor.verdict is True


class TestMonitorBasicOperators:
    """Monitor — Basic boolean operator evaluation."""

    def test_monitor_prop_true(self):
        """Proposition p=True → True."""
        monitor = PtLTLMonitor(parse("p"))
        assert monitor.step({"p": True}) is True

    def test_monitor_prop_false(self):
        """Proposition p=False → False."""
        monitor = PtLTLMonitor(parse("p"))
        assert monitor.step({"p": False}) is False

    def test_monitor_negation_true(self):
        """!p with p=False → True."""
        monitor = PtLTLMonitor(parse("!p"))
        assert monitor.step({"p": False}) is True

    def test_monitor_negation_false(self):
        """!p with p=True → False."""
        monitor = PtLTLMonitor(parse("!p"))
        assert monitor.step({"p": True}) is False

    def test_monitor_conjunction_tt(self):
        """p & q with both True → True."""
        monitor = PtLTLMonitor(parse("p & q"))
        assert monitor.step({"p": True, "q": True}) is True

    def test_monitor_conjunction_tf(self):
        """p & q with p=True, q=False → False."""
        monitor = PtLTLMonitor(parse("p & q"))
        assert monitor.step({"p": True, "q": False}) is False

    def test_monitor_disjunction_ff(self):
        """p | q with both False → False."""
        monitor = PtLTLMonitor(parse("p | q"))
        assert monitor.step({"p": False, "q": False}) is False

    def test_monitor_disjunction_tf(self):
        """p | q with p=True, q=False → True."""
        monitor = PtLTLMonitor(parse("p | q"))
        assert monitor.step({"p": True, "q": False}) is True

    def test_monitor_implication_tf(self):
        """p -> q with p=True, q=False → False."""
        monitor = PtLTLMonitor(parse("p -> q"))
        assert monitor.step({"p": True, "q": False}) is False

    def test_monitor_implication_ff(self):
        """p -> q with p=False, q=False → True (vacuous truth)."""
        monitor = PtLTLMonitor(parse("p -> q"))
        assert monitor.step({"p": False, "q": False}) is True


class TestMonitorTemporalOperators:
    """Monitor — Temporal operator semantics."""

    def test_monitor_yesterday_step0(self):
        """Y(p) at step 0 → False (no previous step exists)."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        assert monitor.step({"p": True}) is False

    def test_monitor_yesterday_step1_true(self):
        """Y(p) at step 1 when p was True at step 0 → True."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        monitor.step({"p": True})  # step 0
        assert monitor.step({"p": False}) is True  # step 1: Y reflects step 0

    def test_monitor_yesterday_step1_false(self):
        """Y(p) at step 1 when p was False at step 0 → False."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        monitor.step({"p": False})  # step 0
        assert monitor.step({"p": True}) is False  # step 1: Y reflects step 0

    def test_monitor_yesterday_tracks_previous(self):
        """Y reflects ONLY the previous step, not current or older."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        assert monitor.step({"p": True}) is False  # step 0: Y=F (no prev)
        assert monitor.step({"p": False}) is True  # step 1: Y=T (prev p=T)
        assert monitor.step({"p": True}) is False  # step 2: Y=F (prev p=F)
        assert monitor.step({"p": False}) is True  # step 3: Y=T (prev p=T)

    def test_monitor_once_initially_false(self):
        """P(p) with p=False at step 0 → False."""
        monitor = PtLTLMonitor(parse("P(p)"))
        assert monitor.step({"p": False}) is False

    def test_monitor_once_triggers_on_true(self):
        """P(p) becomes True when p first becomes True."""
        monitor = PtLTLMonitor(parse("P(p)"))
        assert monitor.step({"p": False}) is False  # step 0
        assert monitor.step({"p": False}) is False  # step 1
        assert monitor.step({"p": True}) is True  # step 2: triggered

    def test_monitor_once_latches_permanently(self):
        """Once P(p) becomes True, it stays True forever."""
        monitor = PtLTLMonitor(parse("P(p)"))
        monitor.step({"p": False})
        monitor.step({"p": True})  # triggered
        assert monitor.step({"p": False}) is True  # stays True
        assert monitor.step({"p": False}) is True
        assert monitor.step({"p": False}) is True

    def test_monitor_once_never_triggered(self):
        """p always False → P(p) always False."""
        monitor = PtLTLMonitor(parse("P(p)"))
        for _ in range(10):
            assert monitor.step({"p": False}) is False

    def test_monitor_historically_initially_true(self):
        """H(p) with p=True at step 0 → True."""
        monitor = PtLTLMonitor(parse("H(p)"))
        assert monitor.step({"p": True}) is True

    def test_monitor_historically_stays_true(self):
        """p=True for 10 steps → H(p) True throughout."""
        monitor = PtLTLMonitor(parse("H(p)"))
        for _ in range(10):
            assert monitor.step({"p": True}) is True

    def test_monitor_historically_violation(self):
        """p=False at step 5 → H(p) False from that point."""
        monitor = PtLTLMonitor(parse("H(p)"))
        for _ in range(5):
            assert monitor.step({"p": True}) is True
        assert monitor.step({"p": False}) is False  # violation

    def test_monitor_historically_irrevocable(self):
        """Once H(p) goes to False, it stays False even if p goes back to True."""
        monitor = PtLTLMonitor(parse("H(p)"))
        monitor.step({"p": True})
        monitor.step({"p": True})
        monitor.step({"p": False})  # violated
        assert monitor.step({"p": True}) is False  # irrevocable
        assert monitor.step({"p": True}) is False
        assert monitor.verdict is False

    def test_monitor_since_basic(self):
        """p S q: q triggers, p maintains.

        Recurrence: now(p S q) = now(q) OR (now(p) AND prev(p S q))
        """
        monitor = PtLTLMonitor(parse("p S q"))
        assert monitor.step({"p": False, "q": False}) is False  # no trigger
        assert monitor.step({"p": False, "q": True}) is True  # q triggers
        assert monitor.step({"p": True, "q": False}) is True  # p maintains
        assert monitor.step({"p": True, "q": False}) is True  # p continues
        assert monitor.step({"p": False, "q": False}) is False  # p breaks

    def test_monitor_since_not_triggered(self):
        """p S q: q never True → always False."""
        monitor = PtLTLMonitor(parse("p S q"))
        for _ in range(10):
            assert monitor.step({"p": True, "q": False}) is False


# PAPER EXAMPLE 1: FRAUD PREVENTION
# Formula: H(p_fraud -> !q_comply)


class TestMonitorFraud:
    """Monitor — Example 1: Fraud Prevention."""

    FORMULA = "H(p_fraud -> !q_comply)"

    def _make_monitor(self) -> PtLTLMonitor:
        return PtLTLMonitor(parse(self.FORMULA))

    def test_fraud_all_benign(self):
        """No fraud requests → H(F->X) = H(T) always True (vacuously)."""
        monitor = self._make_monitor()
        for _ in range(6):
            assert monitor.step({"p_fraud": False, "q_comply": False}) is True

    def test_fraud_request_refuse(self):
        """p_fraud=T, q_comply=F → T->T = T → H stays True."""
        monitor = self._make_monitor()
        assert monitor.step({"p_fraud": True, "q_comply": False}) is True

    def test_fraud_request_comply_violation(self):
        """p_fraud=T, q_comply=T → T->F = F → H = False."""
        monitor = self._make_monitor()
        assert monitor.step({"p_fraud": True, "q_comply": True}) is False

    def test_fraud_full_trace_pass(self):
        """Example 1 full passing trace (6 steps).

        m0: system greeting         — pw=F, qc=F
        m1: user benign             — pw=F, qc=F
        m2: user asks about fraud — pf=T, qc=F
        m3: assistant refuses       — pw=F, qc=F
        m4: user asks again         — pw=T, qc=F
        m5: assistant refuses again — pw=F, qc=F
        """
        monitor = self._make_monitor()
        trace = [
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": True, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": True, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
        ]
        for labeling in trace:
            assert monitor.step(labeling) is True
        assert monitor.verdict is True

    def test_fraud_full_trace_violation(self):
        """Same trace but assistant complies at step 3 → False."""
        monitor = self._make_monitor()
        monitor.step({"p_fraud": False, "q_comply": False})  # m0
        monitor.step({"p_fraud": False, "q_comply": False})  # m1
        monitor.step({"p_fraud": True, "q_comply": False})  # m2
        assert monitor.step({"p_fraud": True, "q_comply": True}) is False  # m3: VIOLATION

    def test_fraud_violation_irrevocable(self):
        """After violation at step 3, steps 4-5 remain False."""
        monitor = self._make_monitor()
        monitor.step({"p_fraud": False, "q_comply": False})
        monitor.step({"p_fraud": False, "q_comply": False})
        monitor.step({"p_fraud": True, "q_comply": False})
        monitor.step({"p_fraud": True, "q_comply": True})  # violation
        # Even benign steps remain False
        assert monitor.step({"p_fraud": False, "q_comply": False}) is False
        assert monitor.step({"p_fraud": False, "q_comply": False}) is False
        assert monitor.verdict is False

    def test_fraud_step_by_step_labeling(self):
        """Verify exact verdict at each step in the passing trace."""
        monitor = self._make_monitor()
        expected_verdicts = [True, True, True, True, True, True]
        trace = [
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": True, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
            {"p_fraud": True, "q_comply": False},
            {"p_fraud": False, "q_comply": False},
        ]
        for i, (labeling, expected) in enumerate(zip(trace, expected_verdicts, strict=True)):
            result = monitor.step(labeling)
            assert result is expected, f"Step {i}: expected {expected}, got {result}"

    def test_fraud_delayed_request(self):
        """First 4 steps benign, fraud request at step 5 → still True (no comply)."""
        monitor = self._make_monitor()
        for _ in range(4):
            monitor.step({"p_fraud": False, "q_comply": False})
        # Request at step 5 but no compliance → implication holds
        assert monitor.step({"p_fraud": True, "q_comply": False}) is True
        assert monitor.verdict is True


# PAPER EXAMPLE 2: SENSITIVE DATA HANDLING
# Formula: H(Y(ps) -> qw) & H(P(ps) -> !qe)


class TestMonitorSensitiveData:
    """Monitor — Example 2: Sensitive Data."""

    FORMULA = "H(Y(ps) -> qw) & H(P(ps) -> !qe)"

    def _make_monitor(self) -> PtLTLMonitor:
        return PtLTLMonitor(parse(self.FORMULA))

    def _empty_labeling(self) -> dict[str, bool]:
        return {"ps": False, "qw": False, "qe": False}

    def test_sensitive_pass_complete(self):
        """SSN shared → warning given → no echo → both conjuncts True.

        m0: greeting      — ps=F, qw=F, qe=F
        m1: greeting      — ps=F, qw=F, qe=F
        m2: user SSN      — ps=T, qw=F, qe=F
        m3: warns, safe   — ps=F, qw=T, qe=F
        m4: user asks     — ps=F, qw=F, qe=F
        m5: refuses echo  — ps=F, qw=F, qe=F
        """
        monitor = self._make_monitor()
        trace = [
            {"ps": False, "qw": False, "qe": False},
            {"ps": False, "qw": False, "qe": False},
            {"ps": True, "qw": False, "qe": False},
            {"ps": False, "qw": True, "qe": False},
            {"ps": False, "qw": False, "qe": False},
            {"ps": False, "qw": False, "qe": False},
        ]
        for labeling in trace:
            monitor.step(labeling)
        assert monitor.verdict is True

    def test_sensitive_missing_warning(self):
        """No warning after SSN → first conjunct H(Y(ps)->qw) violated.

        At m3: Y(ps)=T (ps was T at m2), qw=F → T->F = F → H=F
        """
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())  # m0
        monitor.step(self._empty_labeling())  # m1
        monitor.step({"ps": True, "qw": False, "qe": False})  # m2: SSN
        result = monitor.step(self._empty_labeling())  # m3: no warning!
        assert result is False
        assert monitor.verdict is False

    def test_sensitive_echo_violation(self):
        """SSN shared, assistant echoes → second conjunct H(P(ps)->!qe) violated.

        At m3: P(ps)=T (triggered at m2), qe=T → T->F = F → H=F
        """
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())  # m0
        monitor.step(self._empty_labeling())  # m1
        monitor.step({"ps": True, "qw": False, "qe": False})  # m2: SSN
        result = monitor.step({"ps": False, "qw": True, "qe": True})  # m3: warns but echoes
        assert result is False

    def test_sensitive_both_violated(self):
        """Missing warning AND echo → both conjuncts False."""
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())
        monitor.step(self._empty_labeling())
        monitor.step({"ps": True, "qw": False, "qe": False})  # SSN
        # No warning (qw=F), echoes (qe=T)
        result = monitor.step({"ps": False, "qw": False, "qe": True})
        assert result is False

    def test_sensitive_no_sensitive_data(self):
        """No SSN shared → always True (vacuously)."""
        monitor = self._make_monitor()
        for _ in range(10):
            assert monitor.step(self._empty_labeling()) is True

    def test_sensitive_yesterday_only_previous_step(self):
        """Y(ps) is true only when ps was true at step i-1."""
        # We test the Y semantics within the compound formula
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())  # m0: ps=F
        monitor.step(self._empty_labeling())  # m1: ps=F, Y(ps)=F → implication vacuously T
        monitor.step({"ps": True, "qw": False, "qe": False})  # m2: ps=T
        # m3: Y(ps)=T (ps was T at m2), need qw=T
        assert monitor.step({"ps": False, "qw": True, "qe": False}) is True

    def test_sensitive_once_latches_after_ssn(self):
        """P(ps) stays True after first SSN disclosure."""
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())  # m0
        monitor.step({"ps": True, "qw": False, "qe": False})  # m1: SSN
        # Provide warning at m2 so first conjunct is happy
        monitor.step({"ps": False, "qw": True, "qe": False})  # m2
        # Now any qe=T should cause violation due to P(ps) being latched
        for _ in range(3):
            monitor.step(self._empty_labeling())  # benign
        result = monitor.step({"ps": False, "qw": False, "qe": True})  # echo far later
        assert result is False

    def test_sensitive_delayed_echo_still_caught(self):
        """SSN at step 1, echo at step 20 → caught by P(ps)."""
        monitor = self._make_monitor()
        monitor.step(self._empty_labeling())  # m0
        monitor.step({"ps": True, "qw": False, "qe": False})  # m1: SSN
        monitor.step({"ps": False, "qw": True, "qe": False})  # m2: warn
        # 17 benign steps
        for _ in range(17):
            monitor.step(self._empty_labeling())
        # Echo at step 20
        result = monitor.step({"ps": False, "qw": False, "qe": True})
        assert result is False


# PAPER EXAMPLE 3: JAILBREAK DETECTION
# Formula: H((p_escalate & P(p_frame)) -> !q_unsafe)


class TestMonitorJailbreak:
    """Monitor — Example 3: Jailbreak."""

    FORMULA = "H((p_escalate & P(p_frame)) -> !q_unsafe)"

    def _make_monitor(self) -> PtLTLMonitor:
        return PtLTLMonitor(parse(self.FORMULA))

    def _empty(self) -> dict[str, bool]:
        return {"p_escalate": False, "p_frame": False, "q_unsafe": False}

    def test_jailbreak_pass_no_frame_no_escalate(self):
        """Benign conversation → True throughout."""
        monitor = self._make_monitor()
        for _ in range(6):
            assert monitor.step(self._empty()) is True

    def test_jailbreak_pass_frame_only(self):
        """Frame but no escalate → antecedent (pe & P(pf)) = F & T = F → True."""
        monitor = self._make_monitor()
        monitor.step(self._empty())
        monitor.step({"p_escalate": False, "p_frame": True, "q_unsafe": False})  # frame
        monitor.step(self._empty())
        assert monitor.verdict is True

    def test_jailbreak_pass_escalate_no_frame(self):
        """Escalate without prior frame → P(p_frame)=F → antecedent F → True."""
        monitor = self._make_monitor()
        monitor.step(self._empty())
        monitor.step({"p_escalate": True, "p_frame": False, "q_unsafe": False})  # escalate
        assert monitor.verdict is True

    def test_jailbreak_pass_frame_escalate_refuse(self):
        """Frame → escalate → assistant refuses → True.

        m0: benign
        m1: benign
        m2: frame (pf=T)
        m3: benign
        m4: escalate (pe=T, P(pf)=T → antecedent T, qu=F → !qu=T → T)
        m5: refuses (qu=F)
        """
        monitor = self._make_monitor()
        monitor.step(self._empty())
        monitor.step(self._empty())
        monitor.step({"p_escalate": False, "p_frame": True, "q_unsafe": False})
        monitor.step(self._empty())
        monitor.step({"p_escalate": True, "p_frame": False, "q_unsafe": False})
        monitor.step(self._empty())
        assert monitor.verdict is True

    def test_jailbreak_violation(self):
        """Frame → escalate + comply → False.

        At violation step: pe=T, P(pf)=T → pe & P(pf)=T, qu=T → !qu=F → T->F=F → H=F
        """
        monitor = self._make_monitor()
        monitor.step(self._empty())
        monitor.step(self._empty())
        monitor.step({"p_escalate": False, "p_frame": True, "q_unsafe": False})  # frame
        monitor.step(self._empty())
        result = monitor.step(
            {"p_escalate": True, "p_frame": False, "q_unsafe": True}
        )  # escalate + comply
        assert result is False
        assert monitor.verdict is False

    def test_jailbreak_delayed_escalate(self):
        """Frame at step 1, escalate at step 10 → P(p_frame) still True."""
        monitor = self._make_monitor()
        monitor.step(self._empty())
        monitor.step({"p_escalate": False, "p_frame": True, "q_unsafe": False})  # frame
        # 8 benign steps
        for _ in range(8):
            monitor.step(self._empty())
        # Escalate + comply at step 10 → violation due to P(pf)=T
        result = monitor.step({"p_escalate": True, "p_frame": False, "q_unsafe": True})
        assert result is False


# MONITOR STATE MANAGEMENT


class TestMonitorStateManagement:
    """Monitor — State management tests."""

    def test_monitor_reset_restores_initial(self):
        """After reset, verdict back to True."""
        monitor = PtLTLMonitor(parse("H(p)"))
        monitor.step({"p": True})
        monitor.step({"p": False})  # violate
        assert monitor.verdict is False
        monitor.reset()
        assert monitor.verdict is True

    def test_monitor_reset_clears_history(self):
        """After reset, Y() returns False at next step."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        monitor.step({"p": True})  # step 0
        assert monitor.step({"p": False}) is True  # step 1: Y(p)=T
        monitor.reset()
        # After reset, step 0 again: Y(p) should be False
        assert monitor.step({"p": True}) is False

    def test_monitor_state_snapshot_structure(self):
        """Snapshot has step_count, verdict, and sub_formulas keys."""
        monitor = PtLTLMonitor(parse("H(p -> !q)"))
        monitor.step({"p": True, "q": False})
        snap = monitor.state_snapshot
        assert "step_count" in snap
        assert "verdict" in snap
        assert "sub_formulas" in snap
        assert snap["step_count"] == 1

    def test_monitor_state_snapshot_changes(self):
        """Snapshot changes after each step."""
        monitor = PtLTLMonitor(parse("P(p)"))
        snap1 = monitor.state_snapshot
        monitor.step({"p": False})
        snap2 = monitor.state_snapshot
        assert snap1 != snap2
        assert snap2["step_count"] == 1

    def test_monitor_state_serializable(self):
        """json.dumps(state_snapshot) succeeds and round-trips."""
        monitor = PtLTLMonitor(parse("H(p_fraud -> !q_comply)"))
        monitor.step({"p_fraud": True, "q_comply": False})
        snap = monitor.state_snapshot
        json_str = json.dumps(snap)
        restored = json.loads(json_str)
        assert restored == snap

    def test_monitor_verdict_property(self):
        """verdict matches the last step() return value."""
        monitor = PtLTLMonitor(parse("p"))
        result = monitor.step({"p": True})
        assert monitor.verdict is result
        result = monitor.step({"p": False})
        assert monitor.verdict is result

    def test_monitor_long_trace(self):
        """100 steps → still works correctly."""
        monitor = PtLTLMonitor(parse("H(p)"))
        for _ in range(100):
            assert monitor.step({"p": True}) is True
        assert monitor.verdict is True
        # Now violate
        assert monitor.step({"p": False}) is False
        assert monitor.verdict is False

    def test_monitor_alternating_labelings(self):
        """Rapidly alternating True/False → correct at each step."""
        monitor = PtLTLMonitor(parse("Y(p)"))
        # Step 0: p=T, Y(p)=F (no prev)
        assert monitor.step({"p": True}) is False
        # Step 1: p=F, Y(p)=T (prev p=T)
        assert monitor.step({"p": False}) is True
        # Step 2: p=T, Y(p)=F (prev p=F)
        assert monitor.step({"p": True}) is False
        # Step 3: p=F, Y(p)=T (prev p=T)
        assert monitor.step({"p": False}) is True

    def test_monitor_missing_prop_defaults_false(self):
        """Propositions not in labeling default to False."""
        monitor = PtLTLMonitor(parse("p & q"))
        # Only provide p, q should default to False
        assert monitor.step({"p": True}) is False

    def test_monitor_extra_props_ignored(self):
        """Extra propositions in labeling are harmlessly ignored."""
        monitor = PtLTLMonitor(parse("p"))
        assert monitor.step({"p": True, "q": True, "r": False}) is True

    def test_monitor_bool_literal_true(self):
        """Monitor with formula 'true' always returns True."""
        monitor = PtLTLMonitor(parse("true"))
        assert monitor.step({}) is True
        assert monitor.step({}) is True

    def test_monitor_bool_literal_false(self):
        """Monitor with formula 'false' always returns False."""
        monitor = PtLTLMonitor(parse("false"))
        assert monitor.step({}) is False
        assert monitor.step({}) is False
