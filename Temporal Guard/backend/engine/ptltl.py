"""
ptLTL (Past-Time Linear Temporal Logic) parser and incremental monitor.

Recursive-descent parser producing an AST, with an incremental monitor
that maintains O(|phi|) state per formula using standard ptLTL recurrences
for Y (Yesterday), P (Previously), H (Historically), and S (Since).

Grammar:
    phi ::= true | false | PROP_ID
          | !phi | phi & phi | phi | phi | phi -> phi
          | Y(phi) | P(phi) | H(phi) | phi S phi

Operator precedence (highest to lowest):
    1. Y(), P(), H()  — unary temporal (prefix function-call syntax)
    2. !              — negation (prefix)
    3. &              — conjunction
    4. |              — disjunction
    5. ->             — implication (right-associative)
    6. S              — since (binary infix, lowest precedence)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

# AST Nodes


@dataclass
class ASTNode:
    """Base AST node for ptLTL formulas."""

    pass


@dataclass
class PropNode(ASTNode):
    """Atomic proposition reference."""

    prop_id: str


@dataclass
class BoolNode(ASTNode):
    """Boolean literal."""

    value: bool


@dataclass
class NotNode(ASTNode):
    """Negation."""

    child: ASTNode


@dataclass
class BinOpNode(ASTNode):
    """Binary boolean: &, |, ->"""

    op: str  # "&" | "|" | "->"
    left: ASTNode
    right: ASTNode


@dataclass
class YesterdayNode(ASTNode):
    """Y(phi) — yesterday."""

    child: ASTNode


@dataclass
class PreviouslyNode(ASTNode):
    """P(phi) — previously (existential past)."""

    child: ASTNode


@dataclass
class HistoricallyNode(ASTNode):
    """H(phi) — historically (universal past)."""

    child: ASTNode


@dataclass
class SinceNode(ASTNode):
    """phi S psi — since."""

    left: ASTNode  # phi (maintained)
    right: ASTNode  # psi (trigger)


# ParseError


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""

    def __init__(self, message: str, position: int | None = None):
        self.position = position
        if position is not None:
            message = f"Parse error at position {position}: {message}"
        super().__init__(message)


# Tokenizer


class TokenType(Enum):
    IDENT = auto()
    TRUE = auto()
    FALSE = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    SINCE = auto()
    YESTERDAY = auto()
    PREVIOUSLY = auto()
    HISTORICALLY = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    position: int


# Temporal operator keywords that require '(' after them
_TEMPORAL_OPS = {"Y", "P", "H"}
_KEYWORDS = {"true": TokenType.TRUE, "false": TokenType.FALSE}


def _tokenize(formula: str) -> list[Token]:
    """Tokenize a ptLTL formula string."""
    tokens: list[Token] = []
    i = 0
    length = len(formula)

    while i < length:
        ch = formula[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Single-character tokens
        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, "(", i))
            i += 1
        elif ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")", i))
            i += 1
        elif ch == "!":
            tokens.append(Token(TokenType.NOT, "!", i))
            i += 1
        elif ch == "&":
            tokens.append(Token(TokenType.AND, "&", i))
            i += 1
        elif ch == "|":
            tokens.append(Token(TokenType.OR, "|", i))
            i += 1
        elif ch == "-":
            # Must be ->
            if i + 1 < length and formula[i + 1] == ">":
                tokens.append(Token(TokenType.IMPLIES, "->", i))
                i += 2
            else:
                raise ParseError("Unexpected character '-', did you mean '->'?", i)
        elif ch.isalpha() or ch == "_":
            # Identifier or keyword
            start = i
            while i < length and (formula[i].isalnum() or formula[i] == "_"):
                i += 1
            word = formula[start:i]

            if word in _KEYWORDS:
                tokens.append(Token(_KEYWORDS[word], word, start))
            elif word == "S":
                # S is the Since operator (binary infix)
                tokens.append(Token(TokenType.SINCE, "S", start))
            elif word in _TEMPORAL_OPS:
                # Y, P, H — look ahead for '('
                token_map = {
                    "Y": TokenType.YESTERDAY,
                    "P": TokenType.PREVIOUSLY,
                    "H": TokenType.HISTORICALLY,
                }
                tokens.append(Token(token_map[word], word, start))
            else:
                tokens.append(Token(TokenType.IDENT, word, start))
        else:
            raise ParseError(f"Unexpected character '{ch}'", i)

    tokens.append(Token(TokenType.EOF, "", length))
    return tokens


# Recursive-Descent Parser


class _Parser:
    """Recursive-descent parser for ptLTL formulas.

    Precedence (lowest to highest):
        S       — since (binary infix)
        ->      — implication (right-associative)
        |       — disjunction
        &       — conjunction
        !       — negation (prefix)
        Y/P/H   — temporal (prefix function-call)
        atom    — identifiers, booleans, parenthesized expressions
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self) -> TokenType:
        return self.tokens[self.pos].type

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._current()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ('{tok.value}')", tok.position
            )
        return self._advance()

    # --- Grammar rules (lowest precedence first) ---

    def parse_formula(self) -> ASTNode:
        """Entry point: parse a complete formula."""
        if self._peek() == TokenType.EOF:
            raise ParseError("Empty formula", 0)
        node = self._parse_since()
        if self._peek() != TokenType.EOF:
            tok = self._current()
            raise ParseError(f"Unexpected token '{tok.value}'", tok.position)
        return node

    def _parse_since(self) -> ASTNode:
        """Since: impl (S impl)*  — left-associative."""
        left = self._parse_implication()
        while self._peek() == TokenType.SINCE:
            self._advance()
            right = self._parse_implication()
            left = SinceNode(left=left, right=right)
        return left

    def _parse_implication(self) -> ASTNode:
        """Implication: disjunction (-> implication)?  — right-associative."""
        left = self._parse_disjunction()
        if self._peek() == TokenType.IMPLIES:
            self._advance()
            right = self._parse_implication()  # Right-recursive for right-associativity
            return BinOpNode(op="->", left=left, right=right)
        return left

    def _parse_disjunction(self) -> ASTNode:
        """Disjunction: conjunction (| conjunction)*  — left-associative."""
        left = self._parse_conjunction()
        while self._peek() == TokenType.OR:
            self._advance()
            right = self._parse_conjunction()
            left = BinOpNode(op="|", left=left, right=right)
        return left

    def _parse_conjunction(self) -> ASTNode:
        """Conjunction: negation (& negation)*  — left-associative."""
        left = self._parse_negation()
        while self._peek() == TokenType.AND:
            self._advance()
            right = self._parse_negation()
            left = BinOpNode(op="&", left=left, right=right)
        return left

    def _parse_negation(self) -> ASTNode:
        """Negation: !* atom."""
        if self._peek() == TokenType.NOT:
            self._advance()
            child = self._parse_negation()  # Allow !!p
            return NotNode(child=child)
        return self._parse_atom()

    def _parse_atom(self) -> ASTNode:
        """Atom: temporal operators, identifiers, booleans, parenthesized expressions."""
        tok = self._current()

        # Temporal operators: Y(phi), P(phi), H(phi)
        if tok.type in (TokenType.YESTERDAY, TokenType.PREVIOUSLY, TokenType.HISTORICALLY):
            self._advance()
            self._expect(TokenType.LPAREN)
            child = self._parse_since()  # Full expression inside parens
            self._expect(TokenType.RPAREN)
            if tok.type == TokenType.YESTERDAY:
                return YesterdayNode(child=child)
            elif tok.type == TokenType.PREVIOUSLY:
                return PreviouslyNode(child=child)
            else:
                return HistoricallyNode(child=child)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._parse_since()
            self._expect(TokenType.RPAREN)
            return node

        # Boolean literals
        if tok.type == TokenType.TRUE:
            self._advance()
            return BoolNode(value=True)
        if tok.type == TokenType.FALSE:
            self._advance()
            return BoolNode(value=False)

        # Proposition identifier
        if tok.type == TokenType.IDENT:
            self._advance()
            return PropNode(prop_id=tok.value)

        raise ParseError(f"Unexpected token '{tok.value}'", tok.position)


def parse(formula_str: str) -> ASTNode:
    """Parse a ptLTL formula string into an AST.

    Args:
        formula_str: The formula string to parse.

    Returns:
        The root ASTNode of the parsed formula.

    Raises:
        ParseError: If the formula is syntactically invalid.
    """
    tokens = _tokenize(formula_str)
    parser = _Parser(tokens)
    return parser.parse_formula()


# Incremental Monitor


class PtLTLMonitor:
    """Incremental ptLTL monitor.

    Maintains O(|phi|) state. Processes one event at a time.
    Once an H(phi) formula evaluates to False, it stays False permanently (irrevocable).

    Recurrences:
        now(Y(phi), i)     = prev(phi)                            # False at i=0
        now(P(phi), i)     = now(phi, i) OR prev(P(phi))
        now(H(phi), i)     = now(phi, i) AND prev(H(phi))         # prev(H(phi)) = True at i=0
        now(phi S psi, i)  = now(psi, i) OR (now(phi, i) AND prev(phi S psi))
    """

    def __init__(self, formula: ASTNode):
        """Initialize the monitor.

        Args:
            formula: Parsed AST of the ptLTL formula.
        """
        self._formula = formula
        self._step_count = 0

        # Collect all sub-formula nodes and assign IDs
        self._nodes: list[ASTNode] = []
        self._node_ids: dict[int, int] = {}  # id(node) -> index
        self._collect_nodes(formula)

        # State arrays: now[i] and prev[i] for each sub-formula
        n = len(self._nodes)
        self._now: list[bool] = [False] * n
        self._prev: list[bool] = [False] * n

        # Initialize prev values for temporal operators
        self._init_prev()

    def _collect_nodes(self, node: ASTNode) -> None:
        """Walk AST and register all sub-formula nodes."""
        if id(node) in self._node_ids:
            return
        idx = len(self._nodes)
        self._nodes.append(node)
        self._node_ids[id(node)] = idx

        if isinstance(node, (NotNode, YesterdayNode, PreviouslyNode, HistoricallyNode)):
            self._collect_nodes(node.child)
        elif isinstance(node, (BinOpNode, SinceNode)):
            self._collect_nodes(node.left)
            self._collect_nodes(node.right)

    def _init_prev(self) -> None:
        """Set initial prev values for temporal operators."""
        for i, node in enumerate(self._nodes):
            if isinstance(node, HistoricallyNode):
                # prev(H(phi)) = True at i=0 (vacuously true)
                self._prev[i] = True
            else:
                # Y, P, S all start with prev = False
                self._prev[i] = False

    def _idx(self, node: ASTNode) -> int:
        """Get the index of a node in our state arrays."""
        return self._node_ids[id(node)]

    def _evaluate(self, node: ASTNode, labeling: dict[str, bool]) -> bool:
        """Recursively evaluate a sub-formula for the current step.

        Args:
            node: The AST node to evaluate.
            labeling: Map from prop_id -> True/False for this step.

        Returns:
            The truth value of this sub-formula at the current step.
        """
        idx = self._idx(node)

        if isinstance(node, PropNode):
            val = labeling.get(node.prop_id, False)
            self._now[idx] = val
            return val

        if isinstance(node, BoolNode):
            self._now[idx] = node.value
            return node.value

        if isinstance(node, NotNode):
            val = not self._evaluate(node.child, labeling)
            self._now[idx] = val
            return val

        if isinstance(node, BinOpNode):
            left = self._evaluate(node.left, labeling)
            right = self._evaluate(node.right, labeling)
            if node.op == "&":
                val = left and right
            elif node.op == "|":
                val = left or right
            elif node.op == "->":
                val = (not left) or right
            else:
                raise ValueError(f"Unknown binary operator: {node.op}")
            self._now[idx] = val
            return val

        if isinstance(node, YesterdayNode):
            # now(Y(phi), i) = prev(phi)
            # We still need to evaluate the child to update its state
            self._evaluate(node.child, labeling)
            child_idx = self._idx(node.child)
            val = self._prev[child_idx]
            self._now[idx] = val
            return val

        if isinstance(node, PreviouslyNode):
            # now(P(phi), i) = now(phi, i) OR prev(P(phi))
            child_val = self._evaluate(node.child, labeling)
            val = child_val or self._prev[idx]
            self._now[idx] = val
            return val

        if isinstance(node, HistoricallyNode):
            # now(H(phi), i) = now(phi, i) AND prev(H(phi))
            child_val = self._evaluate(node.child, labeling)
            val = child_val and self._prev[idx]
            self._now[idx] = val
            return val

        if isinstance(node, SinceNode):
            # now(phi S psi, i) = now(psi, i) OR (now(phi, i) AND prev(phi S psi))
            left_val = self._evaluate(node.left, labeling)
            right_val = self._evaluate(node.right, labeling)
            val = right_val or (left_val and self._prev[idx])
            self._now[idx] = val
            return val

        raise TypeError(f"Unknown AST node type: {type(node)}")

    def step(self, labeling: dict[str, bool]) -> bool:
        """Process one message event.

        Args:
            labeling: Map from prop_id -> True/False for this message.

        Returns:
            Current verdict: True = satisfied, False = violated.
        """
        result = self._evaluate(self._formula, labeling)
        # Shift: now becomes prev for next step
        self._prev = self._now[:]
        self._step_count += 1
        return result

    def reset(self) -> None:
        """Reset monitor state (new conversation)."""
        n = len(self._nodes)
        self._now = [False] * n
        self._prev = [False] * n
        self._step_count = 0
        self._init_prev()

    @property
    def verdict(self) -> bool:
        """Current formula verdict.

        Returns True if no steps have been processed (vacuously true)
        or the last step's evaluation result.
        """
        if self._step_count == 0:
            return True
        root_idx = self._idx(self._formula)
        return self._prev[root_idx]

    @property
    def state_snapshot(self) -> dict[str, Any]:
        """Serializable snapshot of monitor state for debugging/UI display."""
        states: dict[str, Any] = {}
        for i, node in enumerate(self._nodes):
            key = self._node_label(node)
            states[key] = {
                "now": self._prev[i],  # After step, prev holds the last "now"
                "node_type": type(node).__name__,
            }
        return {
            "step_count": self._step_count,
            "verdict": self.verdict,
            "sub_formulas": states,
        }

    @staticmethod
    def _node_label(node: ASTNode) -> str:
        """Generate a human-readable label for an AST node."""
        if isinstance(node, PropNode):
            return node.prop_id
        if isinstance(node, BoolNode):
            return str(node.value).lower()
        if isinstance(node, NotNode):
            return f"!({PtLTLMonitor._node_label(node.child)})"
        if isinstance(node, BinOpNode):
            left = PtLTLMonitor._node_label(node.left)
            right = PtLTLMonitor._node_label(node.right)
            return f"({left} {node.op} {right})"
        if isinstance(node, YesterdayNode):
            return f"Y({PtLTLMonitor._node_label(node.child)})"
        if isinstance(node, PreviouslyNode):
            return f"P({PtLTLMonitor._node_label(node.child)})"
        if isinstance(node, HistoricallyNode):
            return f"H({PtLTLMonitor._node_label(node.child)})"
        if isinstance(node, SinceNode):
            left = PtLTLMonitor._node_label(node.left)
            right = PtLTLMonitor._node_label(node.right)
            return f"({left} S {right})"
        return f"<unknown:{type(node).__name__}>"
