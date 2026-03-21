"""SDK data classes for TemporalGuard policies, propositions, and verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from temporalguard.engine.ptltl import PropNode, parse

_BUILTIN_IDS = {"user_turn", "true", "false"}


def _extract_prop_ids(formula: str) -> list[str]:
    """Extract proposition IDs from a ptLTL formula, excluding builtins."""
    ast = parse(formula)
    ids: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, PropNode):
            if node.prop_id not in _BUILTIN_IDS:
                ids.add(node.prop_id)
        for attr in ("child", "left", "right"):
            child = getattr(node, attr, None)
            if child is not None:
                _walk(child)

    _walk(ast)
    return sorted(ids)


@dataclass
class Proposition:
    """An atomic proposition that can be evaluated against a conversation turn."""

    prop_id: str
    role: str  # "user" | "assistant"
    description: str
    few_shot_positive: list[str] = field(default_factory=list)
    few_shot_negative: list[str] = field(default_factory=list)


@dataclass
class Policy:
    """A named ptLTL policy with its formula and associated propositions."""

    name: str
    formula: str
    propositions: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.propositions:
            self.propositions = _extract_prop_ids(self.formula)


@dataclass
class ViolationInfo:
    """Details about a single policy violation."""

    policy_name: str
    formula: str
    violated_at_index: int
    labeling: dict[str, bool] = field(default_factory=dict)
    grounding_details: list[dict] = field(default_factory=list)


@dataclass
class Verdict:
    """Result of checking a message against all policies."""

    passed: bool
    violations: list[ViolationInfo]
    per_policy: dict[str, bool]
    labeling: dict[str, bool]
    grounding_details: list[dict]
    trace_index: int

    @property
    def violation(self) -> ViolationInfo | None:
        """First violation, or None if all passed."""
        return self.violations[0] if self.violations else None
