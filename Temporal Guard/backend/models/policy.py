"""
Pydantic models for propositions, policies, and monitor verdicts.

Defines the core data structures used by the grounding engine,
ptLTL monitor, and monitor orchestrator.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Proposition(BaseModel):
    """An atomic proposition for semantic grounding.

    Attributes:
        prop_id: Unique identifier (e.g., "p_weapon").
        description: Canonical description for grounding (delta_p).
        role: Which message role this proposition applies to ("user" or "assistant").
    """

    prop_id: str
    description: str
    role: str  # "user" | "assistant"
    few_shot_positive: list[str] = Field(default_factory=list)
    few_shot_negative: list[str] = Field(default_factory=list)
    few_shot_generated_at: str | None = None


class Policy(BaseModel):
    """A ptLTL safety policy referencing propositions.

    Attributes:
        policy_id: Unique identifier for the policy.
        name: Human-readable name.
        formula_str: ptLTL formula string.
        propositions: List of prop_ids referenced in the formula.
        enabled: Whether the policy is actively monitored.
    """

    policy_id: str
    name: str
    formula_str: str
    propositions: list[str] = Field(default_factory=list)
    enabled: bool = True


class ViolationInfo(BaseModel):
    """Details about a policy violation.

    Attributes:
        policy_id: Which policy was violated.
        policy_name: Human-readable name of the violated policy.
        formula_str: The ptLTL formula that was violated.
        violated_at_index: Trace index where violation occurred.
        labeling: Proposition truth values at the violating step.
        grounding_details: Reasoning from the grounding engine.
    """

    policy_id: str
    policy_name: str
    formula_str: str
    violated_at_index: int
    labeling: dict[str, bool] = Field(default_factory=dict)
    grounding_details: list[dict] = Field(default_factory=list)


class MonitorVerdict(BaseModel):
    """Result of processing a message through the monitor pipeline.

    Attributes:
        passed: True if all policies are satisfied, False if any violated.
        per_policy: Verdict for each policy (policy_id -> bool).
        labeling: Proposition truth values at this step.
        grounding_details: Detailed grounding results per proposition.
        trace_index: Position in the conversation trace.
        violations: List of violation details (empty if passed).
    """

    passed: bool
    per_policy: dict[str, bool] = Field(default_factory=dict)
    labeling: dict[str, bool] = Field(default_factory=dict)
    grounding_details: list[dict] = Field(default_factory=list)
    trace_index: int = 0
    violations: list[ViolationInfo] = Field(default_factory=list)
