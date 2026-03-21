"""TemporalGuard -- main entry point for the SDK."""
from __future__ import annotations

from temporalguard.engine.grounding import GroundingMethod
from temporalguard.policy import Policy, Proposition
from temporalguard.session import Session


class TemporalGuard:
    """Main entry point. Holds shared config, creates Session instances."""

    def __init__(
        self,
        propositions: list[Proposition],
        policies: list[Policy],
        grounding: GroundingMethod,
    ) -> None:
        self._propositions = list(propositions)
        self._policies = list(policies)
        self._grounding = grounding

    @classmethod
    def from_yaml(cls, path: str, grounding: GroundingMethod) -> TemporalGuard:
        """Create from a YAML policy file."""
        from temporalguard.loader import load_yaml

        propositions, policies = load_yaml(path)
        return cls(propositions=propositions, policies=policies, grounding=grounding)

    def session(self, session_id: str | None = None) -> Session:
        """Create a new monitoring session."""
        return Session(guard=self, session_id=session_id)
