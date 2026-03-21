"""TemporalGuard -- main entry point for the SDK."""
from __future__ import annotations

import logging

from temporalguard.engine.grounding import GroundingMethod, LLMGrounding
from temporalguard.policy import Policy, Proposition
from temporalguard.session import Session

logger = logging.getLogger(__name__)


class TemporalGuard:
    """Main entry point. Holds shared config, creates Session instances."""

    def __init__(
        self,
        propositions: list[Proposition],
        policies: list[Policy],
        grounding: GroundingMethod,
        auto_generate_few_shots: bool = True,
    ) -> None:
        self._propositions = list(propositions)
        self._policies = list(policies)
        self._grounding = grounding

        if auto_generate_few_shots:
            self._maybe_generate_few_shots()

    @classmethod
    def from_yaml(
        cls,
        path: str,
        grounding: GroundingMethod,
        auto_generate_few_shots: bool = True,
    ) -> TemporalGuard:
        """Create from a YAML policy file."""
        from temporalguard.loader import load_yaml

        propositions, policies = load_yaml(path)
        return cls(
            propositions=propositions,
            policies=policies,
            grounding=grounding,
            auto_generate_few_shots=auto_generate_few_shots,
        )

    def session(self, session_id: str | None = None) -> Session:
        """Create a new monitoring session."""
        return Session(guard=self, session_id=session_id)

    def generate_few_shots(self) -> None:
        """Generate few-shot examples for propositions that lack them.

        This is called automatically during construction when
        ``auto_generate_few_shots=True`` (the default).  Call it explicitly
        if you constructed the guard with ``auto_generate_few_shots=False``.
        """
        if not isinstance(self._grounding, LLMGrounding):
            raise TypeError(
                "Few-shot generation requires LLMGrounding, "
                f"got {type(self._grounding).__name__}"
            )
        from temporalguard.grounding.few_shot import generate_few_shots

        generate_few_shots(self._propositions, self._grounding._call_llm)

    def _maybe_generate_few_shots(self) -> None:
        """Auto-generate few-shot examples if grounding supports it."""
        if not isinstance(self._grounding, LLMGrounding):
            return
        needs_generation = any(
            not p.few_shot_positive and not p.few_shot_negative
            for p in self._propositions
        )
        if not needs_generation:
            return
        try:
            self.generate_few_shots()
        except Exception:
            logger.warning(
                "Auto-generation of few-shot examples failed; "
                "propositions will use zero-shot grounding",
                exc_info=True,
            )
