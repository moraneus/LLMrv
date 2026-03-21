"""TemporalGuard -- main entry point for the SDK."""
from __future__ import annotations

import logging
from pathlib import Path

from temporalguard.cache import DEFAULT_CACHE_DIR, load_cache, save_cache
from temporalguard.engine.grounding import GroundingMethod, LLMGrounding
from temporalguard.policy import Policy, Proposition
from temporalguard.session import Session

logger = logging.getLogger(__name__)


class TemporalGuard:
    """Main entry point. Holds shared config, creates Session instances.

    On construction the SDK ensures that every proposition has few-shot
    examples for accurate grounding:

    1. Check the on-disk cache (``<cache_dir>/<fingerprint>.yaml``)
    2. If cached examples exist → load them (no LLM call)
    3. Otherwise → auto-generate via the grounding LLM, then save to cache

    This means the grounding LLM is only called **once** per unique set of
    propositions/policies, even across process restarts.
    """

    def __init__(
        self,
        propositions: list[Proposition],
        policies: list[Policy],
        grounding: GroundingMethod,
        auto_generate_few_shots: bool = True,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._propositions = list(propositions)
        self._policies = list(policies)
        self._grounding = grounding
        self._cache_dir = Path(cache_dir)

        if auto_generate_few_shots:
            self._maybe_generate_few_shots()

    @classmethod
    def from_yaml(
        cls,
        path: str,
        grounding: GroundingMethod,
        auto_generate_few_shots: bool = True,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> TemporalGuard:
        """Create from a YAML policy file."""
        from temporalguard.loader import load_yaml

        propositions, policies = load_yaml(path)
        return cls(
            propositions=propositions,
            policies=policies,
            grounding=grounding,
            auto_generate_few_shots=auto_generate_few_shots,
            cache_dir=cache_dir,
        )

    def session(self, session_id: str | None = None) -> Session:
        """Create a new monitoring session."""
        return Session(guard=self, session_id=session_id)

    def generate_few_shots(self) -> None:
        """Generate few-shot examples for propositions that lack them.

        Generated examples are automatically saved to the cache directory.
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
        self._save_cache()

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

        # Try loading from cache first
        if load_cache(self._propositions, self._policies, self._cache_dir):
            # Check if cache covered everything
            still_needs = any(
                not p.few_shot_positive and not p.few_shot_negative
                for p in self._propositions
            )
            if not still_needs:
                return

        # Generate what's still missing, then save
        try:
            from temporalguard.grounding.few_shot import generate_few_shots

            generate_few_shots(self._propositions, self._grounding._call_llm)
            self._save_cache()
        except Exception:
            logger.warning(
                "Auto-generation of few-shot examples failed; "
                "propositions will use zero-shot grounding",
                exc_info=True,
            )

    def _save_cache(self) -> None:
        """Persist current propositions + policies to the cache directory."""
        try:
            save_cache(self._propositions, self._policies, self._cache_dir)
        except Exception:
            logger.warning("Failed to save few-shot cache", exc_info=True)
