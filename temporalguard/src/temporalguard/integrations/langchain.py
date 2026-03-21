"""LangChain callback handler for TemporalGuard."""
from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

if TYPE_CHECKING:
    from temporalguard.guard import TemporalGuard
    from temporalguard.policy import Verdict
    from temporalguard.session import Session


class TemporalGuardViolation(Exception):
    """Raised when a policy violation is detected during a LangChain call."""

    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        violations = ", ".join(v.policy_name for v in verdict.violations)
        super().__init__(f"Policy violation: {violations}")


class TemporalGuardCallback(AsyncCallbackHandler):
    """Async LangChain callback that checks messages against TemporalGuard policies."""

    def __init__(self, guard: TemporalGuard, session_id: str | None = None) -> None:
        self._session: Session = guard.session(session_id=session_id)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Check incoming user messages against policies."""
        for message_list in messages:
            for msg in message_list:
                if hasattr(msg, "type") and msg.type == "human":
                    verdict = await self._session.check("user", msg.content)
                    if not verdict.passed:
                        raise TemporalGuardViolation(verdict)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Check LLM output against policies."""
        for generation_list in response.generations:
            for generation in generation_list:
                text = generation.text
                if text:
                    verdict = await self._session.check("assistant", text)
                    if not verdict.passed:
                        raise TemporalGuardViolation(verdict)
