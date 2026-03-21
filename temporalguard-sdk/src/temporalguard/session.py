"""Session -- per-conversation monitoring state."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from temporalguard.engine.monitor import ConversationMonitor
from temporalguard.engine.trace import MessageEvent
from temporalguard.policy import Verdict

if TYPE_CHECKING:
    from temporalguard.guard import TemporalGuard


class Session:
    """A single monitoring session tied to a conversation."""

    def __init__(self, guard: TemporalGuard, session_id: str | None = None) -> None:
        self._guard = guard
        self._session_id = session_id or str(uuid.uuid4())
        self._monitor = ConversationMonitor(
            policies=guard._policies,
            propositions=guard._propositions,
            grounding=guard._grounding,
            session_id=self._session_id,
        )

    async def check(self, role: str, text: str) -> Verdict:
        """Check a message against all policies."""
        return await self._monitor.process_message(role, text)

    def reset(self) -> None:
        """Reset session state."""
        self._monitor.reset()

    @property
    def trace(self) -> list[MessageEvent]:
        """Return the list of message events in this session."""
        return self._monitor.trace.messages

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id
