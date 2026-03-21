"""OpenAI SDK wrapper for TemporalGuard."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from temporalguard.session import Session
    from temporalguard.policy import Verdict


class GuardedChatViolation(Exception):
    """Raised when a policy violation is detected during guarded chat."""

    def __init__(self, verdict: Verdict, phase: str = "user") -> None:
        self.verdict = verdict
        self.phase = phase
        violations = ", ".join(v.policy_name for v in verdict.violations)
        super().__init__(f"Policy violation ({phase}): {violations}")


async def guarded_chat(
    session: Session,
    client: Any,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Send messages through an OpenAI client with TemporalGuard policy checks.

    Checks all user messages against the session's policies before calling the
    LLM, then checks the assistant response after receiving it.

    Args:
        session: A TemporalGuard session for policy monitoring.
        client: An async OpenAI client instance.
        messages: The chat messages to send.
        **kwargs: Additional arguments forwarded to ``client.chat.completions.create``.

    Returns:
        The OpenAI chat completion response.

    Raises:
        GuardedChatViolation: If a policy violation is detected in user input
            or assistant output.
    """
    # Check user messages
    for msg in messages:
        if msg.get("role") == "user":
            verdict = await session.check("user", msg["content"])
            if not verdict.passed:
                raise GuardedChatViolation(verdict, phase="user")

    # Call the LLM
    response = await client.chat.completions.create(messages=messages, **kwargs)

    # Check assistant response
    for choice in response.choices:
        text = choice.message.content
        if text:
            verdict = await session.check("assistant", text)
            if not verdict.passed:
                raise GuardedChatViolation(verdict, phase="assistant")

    return response
