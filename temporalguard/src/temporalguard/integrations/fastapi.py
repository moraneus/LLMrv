"""FastAPI middleware for TemporalGuard runtime verification."""
from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from temporalguard.guard import TemporalGuard
    from temporalguard.policy import Verdict


class TemporalGuardMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        guard: TemporalGuard,
        chat_endpoint: str = "/api/chat",
        session_resolver: Callable[[Request], str | None] | None = None,
        on_violation: Callable[[Verdict], Response] | None = None,
    ) -> None:
        super().__init__(app)
        self._guard = guard
        self._chat_endpoint = chat_endpoint
        self._session_resolver = session_resolver
        self._on_violation = on_violation
        self._sessions: dict[str, Any] = {}

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != self._chat_endpoint:
            return await call_next(request)

        body = await request.body()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        role = data.get("role", "user")
        content = data.get("content", data.get("message", ""))
        if not content:
            return await call_next(request)

        session_id = None
        if self._session_resolver:
            session_id = self._session_resolver(request)

        session_key = session_id or "default"
        if session_key not in self._sessions:
            self._sessions[session_key] = self._guard.session(session_id=session_id)
        session = self._sessions[session_key]

        verdict = await session.check(role, content)
        if not verdict.passed:
            if self._on_violation:
                return self._on_violation(verdict)
            return JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "violations": [
                        {"policy_name": v.policy_name, "formula": v.formula}
                        for v in verdict.violations
                    ],
                },
            )

        return await call_next(request)
