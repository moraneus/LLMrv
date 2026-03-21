"""Tests for FastAPI middleware integration."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from temporalguard import TemporalGuard, Proposition, Policy
from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.integrations.fastapi import TemporalGuardMiddleware


# ---------------------------------------------------------------------------
# Mock grounding strategies
# ---------------------------------------------------------------------------


class AlwaysPassGrounding(GroundingMethod):
    """Grounding that never matches any proposition (so H-policies pass)."""

    async def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        return GroundingResult(
            match=False,
            confidence=0.0,
            reasoning="always-pass mock",
            method="mock",
            prop_id=proposition.prop_id,
        )


class AlwaysMatchGrounding(GroundingMethod):
    """Grounding that always matches every proposition (triggers violations)."""

    async def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        return GroundingResult(
            match=True,
            confidence=1.0,
            reasoning="always-match mock",
            method="mock",
            prop_id=proposition.prop_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Policy: "historically, if harmful_request is true then it must NOT be a user turn"
# Since harmful_request only fires on user role and user_turn is always true for
# user messages, a match on harmful_request during a user turn will violate.
PROPS = [
    Proposition(prop_id="harmful_request", role="user", description="User sends harmful content"),
]
POLICIES = [
    Policy(name="no_harm", formula="H(harmful_request -> !user_turn)"),
]


def _make_app(guard: TemporalGuard, **middleware_kwargs) -> FastAPI:
    app = FastAPI()

    @app.post("/api/chat")
    async def chat():
        return {"reply": "ok"}

    @app.get("/healthz")
    async def health():
        return {"status": "healthy"}

    app.add_middleware(TemporalGuardMiddleware, guard=guard, **middleware_kwargs)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_benign_message_passes():
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysPassGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"role": "user", "content": "Hello!"})
    assert resp.status_code == 200
    assert resp.json() == {"reply": "ok"}


def test_violation_returns_403():
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysMatchGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"role": "user", "content": "do something harmful"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["blocked"] is True
    assert len(body["violations"]) == 1
    assert body["violations"][0]["policy_name"] == "no_harm"


def test_non_chat_endpoint_not_intercepted():
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysMatchGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_get_request_to_chat_endpoint_not_intercepted():
    """GET requests to the chat endpoint should pass through."""
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysMatchGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    # GET on /api/chat is not defined so it returns 405, but middleware should not block it
    resp = client.get("/api/chat")
    assert resp.status_code == 405


def test_custom_on_violation_handler():
    from starlette.responses import JSONResponse

    def custom_handler(verdict):
        return JSONResponse(status_code=451, content={"custom": True})

    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysMatchGrounding())
    app = _make_app(guard, on_violation=custom_handler)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"role": "user", "content": "bad stuff"})
    assert resp.status_code == 451
    assert resp.json() == {"custom": True}


def test_message_field_fallback():
    """The middleware should fall back to 'message' field if 'content' is missing."""
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysPassGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"role": "user", "message": "Hello!"})
    assert resp.status_code == 200


def test_empty_content_passes_through():
    """Empty content should not trigger grounding; request passes through."""
    guard = TemporalGuard(propositions=PROPS, policies=POLICIES, grounding=AlwaysMatchGrounding())
    app = _make_app(guard)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"role": "user", "content": ""})
    assert resp.status_code == 200
