"""Example FastAPI chat server with TemporalGuard middleware.

This app acts as a real chat server: it forwards messages to an LLM (Ollama)
and returns the LLM's response — but every message passes through TemporalGuard
first. If a policy is violated, the request is blocked before reaching the LLM.

Usage:
    # Standalone
    uvicorn examples.fastapi_app:app --host 0.0.0.0 --port 8000

    # Via Docker (recommended)
    cd docker && docker compose up --build
"""
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from temporalguard import TemporalGuard, LLMGrounding
from temporalguard.integrations.fastapi import TemporalGuardMiddleware

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
GROUNDING_BASE_URL = os.environ.get("GROUNDING_BASE_URL", "http://localhost:11434")
GROUNDING_MODEL = os.environ.get("GROUNDING_MODEL", "mistral")
CHAT_MODEL = os.environ.get("CHAT_MODEL", GROUNDING_MODEL)

# ---------------------------------------------------------------------------
# TemporalGuard setup — load policies from YAML
# ---------------------------------------------------------------------------
policy_path = os.path.join(os.path.dirname(__file__), "policies.yaml")
guard = TemporalGuard.from_yaml(
    policy_path,
    grounding=LLMGrounding(base_url=GROUNDING_BASE_URL, model=GROUNDING_MODEL),
)

# ---------------------------------------------------------------------------
# FastAPI app with TemporalGuard middleware
# ---------------------------------------------------------------------------
app = FastAPI(title="TemporalGuard Chat Server")

app.add_middleware(
    TemporalGuardMiddleware,
    guard=guard,
    chat_endpoint="/api/chat",
    # Resolve session from X-Session-ID header (optional)
    session_resolver=lambda req: req.headers.get("X-Session-ID"),
)


@app.post("/api/chat")
async def chat(request: dict):
    """Forward a message to the LLM and return its response.

    This endpoint is only reached if TemporalGuard passes the message.
    Expects: {"role": "user", "content": "..."}
    Returns: {"role": "assistant", "content": "..."}
    """
    content = request.get("content", request.get("message", ""))

    # Call Ollama to generate a response
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{GROUNDING_BASE_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            assistant_text = resp.json().get("message", {}).get("content", "")
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"LLM call failed: {e}"},
        )

    return {"role": "assistant", "content": assistant_text}


@app.get("/health")
async def health():
    return {"status": "ok"}
