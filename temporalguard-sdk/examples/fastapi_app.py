"""Example FastAPI app with TemporalGuard middleware."""
import os
from fastapi import FastAPI
from temporalguard import TemporalGuard, LLMGrounding
from temporalguard.integrations.fastapi import TemporalGuardMiddleware

policy_path = os.path.join(os.path.dirname(__file__), "policies.yaml")
guard = TemporalGuard.from_yaml(policy_path,
    grounding=LLMGrounding(
        base_url=os.environ.get("GROUNDING_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("GROUNDING_MODEL", "mistral")))

app = FastAPI(title="TemporalGuard Example")
app.add_middleware(TemporalGuardMiddleware, guard=guard, chat_endpoint="/api/chat")

@app.post("/api/chat")
async def chat(request: dict):
    return {"response": "This is a placeholder response."}

@app.get("/health")
async def health():
    return {"status": "ok"}
