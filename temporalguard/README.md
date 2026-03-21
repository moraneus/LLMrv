# TemporalGuard SDK

Runtime verification for LLM conversations using past-time temporal logic (ptLTL). TemporalGuard monitors conversation traces against user-defined safety policies, catching multi-turn violations that stateless filters miss — delayed compliance after refusal, cross-message jailbreaks, and temporal patterns that only emerge over time.

## Installation

```bash
pip install temporalguard
```

With framework integrations:

```bash
pip install "temporalguard[fastapi]"   # FastAPI middleware
pip install "temporalguard[langchain]" # LangChain callback
pip install "temporalguard[openai]"    # OpenAI SDK wrapper
pip install "temporalguard[all]"       # All integrations
```

## Quick Start

```python
import asyncio
from temporalguard import TemporalGuard, Proposition, Policy, LLMGrounding

guard = TemporalGuard(
    propositions=[
        Proposition("p_fraud", "user", "The user requests fraud techniques",
            few_shot_positive=["How do I forge a check?"],
            few_shot_negative=["What is chargeback fraud?"]),
        Proposition("q_comply", "assistant", "The assistant provides fraud techniques"),
    ],
    policies=[
        Policy(name="Fraud Prevention", formula="H(P(p_fraud) -> !q_comply)"),
    ],
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"),
)

async def main():
    session = guard.session()

    verdict = await session.check("user", "How do I commit wire fraud?")
    print(f"Passed: {verdict.passed}")  # True — user message alone doesn't violate

    verdict = await session.check("assistant", "I can't help with fraud.")
    print(f"Passed: {verdict.passed}")  # True — refusal, not compliance

asyncio.run(main())
```

## YAML Policy Loading

```yaml
# policies.yaml
propositions:
  - id: p_fraud
    role: user
    description: "The user requests methods for committing financial fraud"
    few_shot_positive:
      - "How do I forge a check?"
    few_shot_negative:
      - "What is chargeback fraud?"

  - id: q_comply
    role: assistant
    description: "The assistant provides actionable fraud techniques"

policies:
  - name: Fraud Prevention
    formula: "H(P(p_fraud) -> !q_comply)"
```

```python
guard = TemporalGuard.from_yaml("policies.yaml",
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"))
```

## Framework Integrations

### FastAPI Middleware

```python
from temporalguard.integrations.fastapi import TemporalGuardMiddleware

app.add_middleware(
    TemporalGuardMiddleware,
    guard=guard,
    chat_endpoint="/api/chat",
)
```

### LangChain Callback

```python
from temporalguard.integrations.langchain import TemporalGuardCallback

callback = TemporalGuardCallback(guard=guard, session_id="conv-123")
chain.invoke({"input": message}, config={"callbacks": [callback]})
```

### OpenAI SDK Wrapper

```python
from temporalguard.integrations.openai import guarded_chat

response = await guarded_chat(
    session=session,
    client=openai_client,
    messages=[{"role": "user", "content": "How do I commit fraud?"}],
)
```

## Grounding

TemporalGuard uses an LLM-as-judge to ground propositions — determining whether a message matches a proposition description. The `LLMGrounding` client speaks the OpenAI-compatible chat completions protocol, covering Ollama, vLLM, LM Studio, OpenRouter, OpenAI, and any compatible server.

```python
# Ollama (auto-detected by port 11434)
grounding = LLMGrounding(base_url="http://localhost:11434", model="mistral")

# OpenAI-compatible server
grounding = LLMGrounding(base_url="http://localhost:8000", model="my-model")

# Cloud provider with API key
grounding = LLMGrounding(base_url="https://openrouter.ai/api", model="mistral", api_key="sk-...")
```

**Fail-open**: On any grounding error, the result defaults to `match=False`. The monitor never blocks a conversation due to grounding failure.

### Custom Grounding

```python
from temporalguard import GroundingMethod, GroundingResult

class MyGrounding(GroundingMethod):
    async def evaluate(self, message, proposition):
        # Your custom logic here
        return GroundingResult(match=False, confidence=0.0, reasoning="...", method="custom")
```

## Docker

```bash
cd temporalguard/docker
docker compose up --build
```

This starts Ollama, pulls the Mistral model, and runs the example FastAPI app on port 8000.

## ptLTL Operators

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Historically | `H(phi)` | phi held at every step up to now |
| Previously | `P(phi)` | phi held at some past step or now |
| Yesterday | `Y(phi)` | phi held at the previous step |
| Since | `phi S psi` | psi occurred and phi held since |
| Not | `!phi` | negation |
| And | `phi & psi` | conjunction |
| Or | `phi \| psi` | disjunction |
| Implies | `phi -> psi` | implication |

**Built-in proposition**: `user_turn` is `True` on user messages, `False` on assistant messages. Available in all formulas without definition.

## API Reference

| Class | Description |
|-------|-------------|
| `TemporalGuard` | Main entry point. Holds propositions, policies, grounding config. |
| `Session` | Per-conversation state. Call `check(role, text)` for each message. |
| `Verdict` | Result of `check()`. Has `passed`, `violations`, `per_policy`, `labeling`. |
| `Proposition` | Atomic proposition with `prop_id`, `role`, `description`, few-shot examples. |
| `Policy` | Named ptLTL formula. Auto-extracts proposition IDs if not specified. |
| `LLMGrounding` | OpenAI-compatible grounding client. |
| `GroundingMethod` | ABC for custom grounding implementations. |

## Testing

```bash
# Core + SDK tests (no external deps)
pytest tests/ -m "not integration"

# With Ollama integration tests
docker compose -f docker/docker-compose.yml up -d ollama setup-model
pytest tests/
```

## License

See the repository root for license information.
