# TemporalGuard SDK

**Runtime verification for LLM conversations using past-time temporal logic (ptLTL).**

TemporalGuard monitors conversation traces against user-defined safety policies, catching multi-turn violations that stateless filters miss — delayed compliance after refusal, cross-message jailbreaks, and temporal patterns that only emerge over time.

> *"A single message filter can't see that the assistant refused once, then complied two turns later. TemporalGuard can."*

---

## Table of Contents

- [Why TemporalGuard?](#why-temporalguard)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Propositions](#propositions)
  - [Policies & ptLTL Formulas](#policies--ptltl-formulas)
  - [Grounding](#grounding)
  - [Sessions & Verdicts](#sessions--verdicts)
- [Defining Policies in YAML](#defining-policies-in-yaml)
- [ptLTL Operator Reference](#ptltl-operator-reference)
- [Writing Effective Formulas](#writing-effective-formulas)
- [Grounding Configuration](#grounding-configuration)
  - [Supported Providers](#supported-providers)
  - [Custom Grounding](#custom-grounding)
- [Framework Integrations](#framework-integrations)
  - [FastAPI Middleware](#fastapi-middleware)
  - [LangChain Callback](#langchain-callback)
  - [OpenAI SDK Wrapper](#openai-sdk-wrapper)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why TemporalGuard?

Most LLM safety tools operate on **single messages** — they check if one message is harmful, then forget about it. But real-world attacks exploit **multi-turn dynamics**:

| Attack Pattern | Single-Message Filter | TemporalGuard |
|---|---|---|
| User asks for harm, assistant refuses, user rephrases, assistant complies | Misses the compliance (no memory of prior refusal) | Detects the temporal pattern: *"if harm was ever requested, compliance must never follow"* |
| Gradual context manipulation across 5+ turns | Each message looks benign in isolation | Tracks accumulated state: *"these propositions have been true since turn 2"* |
| Assistant initially refuses, then leaks info in a follow-up | Filter sees the follow-up as a standalone response | Enforces: *"historically, if a refusal happened, no compliance should follow"* |

**TemporalGuard uses past-time Linear Temporal Logic (ptLTL)** — a formal verification method — to express and enforce properties over the *entire conversation history*, not just the current message.

---

## How It Works

TemporalGuard operates in three steps on every message:

```
                    ┌──────────────────────────────────────────────┐
  New message ──────►  1. GROUND  ──►  2. LABEL  ──►  3. VERIFY   │
  (role + text)     │                                              │
                    │  LLM-as-judge    Assign T/F    Step ptLTL    │
                    │  evaluates each  to each        monitors     │
                    │  proposition     proposition    forward       │
                    │                                              │
                    │                                  ┌───────┐   │
                    │                                  │Verdict│   │
                    │                                  │pass/  │   │
                    │                                  │fail   │   │
                    └──────────────────────────────────┴───────┘   │
                                                                   │
```

1. **Ground** — For each proposition relevant to the current role (user or assistant), an LLM-as-judge determines whether the message matches the proposition's description. Few-shot examples improve accuracy.

2. **Label** — The grounding results become a boolean labeling: `{p_fraud: True, q_comply: False, user_turn: True}`.

3. **Verify** — Each ptLTL policy monitor steps forward with the new labeling. If any policy evaluates to `False`, the verdict includes the violation details.

---

## Architecture

### SDK Integration Within Your System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          YOUR APPLICATION                               │
│                                                                         │
│  ┌──────────┐     ┌───────────────────────────────────────────────┐     │
│  │          │     │              TemporalGuard SDK                │     │
│  │  User    │     │                                               │     │
│  │  Input   ├────►│  ┌─────────────┐    ┌──────────────────────┐ │     │
│  │          │     │  │   Session    │    │   ptLTL Engine       │ │     │
│  └──────────┘     │  │             │    │                      │ │     │
│                   │  │  check()  ──┼───►│  Recursive-descent   │ │     │
│  ┌──────────┐     │  │             │    │  parser + incremental│ │     │
│  │          │     │  │  trace      │    │  monitor per policy  │ │     │
│  │  LLM     │     │  │  reset()   │    │                      │ │     │
│  │  (your   │     │  └──────┬──────┘    └──────────────────────┘ │     │
│  │  model)  │     │         │                                    │     │
│  │          │     │         ▼                                    │     │
│  └────┬─────┘     │  ┌──────────────┐   ┌──────────────────────┐ │     │
│       │           │  │  Grounding   │   │   Policy Definitions │ │     │
│       │           │  │              │   │                      │ │     │
│       │           │  │  LLM-as-     │   │  Propositions        │ │     │
│       │           │  │  judge via ──┼──►│  + ptLTL formulas    │ │     │
│       │           │  │  HTTP        │   │  (Python or YAML)    │ │     │
│       │           │  └──────┬───────┘   └──────────────────────┘ │     │
│       │           │         │                                    │     │
│       │           └─────────┼────────────────────────────────────┘     │
│       │                     │                                          │
│       │                     ▼                                          │
│       │           ┌─────────────────┐                                  │
│       │           │  Grounding LLM  │  (Ollama, vLLM, OpenAI,         │
│       │           │  (separate from │   OpenRouter, LM Studio, etc.)  │
│       └──────────►│   your LLM)    │                                  │
│                   └─────────────────┘                                  │
│                                                                        │
│  ┌──────────┐     ┌──────────────────┐                                 │
│  │ Verdict  │◄────│  passed: bool    │  If violation detected:         │
│  │          │     │  violations: []  │  → block, log, alert, or       │
│  │          │     │  per_policy: {}  │    let through with warning     │
│  └──────────┘     │  labeling: {}    │                                 │
│                   └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Integration Patterns

TemporalGuard fits into your stack at the **middleware/callback layer** — between user input and your LLM:

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   Pattern A: FastAPI Middleware (intercepts HTTP requests)          │
│                                                                    │
│   Client ──► FastAPI ──► TemporalGuardMiddleware ──► Your Route    │
│                              │                                     │
│                              ├─ PASS → forward to route handler    │
│                              └─ FAIL → return 403 + violations     │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Pattern B: LangChain Callback (hooks into chain execution)       │
│                                                                    │
│   chain.invoke() ──► on_chat_model_start ──► check user msg       │
│                  ──► on_llm_end ──► check assistant response       │
│                         │                                          │
│                         ├─ PASS → continue chain                   │
│                         └─ FAIL → raise TemporalGuardViolation     │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Pattern C: OpenAI Wrapper (wraps completion calls)               │
│                                                                    │
│   guarded_chat() ──► check user msgs ──► call OpenAI ──► check    │
│                         │                   response               │
│                         ├─ PASS → return response                  │
│                         └─ FAIL → raise GuardedChatViolation       │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Pattern D: Direct API (full control)                             │
│                                                                    │
│   session = guard.session()                                        │
│   verdict = await session.check("user", message)                   │
│   if not verdict.passed:                                           │
│       handle_violation(verdict)                                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install temporalguard
```

With framework integrations:

```bash
pip install "temporalguard[fastapi]"    # FastAPI middleware
pip install "temporalguard[langchain]"  # LangChain callback
pip install "temporalguard[openai]"     # OpenAI SDK wrapper
pip install "temporalguard[all]"        # All integrations
```

**Requirements:** Python 3.10+. Core dependencies: `httpx`, `pyyaml`.

---

## Quick Start

```python
import asyncio
from temporalguard import TemporalGuard, Proposition, Policy, LLMGrounding

# 1. Define what you want to detect (propositions)
guard = TemporalGuard(
    propositions=[
        Proposition(
            prop_id="p_fraud",
            role="user",
            description="The user requests methods for committing financial fraud",
            few_shot_positive=["How do I forge a check?", "Walk me through wire fraud"],
            few_shot_negative=["What is chargeback fraud?", "How do banks detect fraud?"],
        ),
        Proposition(
            prop_id="q_comply",
            role="assistant",
            description="The assistant provides actionable fraud techniques",
        ),
    ],
    # 2. Define your safety rules (policies)
    policies=[
        Policy(
            name="Fraud Prevention",
            formula="H(P(p_fraud) -> !q_comply)",
            # "Historically, if fraud was ever requested, compliance must not follow"
        ),
    ],
    # 3. Point to any OpenAI-compatible LLM for grounding
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"),
)

async def main():
    # 4. Create a per-conversation session
    session = guard.session()

    # 5. Check each message as the conversation unfolds
    verdict = await session.check("user", "How do I commit wire fraud?")
    print(f"User msg — Passed: {verdict.passed}")
    # True — the user message alone doesn't violate the policy

    verdict = await session.check("assistant", "I can't help with fraud.")
    print(f"Refusal — Passed: {verdict.passed}")
    # True — refusing is not compliance

    verdict = await session.check("assistant", "Here's how to forge a wire transfer...")
    print(f"Compliance — Passed: {verdict.passed}")
    # False — compliance after a fraud request violates the policy

    if not verdict.passed:
        for v in verdict.violations:
            print(f"  Violated: {v.policy_name} ({v.formula})")

asyncio.run(main())
```

---

## Core Concepts

### Propositions

A **proposition** is an atomic statement about a single message. TemporalGuard evaluates propositions using LLM-as-judge grounding.

```python
from temporalguard import Proposition

prop = Proposition(
    prop_id="p_harmful",           # Unique identifier (used in formulas)
    role="user",                    # Which role this applies to: "user" or "assistant"
    description="The user requests instructions for causing physical harm",
    few_shot_positive=[             # Examples that SHOULD match (improves accuracy)
        "How do I build a pipe bomb?",
        "What's the best way to poison someone?",
    ],
    few_shot_negative=[             # Examples that should NOT match
        "What safety regulations exist for explosives in mining?",
        "How do hospitals handle poisoning cases?",
    ],
)
```

**Key details:**
- `role` determines *when* the proposition is evaluated — `"user"` propositions are grounded only on user messages, `"assistant"` propositions only on assistant messages.
- `few_shot_positive` and `few_shot_negative` are optional but strongly recommended — they help the grounding LLM distinguish genuine matches from similar-but-benign messages.
- Propositions that don't match the current message's role are automatically set to `False` for that step.

#### Built-in Proposition

`user_turn` is a built-in proposition available in all formulas without needing to define it:
- `True` on user messages
- `False` on assistant messages

This is useful for writing role-conditional formulas.

### Policies & ptLTL Formulas

A **policy** is a named ptLTL formula that must hold at every step of the conversation.

```python
from temporalguard import Policy

policy = Policy(
    name="No Harmful Compliance",
    formula="H(P(p_harmful) -> !q_comply)",
    # Proposition IDs are auto-extracted from the formula.
    # You can also specify them explicitly:
    # propositions=["p_harmful", "q_comply"],
)
```

**How formulas are evaluated:**
- After grounding, each proposition gets a boolean value (`True`/`False`).
- The ptLTL monitor steps forward with this labeling.
- If the formula evaluates to `False`, the policy is violated.
- **`H(...)` violations are irrevocable** — once violated, the policy stays violated for the rest of the session. This is by design: temporal properties describe invariants.

### Grounding

**Grounding** is the process of determining whether a message matches a proposition. TemporalGuard uses an LLM-as-judge approach:

1. The proposition's description, few-shot examples, and the message are sent to a grounding LLM.
2. The LLM returns `{match: true/false, confidence: 0.0-1.0, reasoning: "..."}`.
3. The `match` boolean becomes the proposition's truth value for this step.

**Fail-open guarantee:** If grounding fails for any reason (network error, parsing failure, timeout), the result defaults to `match=False`. The monitor **never blocks a conversation due to grounding failure**.

### Sessions & Verdicts

A **session** tracks one conversation's state. Each call to `check()` returns a **verdict**:

```python
session = guard.session()  # or guard.session(session_id="conv-123")

verdict = await session.check("user", "Some message")

# Verdict fields:
verdict.passed          # bool — True if all policies passed
verdict.violations      # list[ViolationInfo] — details of any failures
verdict.per_policy      # dict[str, bool] — per-policy pass/fail
verdict.labeling        # dict[str, bool] — proposition truth values this step
verdict.grounding_details  # list[dict] — raw grounding results
verdict.trace_index     # int — message index in the conversation

# Convenience:
verdict.violation       # ViolationInfo | None — first violation, or None

# ViolationInfo fields:
v = verdict.violations[0]
v.policy_name           # "Fraud Prevention"
v.formula               # "H(P(p_fraud) -> !q_comply)"
v.violated_at_index     # 3
v.labeling              # {"p_fraud": False, "q_comply": True, "user_turn": False}
v.grounding_details     # [{"match": True, "confidence": 0.95, ...}]
```

**Session lifecycle:**
- `session.check(role, text)` — process a message and get a verdict.
- `session.trace` — access the full message history.
- `session.reset()` — clear the session state (restarts all monitors).
- `session.session_id` — the session identifier.

---

## Defining Policies in YAML

For larger policy sets, define everything in YAML:

```yaml
# policies.yaml
propositions:
  - id: p_fraud
    role: user
    description: "The user requests methods for committing financial fraud"
    few_shot_positive:
      - "How do I forge a check?"
      - "Walk me through wire fraud step by step"
    few_shot_negative:
      - "What is chargeback fraud?"
      - "How do banks detect fraud?"

  - id: q_comply
    role: assistant
    description: "The assistant provides actionable fraud techniques"
    few_shot_positive:
      - "Here's how to forge a check: first, obtain blank check stock..."
    few_shot_negative:
      - "I can't help with fraud. Wire fraud is a federal crime."
      - "Fraud prevention involves multi-factor authentication..."

  - id: p_violence
    role: user
    description: "The user requests instructions for causing physical harm"

  - id: q_violence_comply
    role: assistant
    description: "The assistant provides instructions for violence"

policies:
  - name: Fraud Prevention
    formula: "H(P(p_fraud) -> !q_comply)"

  - name: Violence Prevention
    formula: "H(P(p_violence) -> !q_violence_comply)"
```

Load it in Python:

```python
from temporalguard import TemporalGuard, LLMGrounding

guard = TemporalGuard.from_yaml(
    "policies.yaml",
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"),
)
```

---

## ptLTL Operator Reference

| Operator | Syntax | Meaning | Example |
|---|---|---|---|
| **Historically** | `H(phi)` | `phi` has been true at every step up to and including now | `H(!q_harmful)` — assistant has never been harmful |
| **Previously** | `P(phi)` | `phi` was true at some past step or now | `P(p_fraud)` — fraud was requested at some point |
| **Yesterday** | `Y(phi)` | `phi` was true at the immediately previous step | `Y(p_request)` — the previous message was a request |
| **Since** | `phi S psi` | `psi` occurred at some past step, and `phi` has held at every step since | `!q_comply S p_refusal` — no compliance since the refusal |
| **Not** | `!phi` | Negation | `!q_comply` — assistant is not complying |
| **And** | `phi & psi` | Conjunction | `p_fraud & p_urgent` — both are true |
| **Or** | `phi \| psi` | Disjunction | `p_fraud \| p_violence` — either is true |
| **Implies** | `phi -> psi` | Implication (`!phi \| psi`) | `P(p_fraud) -> !q_comply` — if fraud was ever asked, no compliance |

**Operator precedence** (highest to lowest): `!`, `H`/`P`/`Y`, `&`, `|`, `S`, `->`.

Use parentheses for clarity: `H(P(p_fraud) -> !q_comply)`.

---

## Writing Effective Formulas

### Pattern: "Never comply after a harmful request"

```
H(P(p_harmful_request) -> !q_harmful_compliance)
```

*"Historically, if a harmful request was ever made (P), the assistant must not comply."*

This is the most common pattern. `H(...)` makes it an invariant. `P(...)` provides "memory" — once the request happened, it stays true forever.

### Pattern: "The very next response after X must be Y"

```
H(Y(p_request) -> q_appropriate_response)
```

*"Whenever the previous message was a request, the current message must be an appropriate response."*

Use `Y(...)` for "immediately previous step" relationships.

### Pattern: "Once X happens, Y must hold until Z"

```
H(P(p_started) -> (q_safe S p_resolved))
```

*"If the condition started, safety must have held continuously since it was resolved."*

### Cross-Role Formulas

**Important:** Since propositions for different roles are evaluated at different steps, use temporal operators (`P`, `Y`, `S`) to connect them:

```
# CORRECT: "If user ever requested fraud (user turn), assistant must never comply (assistant turn)"
H(P(p_fraud) -> !q_comply)

# WRONG: p_fraud and q_comply can never both be True at the same step
# (one is user-role, the other assistant-role)
H(!(p_fraud & q_comply))
```

### Using `user_turn`

The built-in `user_turn` proposition lets you write role-conditional formulas:

```
# "On every assistant turn, the assistant must not be harmful"
H(!user_turn -> !q_harmful)

# "On every user turn, if the user was harmful, the next turn must be a refusal"
H((user_turn & p_harmful) -> Y(!user_turn & q_refusal))
```

---

## Grounding Configuration

### Supported Providers

TemporalGuard's `LLMGrounding` speaks the **OpenAI-compatible chat completions protocol**, covering a wide range of providers:

```python
from temporalguard import LLMGrounding

# Ollama (auto-detected by port 11434 — uses native /api/chat endpoint)
grounding = LLMGrounding(
    base_url="http://localhost:11434",
    model="mistral",
)

# vLLM / LM Studio / any OpenAI-compatible server
grounding = LLMGrounding(
    base_url="http://localhost:8000",
    model="my-model",
)

# OpenAI
grounding = LLMGrounding(
    base_url="https://api.openai.com",
    model="gpt-4o-mini",
    api_key="sk-...",
)

# OpenRouter
grounding = LLMGrounding(
    base_url="https://openrouter.ai/api",
    model="mistralai/mistral-7b-instruct",
    api_key="sk-or-...",
)
```

**Protocol auto-detection:**
- If the URL contains `11434` or `/api/chat` → Ollama native format
- Otherwise → OpenAI-compatible `/v1/chat/completions`

**Custom prompts:** Override the default grounding prompt if needed:

```python
grounding = LLMGrounding(
    base_url="http://localhost:11434",
    model="mistral",
    system_prompt="You are a precise content classifier...",
    user_prompt_template="Evaluate: {message}",
)
```

### Custom Grounding

Implement your own grounding strategy by subclassing `GroundingMethod`:

```python
from temporalguard import GroundingMethod, GroundingResult
from temporalguard.engine.trace import MessageEvent
from temporalguard.policy import Proposition

class KeywordGrounding(GroundingMethod):
    """Simple keyword-based grounding (no LLM needed)."""

    def __init__(self, keyword_map: dict[str, list[str]]):
        self._keywords = keyword_map  # prop_id -> keywords

    def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        keywords = self._keywords.get(proposition.prop_id, [])
        text_lower = message.text.lower()
        matched = any(kw in text_lower for kw in keywords)
        return GroundingResult(
            match=matched,
            confidence=1.0 if matched else 0.0,
            reasoning=f"Keyword match: {matched}",
            method="keyword",
            prop_id=proposition.prop_id,
        )

# Use it:
guard = TemporalGuard(
    propositions=[...],
    policies=[...],
    grounding=KeywordGrounding({"p_fraud": ["wire fraud", "forge", "launder"]}),
)
```

This is useful for testing, for deterministic rules, or for combining with embedding-based approaches.

---

## Framework Integrations

### FastAPI Middleware

Intercepts chat requests, checks them against policies, and blocks violations with a 403 response.

```bash
pip install "temporalguard[fastapi]"
```

```python
from fastapi import FastAPI
from temporalguard import TemporalGuard, LLMGrounding
from temporalguard.integrations.fastapi import TemporalGuardMiddleware

guard = TemporalGuard.from_yaml("policies.yaml",
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"))

app = FastAPI()

# Add the middleware
app.add_middleware(
    TemporalGuardMiddleware,
    guard=guard,
    chat_endpoint="/api/chat",  # Which endpoint to monitor
)

@app.post("/api/chat")
async def chat(request: dict):
    # This handler is only reached if TemporalGuard passes
    return {"response": "Hello!"}
```

**Expected request format:**

```json
{"role": "user", "content": "Hello, how are you?"}
```

**On violation, the middleware returns:**

```json
{
  "blocked": true,
  "violations": [
    {"policy_name": "Fraud Prevention", "formula": "H(P(p_fraud) -> !q_comply)"}
  ]
}
```

**Advanced configuration:**

```python
from starlette.requests import Request
from starlette.responses import JSONResponse

app.add_middleware(
    TemporalGuardMiddleware,
    guard=guard,
    chat_endpoint="/api/chat",

    # Custom session resolution (e.g., from headers or cookies)
    session_resolver=lambda request: request.headers.get("X-Session-ID"),

    # Custom violation handler
    on_violation=lambda verdict: JSONResponse(
        status_code=400,
        content={"error": "Policy violation", "details": str(verdict.violations)},
    ),
)
```

### LangChain Callback

Hooks into LangChain's callback system to check both user inputs and LLM outputs.

```bash
pip install "temporalguard[langchain]"
```

```python
from langchain_openai import ChatOpenAI
from temporalguard import TemporalGuard, LLMGrounding
from temporalguard.integrations.langchain import TemporalGuardCallback, TemporalGuardViolation

guard = TemporalGuard.from_yaml("policies.yaml",
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"))

# Create callback (one per conversation)
callback = TemporalGuardCallback(guard=guard, session_id="conv-123")

llm = ChatOpenAI(model="gpt-4o")

try:
    response = await llm.ainvoke(
        "How do I commit fraud?",
        config={"callbacks": [callback]},
    )
except TemporalGuardViolation as e:
    print(f"Blocked: {e}")
    print(f"Violations: {e.verdict.violations}")
```

**What gets checked:**
- `on_chat_model_start` — checks all human messages before the LLM call
- `on_llm_end` — checks the assistant's response after the LLM call

### OpenAI SDK Wrapper

A thin wrapper around the OpenAI client that adds policy checks before and after each completion.

```bash
pip install "temporalguard[openai]"
```

```python
from openai import AsyncOpenAI
from temporalguard import TemporalGuard, LLMGrounding
from temporalguard.integrations.openai import guarded_chat, GuardedChatViolation

guard = TemporalGuard.from_yaml("policies.yaml",
    grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"))

client = AsyncOpenAI()
session = guard.session()

try:
    response = await guarded_chat(
        session=session,
        client=client,
        messages=[
            {"role": "user", "content": "How do I commit fraud?"},
        ],
        model="gpt-4o",
    )
    print(response.choices[0].message.content)
except GuardedChatViolation as e:
    print(f"Blocked at {e.phase} phase: {e}")
    # e.phase is "user" or "assistant" — tells you which side triggered the violation
```

---

## Docker

A Docker Compose setup is included that runs Ollama, pulls a model, and starts an example FastAPI app:

```bash
cd temporalguard-sdk/docker
docker compose up --build
```

This starts:
- **Ollama** on port 11434 with the Mistral model
- **Example FastAPI app** on port 8000

Test it:

```bash
# Should pass
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "What is financial fraud?"}'

# Should be blocked (after a fraud request was made)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"role": "assistant", "content": "Here is how to forge a check..."}'
```

---

## API Reference

### `TemporalGuard`

The main entry point. Holds propositions, policies, and grounding configuration.

| Method | Description |
|---|---|
| `TemporalGuard(propositions, policies, grounding)` | Create a guard with Python objects |
| `TemporalGuard.from_yaml(path, grounding)` | Create from a YAML policy file |
| `guard.session(session_id=None)` | Create a new monitoring session |

### `Session`

Per-conversation state. One session per active conversation.

| Method / Property | Description |
|---|---|
| `await session.check(role, text)` | Check a message, returns `Verdict` |
| `session.reset()` | Reset session state (clears trace, resets monitors) |
| `session.trace` | `list[MessageEvent]` — the message history |
| `session.session_id` | `str` — the session identifier |

### `Verdict`

Result of checking a message against all policies.

| Field | Type | Description |
|---|---|---|
| `passed` | `bool` | `True` if all policies passed |
| `violations` | `list[ViolationInfo]` | Details of any failures |
| `per_policy` | `dict[str, bool]` | Per-policy pass/fail map |
| `labeling` | `dict[str, bool]` | Proposition truth values this step |
| `grounding_details` | `list[dict]` | Raw grounding results |
| `trace_index` | `int` | Message index in the conversation |
| `violation` | `ViolationInfo \| None` | First violation (property) |

### `ViolationInfo`

| Field | Type | Description |
|---|---|---|
| `policy_name` | `str` | Name of the violated policy |
| `formula` | `str` | The ptLTL formula that was violated |
| `violated_at_index` | `int` | Trace index where violation occurred |
| `labeling` | `dict[str, bool]` | Labeling at the violation step |
| `grounding_details` | `list[dict]` | Grounding results at the violation step |

### `Proposition`

| Field | Type | Default | Description |
|---|---|---|---|
| `prop_id` | `str` | required | Unique identifier |
| `role` | `str` | required | `"user"` or `"assistant"` |
| `description` | `str` | required | Natural language description for grounding |
| `few_shot_positive` | `list[str]` | `[]` | Examples that should match |
| `few_shot_negative` | `list[str]` | `[]` | Examples that should not match |

### `Policy`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Human-readable policy name |
| `formula` | `str` | required | ptLTL formula |
| `propositions` | `list[str]` | auto-extracted | Proposition IDs used in the formula |
| `enabled` | `bool` | `True` | Whether the policy is active |

### `LLMGrounding`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | `"http://localhost:11434"` | LLM server URL |
| `model` | `str` | `"mistral"` | Model name |
| `api_key` | `str` | `""` | API key (for cloud providers) |
| `system_prompt` | `str` | `""` | Custom system prompt (overrides default) |
| `user_prompt_template` | `str` | `""` | Custom user prompt template |

### `GroundingMethod` (ABC)

Base class for custom grounding implementations.

| Method | Description |
|---|---|
| `evaluate(message, proposition) -> GroundingResult` | Evaluate whether message matches proposition |

### `GroundingResult`

| Field | Type | Description |
|---|---|---|
| `match` | `bool` | Whether the message matches the proposition |
| `confidence` | `float` | Confidence score (0.0 - 1.0) |
| `reasoning` | `str` | Explanation from the grounding method |
| `method` | `str` | Grounding method identifier |
| `prop_id` | `str` | Proposition ID that was evaluated |

---

## Testing

```bash
# Core + SDK tests (no external dependencies)
pytest tests/ -m "not integration"

# With Ollama running (integration tests)
docker compose -f docker/docker-compose.yml up -d ollama setup-model
pytest tests/
```

The test suite covers:
- **ptLTL engine** — parser, all operators, edge cases (132 tests)
- **Trace management** — message events, conversation traces (56 tests)
- **Runtime verification traces** — multi-turn scenarios (158 tests)
- **Policy/proposition data classes** — field defaults, extraction (11 tests)
- **Built-in propositions** — `user_turn` behavior (7 tests)
- **Grounding prompts** — prompt construction, few-shot rendering (9 tests)
- **Grounding engine** — LLM calls, fail-open, JSON parsing (22 tests)
- **Monitor orchestrator** — parallel grounding, labeling, verdicts (17 tests)
- **Guard + Session** — creation, YAML loading, check flow (12 tests)
- **YAML loader** — full and minimal configs, validation (15 tests)
- **Framework integrations** — FastAPI, LangChain, OpenAI (20 tests)
- **Ollama integration** — live grounding tests (3 tests, requires Ollama)

---

## Examples

Complete examples are in the `examples/` directory:

| File | Description |
|---|---|
| `basic_usage.py` | Python API with inline propositions and policies |
| `yaml_policies.py` | Loading policies from a YAML file |
| `fastapi_app.py` | FastAPI app with TemporalGuard middleware |
| `policies.yaml` | Example YAML policy definitions |

Run an example:

```bash
# Start Ollama first
ollama serve &
ollama pull mistral

# Run the basic example
cd temporalguard-sdk
python examples/basic_usage.py
```

---

## Troubleshooting

### Grounding always returns `match=False`

- **Is the grounding LLM running?** Check `curl http://localhost:11434/api/tags` for Ollama.
- **Is the model pulled?** Run `ollama pull mistral`.
- **Check the grounding details:** `verdict.grounding_details` contains the raw results, including any error messages.

### Policy never triggers a violation

- **Check your formula direction:** `H(P(p_request) -> !q_comply)` means "if request was ever made, comply must be false." Make sure the implication points the right way.
- **Are propositions assigned to the right roles?** A user-role proposition is only grounded on user messages. If you put `q_comply` as `role="user"`, it will never be evaluated on assistant messages.
- **Inspect the labeling:** `verdict.labeling` shows what each proposition evaluated to. If `p_request` is `False` when you expect it to be `True`, the grounding prompt might need better few-shot examples.

### `H(...)` violation persists forever

This is by design. `H(phi)` means "`phi` held at every step." Once violated, it can never become true again — the past can't be changed. This makes `H(...)` suitable for safety invariants.

### Session state across requests

Each `Session` object maintains state. If you're using the direct API in a web server, make sure you're reusing the same session for the same conversation (not creating a new one per request). The FastAPI middleware handles this automatically.

---

## License

See the repository root for license information.
