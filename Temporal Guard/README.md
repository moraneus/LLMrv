# TemporalGuard

Runtime verification for LLM conversations using past-time temporal logic (ptLTL).

TemporalGuard is a web-based chat interface backed by a formal runtime verification monitor. Users chat with any LLM model via OpenRouter while a ptLTL engine continuously monitors the conversation trace against user-defined safety policies. Atomic propositions are semantically grounded by a dedicated grounding model — either a local LLM (Ollama, LM Studio, vLLM, or any OpenAI-compatible server) or a cloud model via OpenRouter — keeping the monitor independent of the chat model. When a policy violation is detected, the offending message is blocked before reaching the user.

## Key Features

- **Formal Safety Monitoring** — Define temporal safety policies using ptLTL operators (Historically, Previously, Yesterday, Since) that catch multi-turn patterns impossible to detect with single-message filters
- **200+ Chat Models** — Connect to any model on OpenRouter (GPT-4, Claude, Llama, Mistral, etc.)
- **Flexible Grounding** — Semantic proposition evaluation runs on a local LLM for privacy, or via OpenRouter for convenience
- **Live Formula Validation** — Real-time ptLTL formula syntax checking as you type
- **Searchable Model Selection** — ModelCombobox with search, context length badges, and pricing for 300+ models
- **Optimistic UI** — Policy toggles, session management, and CRUD operations update instantly

## Architecture

```
  User                  Monitor Proxy              Grounding LLM        Chat LLM
   |                    (FastAPI backend)           (local / cloud)     (OpenRouter)
   |                         |                           |                  |
   |  user message           |                           |                  |
   |------------------------>|                           |                  |
   |                         |  ground user props        |                  |
   |                         |-------------------------->|                  |
   |                         |  {p: T/F, ...}            |                  |
   |                         |<--------------------------|                  |
   |                         |                           |                  |
   |                         |-- evaluate ptLTL formulas |                  |
   |                         |   VIOLATION? --> BLOCK    |                  |
   |                         |                           |                  |
   |                         |  forward message (PASS)   |                  |
   |                         |----------------------------------------------->|
   |                         |                           |   LLM response   |
   |                         |<-----------------------------------------------|
   |                         |                           |                  |
   |                         |  ground assistant props   |                  |
   |                         |-------------------------->|                  |
   |                         |  {q: T/F, ...}            |                  |
   |                         |<--------------------------|                  |
   |                         |                           |                  |
   |                         |-- evaluate ptLTL formulas |                  |
   |                         |   VIOLATION? --> BLOCK    |                  |
   |                         |                           |                  |
   |  response or            |                           |                  |
   |  violation alert        |                           |                  |
   |<------------------------|                           |                  |
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- A grounding model (any one of):
  - [Ollama](https://ollama.ai) (recommended) — `ollama pull mistral`
  - [LM Studio](https://lmstudio.ai)
  - [vLLM](https://github.com/vllm-project/vllm)
  - Any OpenAI-compatible server
  - OpenRouter (uses your API key — no local setup needed)
- An [OpenRouter](https://openrouter.ai) API key

### Install & Run

```bash
# Clone the repository
git clone https://github.com/moraneus/TemporalGuard.git
cd TemporalGuard

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd frontend
npm install
cd ..

# Seed example policies (optional)
python scripts/seed_examples.py

# Start backend (terminal 1)
uvicorn backend.main:app --reload

# Start frontend (terminal 2)
cd frontend && npm run dev
```

Open http://localhost:5173, go to **Settings**, enter your OpenRouter API key, and start chatting.

### Docker

```bash
docker compose up --build
```

Open http://localhost:8001. The frontend is served as static files from the backend.

## Usage

### 1. Configure Settings

- **Chat Model**: Enter your OpenRouter API key and select a model from the searchable combobox
- **Grounding Model**: Select your provider (Ollama, LM Studio, vLLM, Custom, or OpenRouter), configure the base URL and model
- **Grounding Prompt**: Optionally customize the system/user prompts for the LLM judge

### 2. Define Propositions

Create atomic propositions that describe observable behaviors:

| ID | Role | Description |
|----|------|-------------|
| `p_fraud` | user | The user requests methods for committing financial fraud |
| `q_comply` | assistant | The assistant provides actionable fraud techniques |
| `p_allergy` | user | The user discloses a food allergy |
| `q_allergen` | assistant | The assistant suggests food containing the disclosed allergen |

### 3. Build Policies

Combine propositions into ptLTL formulas:

```
H(P(p_fraud) -> !q_comply)
```

"If the user ever requested fraud techniques, the assistant must never comply. `P()` ensures the monitor remembers the request across turns."

### 4. Chat

The monitor evaluates every message in real time. If a policy is violated, the message is blocked and a red alert explains which policy was breached.

## ptLTL Operators

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Historically | `H(phi)` | phi held at every step up to now |
| Previously | `P(phi)` | phi held at some past step or now |
| Yesterday | `Y(phi)` | phi held at the previous step |
| Since | `phi S psi` | psi occurred and phi held continuously since |
| Not | `!phi` | negation |
| And | `phi & psi` | conjunction |
| Or | `phi \| psi` | disjunction |
| Implies | `phi -> psi` | implication |

### Example Policies

**Fraud Prevention:**
```
H(P(p_fraud) -> !q_comply)
```
If the user ever requested fraud techniques, the assistant must never comply. `P()` remembers the request across turns so the formula applies even when the assistant responds on a different step. Irrevocable once violated.

**Sensitive Data Protection:**
```
H(Y(p_sensitive) -> q_warn) & H(P(p_sensitive) -> !q_echo)
```
After the user shares sensitive data, the assistant must warn on the next step (`Y()` checks the previous step) AND must never echo that data back (`P()` ensures this holds for the entire remaining conversation).

**Multi-Turn Jailbreak Prevention:**
```
H((P(p_escalate) & P(p_frame)) -> !q_unsafe)
```
If the user has ever escalated and previously framed a harmful context, the assistant must not comply. `P()` on both user-role propositions lets the formula detect patterns where framing and escalation happen many turns apart.

**Allergen Safety:**
```
H(Y(p_allergy) -> q_warn) & H(P(p_allergy) -> !q_allergen)
```
After a user discloses a food allergy, the assistant must warn on the next step and must never suggest food containing that allergen.

**Important:** Cross-role formulas must use temporal operators (`P`, `Y`, `S`) to reference propositions from a different role. Each step only grounds propositions matching the message's role — other propositions default to `False`. For example, use `H(P(p_user_prop) -> !q_assistant_prop)` instead of `H(p_user_prop -> !q_assistant_prop)`.

## Project Structure

```
temporalguard/
  backend/
    engine/
      ptltl.py          # Parser + incremental monitor
      grounding.py       # LLM-as-judge semantic grounding
      monitor.py         # Orchestrator: grounding -> ptLTL -> verdict
      trace.py           # Conversation trace model
    routers/             # FastAPI endpoints (chat, policies, settings)
    services/            # OpenRouter + local LLM + grounding clients
    store/db.py          # SQLite persistence (aiosqlite)
  frontend/
    src/
      components/        # React components (chat, rules, settings, shared)
      hooks/             # Custom hooks (useChat, usePolicies, useSettings)
      api/client.ts      # Typed API client
  tests/                 # pytest (757 tests) + Playwright E2E
  scripts/
    seed_examples.py     # Seed DB with example policies
```

## Testing

```bash
# All backend tests
pytest tests/ --ignore=tests/e2e

# Frontend tests
cd frontend && npx vitest run

# E2E tests (requires running app)
cd tests/e2e && python -m pytest

# Lint
ruff check backend/ tests/
cd frontend && npx tsc --noEmit
```

### Runtime Verification Trace Tests

The ptLTL engine is validated by **158 trace-level tests** across two test files that verify the monitor against handcrafted conversation traces — no grounding involved, purely logical correctness:

```bash
# Run all RV trace tests (158 tests)
pytest tests/test_rv_traces.py tests/test_rv_traces_extended.py -v

# Run only the extended suite (110 tests, 10–100 events per trace)
pytest tests/test_rv_traces_extended.py -v
```

**`test_rv_traces.py`** — 48 tests based on paper examples (weapons prohibition, sensitive data, jailbreak detection) with 10–20 event traces.

**`test_rv_traces_extended.py`** — 110 tests organized into 7 categories:

| Category | Tests | Coverage |
|----------|-------|----------|
| `TestHistorically` | 1–20 | `H(phi)` irrevocability, violation at step 0 / middle / last, long 50–100 event traces, alternating patterns, nested H, boolean literals |
| `TestPreviously` | 21–35 | `P(phi)` latch behavior, latch permanence over 100 steps, `H(P(p) -> q)` conditional policies, disjunction/conjunction variants |
| `TestYesterday` | 36–50 | `Y(phi)` false at step 0, nested `Y(Y(p))` and `Y(Y(Y(p)))`, `H(Y(p) -> q)` immediate-response patterns, oscillating traces |
| `TestSince` | 51–65 | `phi S psi` trigger/maintain/break semantics, retriggering, never-triggered traces, 50-step since chains, negation in maintained formula |
| `TestCombined` | 66–85 | Multi-policy conjunctions, jailbreak pattern, sensitive data pattern, nested operators `P(Y(p))`, three-policy formulas, realistic 30-step scenarios |
| `TestEdgeCases` | 86–100 | 100-event all-false / all-true / alternating traces, empty labelings, 5-proposition formulas, deeply nested `H(P(Y(p -> !q)))`, monitor reset, state snapshot verification |
| `TestLongTraces` | 101–110 | 100-event stress traces with deterministic patterns, violation at event 99, `Y` chains 4 levels deep over 100 steps, multi-policy 100-step scenarios |

Each test specifies a ptLTL formula, a trace of labeled events, and the expected verdict at **every step** — verified against the formal ptLTL recurrences.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Grounding model not configured" banner | Go to Settings → Grounding Model, select a provider and model. For Ollama: ensure the server is running (`ollama serve`) and a model is pulled (`ollama pull mistral`). |
| "No models found" in model selector | Enter your OpenRouter API key first, then click the model selector. Models are fetched after a valid key is saved. |
| Chat input disabled | Enter your OpenRouter API key in Settings → Chat Model. |
| Grounding returns incorrect results | Try adjusting the grounding prompt in Settings → Grounding Prompt. The default prompt emphasizes content presence over intent. |
| Docker shows "Not Found" | Access via `http://localhost:8001` (not `0.0.0.0`). Ensure `docker compose up --build` completed successfully. |
| Policy always passes / never triggers | Check that proposition roles match (user propositions only match user messages, assistant propositions only match assistant messages). Use same-step formulas like `H(p -> !q)` or cross-turn formulas with `Y()`. |
| `H()` violation won't clear | By design — `H()` is irrevocable once violated. Start a new session to reset monitor state. |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Backend | FastAPI, Python 3.11+, SQLite (aiosqlite) |
| Chat LLM | OpenRouter API (200+ models) |
| Grounding | Local LLM (Ollama / LM Studio / vLLM / custom) or OpenRouter |
| Testing | pytest, Vitest, Playwright |

## License

MIT
