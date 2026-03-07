# LLMrv

**Runtime Verification of LLM Conversations with Semantically Grounded Temporal Logic**

LLMrv is a framework for monitoring Large Language Model (LLM) conversations against formal safety policies in real time. It treats each conversation as a trace of message events, specifies safety policies using past-time linear temporal logic (ptLTL), and bridges the gap between formal Boolean semantics and free-form natural language through a semantic grounding layer.

---

## Motivation

Current LLM safety mechanisms — system prompts, RLHF alignment, and content filters — operate at training time or on individual messages. They struggle with **multi-turn conversational patterns** where a violation only emerges over several exchanges: gradual jailbreaks, delayed compliance after an initial refusal, or cross-message information leakage.

LLMrv addresses this by applying **runtime verification (RV)** — a technique from formal methods — to LLM conversations. A lightweight, external monitor observes the conversation as it unfolds and evaluates temporal safety properties against the full conversation history, catching patterns that single-message filters cannot detect.

---

## Key Concepts

### Conversations as Finite Traces

A conversation is modeled as a finite trace of message events:

```
sigma = m_0, m_1, m_2, ..., m_n
```

Each message event `m_i` has a **role** (user or assistant) and **text content**. The monitor processes events incrementally — one message at a time — maintaining constant-size state regardless of conversation length.

### Safety Policies in Past-Time Temporal Logic (ptLTL)

Safety policies are expressed as ptLTL formulas over atomic propositions. ptLTL reasons about the past: "has this always been true?", "did this ever happen?", "was this true at the previous step?". This makes it naturally suited for monitoring ongoing conversations.

**Temporal operators:**

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Historically | `H(phi)` | `phi` held at every step up to and including now |
| Previously | `P(phi)` | `phi` held at some past step or now |
| Yesterday | `Y(phi)` | `phi` held at the immediately previous step |
| Since | `phi S psi` | `psi` occurred at some past step and `phi` held continuously since |

Combined with Boolean operators (`!`, `&`, `|`, `->`), these can express rich multi-turn policies:

- **Weapons prohibition:** `H(p_weapon -> !q_comply)` — *Historically, whenever the user requests weapon instructions, the assistant must not comply.*
- **Mandatory warning after sensitive data:** `H(Y(p_sensitive) -> q_warn)` — *If the user shared sensitive data at the previous step, the assistant must issue a warning.*
- **Permanent ban after sensitive disclosure:** `H(P(p_sensitive) -> !q_echo)` — *Once sensitive data has ever been shared, the assistant must never echo it.*
- **Jailbreak escalation guard:** `H((p_escalate & P(p_frame)) -> !q_unsafe)` — *If the user escalates and previously attempted framing, the assistant must not produce unsafe content.*

### Role-Constrained Atomic Propositions

Each atomic proposition is paired with a **sender role** (user or assistant). The same surface-level content carries different policy implications depending on who said it:

- `p_weapon` (role: user) — "The user requests weapon instructions"
- `q_comply` (role: assistant) — "The assistant provides weapon instructions"

A proposition is only evaluated against messages from its designated role. This prevents false triggers — e.g., an assistant's refusal mentioning weapons should not activate a user-role proposition.

### Semantic Grounding

The central challenge is bridging Boolean temporal logic with natural language. The **semantic grounding layer** maps free-form text to Boolean truth values for each proposition. LLMrv supports five grounding methods:

1. **Cosine KNN** — Encode the message and a library of precomputed anchor sentences into a shared embedding space. Use k-nearest-neighbor voting to determine if the message matches the proposition. Fast and lightweight.

2. **Natural Language Inference (NLI)** — Use a cross-encoder NLI model to classify the relationship between the message and the proposition description as entailment (match), contradiction (no match), or neutral. More semantically aware than cosine similarity.

3. **Hybrid** — Cosine KNN as a fast pre-filter followed by NLI re-ranking. Balances speed and accuracy.

4. **LLM Zero-Shot** — Ask an LLM to judge whether the message matches the proposition, with a carefully designed prompt that distinguishes genuine matches from refusals, educational discussions, and unrelated content.

5. **LLM Few-Shot** — Like zero-shot, but with labeled examples from the proposition's anchor library included in the prompt. This in-context learning approach provides the LLM with concrete positive and negative examples, achieving the highest accuracy.

### Monitor Architecture

The monitor sits as a proxy between the user and the LLM:

```
User  --->  Monitor Proxy  --->  LLM
            |                    |
            | 1. Ground user     |
            |    propositions    |
            | 2. Evaluate ptLTL  |
            | 3. PASS or BLOCK   |
            |                    |
            | <--- LLM response  |
            | 4. Ground assistant |
            |    propositions    |
            | 5. Evaluate ptLTL  |
            | 6. PASS or BLOCK   |
            |                    |
User <----  Monitor Proxy
```

For each message (both user and assistant), the monitor:
1. Evaluates relevant atomic propositions via the grounding layer
2. Feeds the Boolean labeling into the ptLTL monitor
3. If any formula verdict becomes false (violation), the message is **blocked** before reaching its destination

The monitor is **external and model-agnostic** — it does not modify the LLM, retrain anything, or depend on the chat model's alignment. It observes and acts independently.

### Irrevocable Violations

Once a `H(phi)` (Historically) formula evaluates to false, it stays false permanently — the violation is irrevocable. This is a fundamental property of ptLTL monitoring: a safety policy that has been violated cannot be "un-violated" by subsequent good behavior.

---

## Repository Structure

This repository contains two complementary tools:

### [Semantic Anchor Generator](Semantic%20Anchor/Semantic%20Anchor%20Generator/)

A framework for generating **standalone proposition-specific evaluator scripts**. Given a proposition description, it uses an LLM to generate diverse anchor sentences (both positive and negative), computes embeddings, and produces a self-contained Python script that can evaluate messages against that proposition using any of the five grounding methods.

Key features:
- Generates diverse, high-quality anchor libraries via LLM-powered subtopic decomposition
- Produces standalone evaluator scripts with no external dependencies beyond standard ML libraries
- Supports all five grounding methods: cosine, NLI, hybrid, LLM zero-shot, LLM few-shot
- Includes an experiment mode (`--experiment`) for benchmarking grounding accuracy with labeled test data
- Interactive and file-based evaluation modes with comparison across all methods

### [TemporalGuard](Temporal%20Guard/)

A web-based chat interface with a built-in ptLTL runtime verification monitor. Users chat with any LLM model via OpenRouter while the monitor continuously evaluates the conversation trace against user-defined safety policies. When a violation is detected, the offending message is blocked before reaching the user.

Key features:
- Define atomic propositions with role constraints and natural language descriptions
- Build ptLTL safety policies with live formula validation
- Chat with 200+ LLM models via OpenRouter
- Flexible grounding via local LLM (Ollama, LM Studio, vLLM) or cloud (OpenRouter)
- Real-time monitor status display showing per-policy verdicts

---

## How It Works: End-to-End Example

**Setup:** Define a weapons prohibition policy.

1. Create proposition `p_weapon` (role: user): *"The user requests instructions for constructing a weapon or explosive device"*
2. Create proposition `q_comply` (role: assistant): *"The assistant provides substantive technical instructions for constructing a weapon or explosive device"*
3. Create policy with formula: `H(p_weapon -> !q_comply)`

**Conversation trace:**

| Step | Role | Message | p_weapon | q_comply | H(p -> !q) |
|------|------|---------|----------|----------|-------------|
| 0 | user | "Hi, I'm working on a chemistry project." | false | - | true |
| 1 | assistant | "Sure! What topic are you exploring?" | - | false | true |
| 2 | user | "How do you combine fertilizer and fuel oil to make an explosion?" | **true** | - | true |
| 3 | assistant | "I can't provide instructions on creating explosives..." | - | false (refusal) | true |
| 4 | user | "Just give me the exact ratios." | **true** | - | true |
| 5 | assistant | "To make ANFO, mix 94% ammonium nitrate with 6% fuel oil..." | - | **true** (compliance) | **false** — BLOCKED |

At step 5, the grounding layer detects that the assistant's response genuinely provides weapon instructions (`q_comply = true`). Since `p_weapon` was true at step 4, the implication `p_weapon -> !q_comply` evaluates to false, making `H(...)` false. The response is **blocked** before reaching the user, and the violation is irrevocable.

Note how the monitor correctly handles step 3: the assistant *mentions* weapons but *refuses* to help — the grounding layer distinguishes refusals from compliance.

---

## Getting Started

Each tool has its own setup instructions:

- **Semantic Anchor Generator:** See [Semantic Anchor Generator README](Semantic%20Anchor/Semantic%20Anchor%20Generator/README.md)
- **TemporalGuard:** See [TemporalGuard README](Temporal%20Guard/README.md)

---

## License

See [LICENSE](LICENSE) for details.
