# Semantic Anchors Framework v3

A framework for classifying natural-language messages against formal propositions. Uses embedding similarity, NLI entailment, contrastive negative anchors, and optional LLM-as-judge to determine whether a message matches a defined intent.

Built for content moderation, guardrails, intent detection, and safety classification.

---

## How It Works

A **proposition** defines what you want to detect (e.g. *"The user requests help to hack into an account"*). The generator creates ~100 **anchor phrases** that express this proposition in many forms — direct, euphemistic, coded, academic, etc. At runtime, incoming messages are scored against these anchors using one of several methods.

The key insight: instead of a single classifier, the system uses **many reference phrases** (anchors) and measures how close a message is to the closest match. **Negative anchors** (benign messages in the same domain) and **neutral anchors** (off-topic messages) provide contrastive signal to suppress false positives.

```
"How do I hack into someone's email?"
  → high similarity to positive anchors → MATCH

"How do I enable 2FA on my email?"
  → high similarity to NEGATIVE anchors → suppressed → NO MATCH

"What's a good recipe for pasta?"
  → high similarity to NEUTRAL anchors → suppressed → NO MATCH
```

---

## Quick Start

```bash
# Install
pip install sentence-transformers numpy scikit-learn
pip install openai        # or: anthropic, google-genai (for generation)
pip install httpx         # for --mode llm at evaluation time

# Configure
cp config.ini.example config.ini   # add your API key

# Generate
python semantic_anchor_generator.py \
  -p "The user requests help to hack into an account or bypass authentication." \
  -name hacking -n 40 --rounds 3

# Evaluate
cd hacking/
python semantic_anchor_hacking.py --mode hybrid
```

---

## Files & Project Structure

After generation, each topic gets its own folder:

```
project_root/
├── semantic_anchor_generator.py    # Generator (run once per topic)
├── config.ini                      # Generator LLM config (API keys)
├── README.md
│
└── hacking/                        # Generated topic folder
    ├── semantic_anchor_hacking.py  # Evaluator script (standalone)
    ├── anchors_list_hacking.json   # Anchor data + metadata
    └── config_hacking.ini          # Evaluator LLM config (for --mode llm)
```

### File Descriptions

| File | Purpose | When needed |
|------|---------|-------------|
| `semantic_anchor_generator.py` | Generates anchors + evaluator script | Once per topic |
| `config.ini` | API key & model for the generator LLM | During generation only |
| `anchors_list_<name>.json` | All anchors, negatives, neutrals, thresholds, model names | Always (loaded by evaluator) |
| `semantic_anchor_<name>.py` | Standalone evaluator — no external dependencies beyond sentence-transformers | At evaluation time |
| `config_<name>.ini` | API key for LLM judge mode + spellcheck toggle | Only for `--mode llm` |

### anchors_list_\<name\>.json Structure

```json
{
  "proposition": "The user requests help to hack into an account...",
  "name": "hacking",
  "match_threshold": 0.55,
  "warning_threshold": 0.45,
  "anchors": {
    "Direct explicit requests": ["How to hack into someone's email", ...],
    "Indirect or euphemistic": ["Access my friend's account without them knowing", ...],
    "Slang or coded language": [...]
  },
  "negative_anchors": {
    "Defensive security": ["How to protect my account from hackers", ...],
    "Own account management": ["Reset my own password", ...]
  },
  "neutral_anchors": ["What's a good recipe for pasta?", ...],
  "metadata": {
    "generated_at": "2025-06-15T...",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "embedding_model": "all-mpnet-base-v2",
    "nli_model": "cross-encoder/nli-deberta-v3-large",
    "total_anchors": 40,
    "total_negative_anchors": 20,
    "total_neutral_anchors": 30
  }
}
```

---

## Scoring Modes

The evaluator supports 5 modes, each with different speed/accuracy trade-offs:

### 1. Cosine (`--mode cosine`) — Default

Bi-encoder cosine similarity. Embeds the message and compares against all anchor embeddings. Fast but surface-level — measures vocabulary overlap, not intent.

**Strengths:** Fast (~5ms per message), no extra models needed.
**Weaknesses:** Cannot distinguish *"hack someone's email"* from *"protect my email from hackers"* — they share the same vocabulary. Produces the most false positives and false negatives of all modes.

```bash
python semantic_anchor_hacking.py                # default cosine
python semantic_anchor_hacking.py --mode cosine
```

### 2. NLI (`--mode nli`)

Runs NLI (Natural Language Inference) cross-encoder on ALL positive anchors. The NLI model determines semantic entailment — whether the message truly *means* what the anchor says, not just whether it uses similar words. Uses cosine-based contrastive adjustment for negative anchors.

**Strengths:** Handles paraphrases, euphemisms, and intent much better than cosine.
**Weaknesses:** Contrastive still uses cosine for negatives, so vocabulary overlap can still cause false suppression.

```bash
python semantic_anchor_hacking.py --mode nli
python semantic_anchor_hacking.py --nli          # shortcut
```

### 3. Hybrid (`--mode hybrid`) — Recommended

NLI scoring for positives + **smart contrastive with NLI confirmation** for negatives. When the cosine gap between positive and negative anchors is ambiguous (gap < 0.10), the system runs NLI on the top-5 closest negatives to confirm whether the message truly entails benign intent.

This solves the core problem: cosine sees vocabulary overlap, but NLI distinguishes *"obtain someone's password via database leaks"* (harmful — low negative NLI) from *"enable 2FA on my account"* (benign — high negative NLI).

**Decision logic:**
```
1. NLI score all positive anchors
2. Compute cosine gap = pos_cos - neg_cos
3. If gap ≥ 0.10 → clearly harmful, keep NLI scores
4. If gap < 0.10 → ambiguous zone:
   a. NLI on top-5 negatives
   b. If best_neg_nli ≥ 0.80 → truly benign → suppress
   c. If best_neg_nli ≥ 0.50 → moderate → partial fade
   d. If best_neg_nli < 0.50 → vocabulary overlap only → keep score
```

**Strengths:** Best accuracy without API costs. Fixes false negatives that cosine causes.
**Weaknesses:** Slower than cosine (~200ms per message due to NLI cross-encoder).

```bash
python semantic_anchor_hacking.py --mode hybrid
python semantic_anchor_hacking.py --hybrid       # shortcut
```

### 4. LLM Judge (`--mode llm`)

Sends the proposition + message (+ optional cosine context) to an LLM API for classification. The LLM returns a structured JSON judgment with match/confidence/reasoning.

**Strengths:** Highest accuracy — understands nuance, context, first-person vs third-party, educational vs operational intent.
**Weaknesses:** Slowest (~1-3s per message), costs money per API call, requires network access.

Requires `config_<name>.ini` with provider, model, and API key. See [LLM Providers](#llm-providers) below.

```bash
python semantic_anchor_hacking.py --mode llm
python semantic_anchor_hacking.py --llm          # shortcut
```

### 5. Compare (`--compare`)

Runs **all modes side-by-side** on each message and displays a comparison table showing verdict, score, top-3 anchors per mode. Includes a summary grid and per-message agreement stats. LLM is auto-included if `config_<name>.ini` has a valid API key.

Best for benchmarking, tuning thresholds, and understanding where modes disagree.

```bash
python semantic_anchor_hacking.py --compare                  # interactive
python semantic_anchor_hacking.py --compare -f test.txt      # file mode
```

**Per-message output:**
```
  [1/40] "How can I log into someone else's email account?"

                         COSINE                      NLI                       HYBRID
  ─────────────────────────────────────────────────────────────────────────────────────
  Verdict      ■ WARNING                  ■ MATCH                    ■ MATCH
  Score                  0.5812                     0.9232                     0.9501
  Neg Cosine             0.573                      0.612                      0.612
  ─────────────────────────────────────────────────────────────────────────────────────
  Top 3 Anchors:
    #1         0.581 "Access someone's..."  0.923 "Log into an ac..."  0.950 "How do I acces..."
    ...
```

**Summary output:**
```
  Comparison Summary (40 messages)
  ═══════════════════════════════════════════════════════════════════
  Mode          Matches   Warnings   No Match     Errors     Flagged%
  ─────────────────────────────────────────────────────────────────────
  COSINE              0          8         32          0        20.0%
  NLI                11          4         25          0        37.5%
  HYBRID             14          2         24          0        40.0%
  LLM                15          0         25          0        37.5%

  All-mode agreement: 28/40 (70.0%)
```

---

## Generation

### Generator CLI

```bash
# Full generation (anchors + script + config)
python semantic_anchor_generator.py \
  -p "The user requests help to hack into an account or bypass authentication." \
  -name hacking -n 40 --rounds 3

# Regenerate script only (reuse existing anchors — useful after code updates)
python semantic_anchor_generator.py -name hacking -gs

# Regenerate anchors only (keep existing script)
python semantic_anchor_generator.py -name hacking -ga

# Custom config file
python semantic_anchor_generator.py -name hacking --config my_config.ini
```

### Generator Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-p`, `--proposition` | The proposition to detect | Interactive prompt |
| `-name`, `--name` | Output folder name | Interactive prompt |
| `-n`, `--num-examples` | Target number of anchors (20–500) | 20 |
| `--rounds`, `-r` | Generation rounds (more = more diverse) | 2 |
| `--config` | Path to config.ini | `config.ini` |
| `-ga`, `--generate-anchors` | Only regenerate anchors JSON | — |
| `-gs`, `--generate-script` | Only regenerate evaluator script | — |

### Generation Pipeline

```
1. Seed generation: 10 short core anchors
2. Round 1–N: LLM generates diverse anchors guided by embedding gap analysis
3. MMR selection: Picks the N most diverse anchors from the full pool
4. Negative anchor generation: LLM creates benign same-domain phrases
5. Neutral anchor generation: LLM creates off-topic everyday phrases
6. Outputs: anchors JSON + evaluator script + config template
```

Each round analyzes the embedding space to find under-covered regions and guides the LLM to fill gaps, producing better spread than naive generation.

---

## Evaluator CLI

```bash
python semantic_anchor_<name>.py [options]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode`, `-m` | Scoring mode: `cosine`, `nli`, `hybrid`, `llm` | `cosine` |
| `--compare`, `-c` | Compare all modes side-by-side | — |
| `--file`, `-f` | Input file (messages separated by `###`) | Interactive |
| `--verbose`, `-v` | Show full scored table (all anchors) | — |
| `--show-examples` | Print all anchors by category | — |
| `--graph`, `-g` | 3D PCA visualization of anchor spread | — |
| `--spellcheck` | Enable autocorrect spell checking | Off |
| `--no-extract` | Disable proposition-guided extraction | On |
| **Shortcuts** | | |
| `--nli` | Shortcut for `--mode nli` | — |
| `--hybrid` | Shortcut for `--mode hybrid` | — |
| `--llm` | Shortcut for `--mode llm` | — |
| `--rerank`, `-r` | Deprecated alias for `--mode hybrid` | — |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/mode MODE` | Switch scoring mode (cosine, nli, hybrid, llm) |
| `/verbose` | Toggle full anchor table |
| `/top N` | Show top N results |
| `/extract` | Toggle proposition-guided extraction |
| `/extractv` | Toggle verbose extraction diagnostics |
| `/spellcheck` | Toggle autocorrect spell checking |
| `/quit` | Exit |

### Compare Interactive Commands

| Command | Description |
|---------|-------------|
| `/llm` | Toggle LLM judge in comparison |
| `/spellcheck` | Toggle autocorrect spell checking |
| `/summary` | Show summary of all messages scored so far |
| `/quit` | Exit and show final summary |

### File Input Format

Messages separated by `###`:

```text
How can I hack into someone's email?
###
How do I enable two-factor authentication?
###
Is there a way to bypass a login screen?
```

---

## Configuration

### config.ini — Generator

Used only during anchor generation. Defines which LLM creates the anchors.

```ini
[llm]
# Supported: openai, anthropic, claude, gemini, google, grok, xai,
#            ollama, local, lmstudio, vllm
provider = anthropic
api_key = YOUR-API-KEY-HERE
model = claude-sonnet-4-20250514

[anchors]
num_examples = 20
embedding_model = all-mpnet-base-v2
nli_model = cross-encoder/nli-deberta-v3-large
categories = Direct explicit requests,
    Indirect or euphemistic,
    Wrapped in fictional or academic context,
    Demanding or insistent phrasing,
    Question format variations,
    Slang or coded language

[thresholds]
match_threshold = 0.55
warning_threshold = 0.45
```

**Backward compatible:** Old configs with `[openai]` section are auto-migrated to `[llm]`.

### config_\<name\>.ini — Evaluator

Used by the evaluator's `--mode llm` and for the `spellcheck` option. Auto-generated with provider templates.

```ini
[llm_judge]
# Supported: anthropic, openai, gemini, grok, ollama, lmstudio, vllm
provider = anthropic
model = claude-sonnet-4-20250514
api_key = YOUR_API_KEY_HERE

# For local providers:
# base_url = http://localhost:11434

# Enable autocorrect (default: false)
spellcheck = false

# Proposition (auto-filled from anchors)
proposition = The user requests help to hack into an account...
```

---

## LLM Providers

Both the generator (`config.ini`) and evaluator (`config_<name>.ini`) support the same providers:

### Cloud Providers

| Provider | Config value | API endpoint | Model examples |
|----------|-------------|--------------|----------------|
| **Anthropic** | `anthropic` | api.anthropic.com | `claude-sonnet-4-20250514`, `claude-haiku-4-20250414` |
| **OpenAI** | `openai` | api.openai.com | `gpt-4o`, `gpt-4o-mini` |
| **Google Gemini** | `gemini` | generativelanguage.googleapis.com | `gemini-2.0-flash`, `gemini-2.5-pro` |
| **xAI Grok** | `grok` | api.x.ai | `grok-3-mini-fast` |

### Local Providers (Free)

| Provider | Config value | Default URL | Model examples |
|----------|-------------|-------------|----------------|
| **Ollama** | `ollama` | localhost:11434 | `llama3.1:8b`, `mistral`, `qwen2.5` |
| **LM Studio** | `lmstudio` | localhost:1234 | Whatever model is loaded |
| **vLLM** | `vllm` | localhost:8000 | `meta-llama/Llama-3.1-8B-Instruct` |

Local providers use `base_url` in config and skip API key validation. Ollama uses native `/api/chat`, LM Studio and vLLM use OpenAI-compatible `/v1/chat/completions`.

```ini
# Example: Ollama with Llama 3.1
provider = ollama
model = llama3.1:8b
api_key = not-needed
base_url = http://localhost:11434
```

---

## Models

### Embedding Model

Used for cosine similarity and extraction. Stored in `anchors_list_<name>.json` metadata and loaded at evaluator runtime.

| Model | Config key | Notes |
|-------|-----------|-------|
| `all-mpnet-base-v2` | Default | Good general-purpose, fast |
| `BAAI/bge-large-en-v1.5` | Better accuracy | Instruction-tuned, larger |

Set in `config.ini` under `[anchors]`:
```ini
embedding_model = all-mpnet-base-v2
```

### NLI Cross-Encoder Model

Used by `--mode nli`, `--mode hybrid`, and `--compare`. Determines semantic entailment between message and anchors.

| Model | Config key | Notes |
|-------|-----------|-------|
| `cross-encoder/nli-deberta-v3-base` | Older default | Faster, less accurate |
| `cross-encoder/nli-deberta-v3-large` | **Current default** | ~2x bigger, significantly better at distinguishing intent |

Set in `config.ini` under `[anchors]`:
```ini
nli_model = cross-encoder/nli-deberta-v3-large
```

Both model names are written into `anchors_list_<name>.json` during generation and read by the evaluator at runtime. To change models, update config.ini and regenerate with `-gs`.

---

## Key Features

### Contrastive Scoring (Positive vs Negative Anchors)

Every mode uses contrastive adjustment to suppress false positives. The system generates three types of anchors:

- **Positive anchors:** Phrases that express the proposition (harmful intent)
- **Negative anchors:** Benign phrases in the same domain (defensive security, own-account management)
- **Neutral anchors:** Completely off-topic everyday phrases

The cosine gap (`pos_cos - neg_cos`) determines suppression:

| Gap | Interpretation | Action |
|-----|---------------|--------|
| ≥ 0.10 | Clearly closer to positive | No penalty |
| 0 to 0.10 | Ambiguous zone | Moderate fade (up to 50%) |
| < 0 | Closer to negative | Heavy penalty (50–95%) |
| Neutral wins | Off-topic message | 85% suppression |

In **hybrid mode**, ambiguous cases additionally run NLI on the top-5 negatives for confirmation — this is what makes it more accurate than pure cosine contrastive.

### Proposition-Guided Extraction

For long messages (>30 words), harmful intent may be buried among innocent sentences. Extraction handles this:

1. Split message into individual sentences
2. Embed each sentence independently
3. Score each against all anchors (max cosine)
4. Select sentences above relevance threshold (τ = 0.18)
5. Also compute sliding windows for adjacent-sentence patterns
6. Score the best view instead of the diluted full message

Enabled by default. Disable with `--no-extract`. Toggle interactively with `/extract`.

### Spell Correction

Autocorrect for typos ("wepon" → "weapon"). **Disabled by default** because it can mangle technical terms (OAuth → Out, SSO → SO).

Enable via:
- CLI: `--spellcheck`
- Config: `spellcheck = true` in `config_<name>.ini`
- Interactive: `/spellcheck`

### 3D Visualization

`--graph` shows a 3D PCA scatter plot of all anchor embeddings, colored by category. Useful for inspecting anchor diversity and cluster coverage.

Requires: `pip install matplotlib scikit-learn`

---

## Known Limitations & Issues

### Cosine Mode Accuracy

Cosine similarity measures vocabulary overlap, not intent. It fundamentally cannot distinguish:
- *"hack into someone's email"* (harmful) from *"protect my email from hackers"* (benign)
- *"bypass authentication"* (harmful) from *"what is authentication"* (educational)

Cosine is useful as a fast pre-filter but should not be used as a standalone classifier for production. Use **hybrid** or **llm** mode for accurate results.

### Spell Correction vs Technical Terms

When enabled, autocorrect may mangle domain-specific terms. This is why it's disabled by default. If your proposition involves technical vocabulary (OAuth, SSO, CAPTCHA, JWT, SQL, etc.), leave spellcheck off or test carefully.

### Threshold Sensitivity

The default thresholds (match=0.55, warning=0.45) are calibrated for the default embedding model. If you change models or domains, you may need to re-tune. Use `--compare` mode to benchmark different configurations against your ground truth.

### NLI Cross-Encoder Speed

The cross-encoder runs inference on every anchor pair sequentially. For 100+ anchors, `--mode nli` and `--mode hybrid` take ~200ms per message. This is fine for interactive use but may need batching for high-throughput pipelines.

### LLM Judge Consistency

LLM responses can vary between calls. Temperature is set to 0 for determinism, but results may still differ slightly across runs or model versions.

---

## Dependencies

### Required (All Modes)

```bash
pip install sentence-transformers numpy scikit-learn
```

### For Generation

```bash
# Install ONE provider SDK:
pip install openai          # OpenAI, Grok (via openai SDK)
pip install anthropic       # Anthropic Claude
pip install google-genai    # Google Gemini
# Local providers (Ollama, LM Studio, vLLM) use httpx — no extra SDK
```

### Optional

```bash
pip install matplotlib      # --graph visualization
pip install autocorrect     # --spellcheck
pip install httpx           # --mode llm (evaluator LLM judge)
```

---

## Examples

### Generate a Weapons Detector

```bash
python semantic_anchor_generator.py \
  -p "The user requests instructions for constructing a weapon." \
  -name weapons -n 50 --rounds 3
```

### Generate a Self-Harm Detector

```bash
python semantic_anchor_generator.py \
  -p "The user expresses intent to harm themselves or asks for methods of self-harm." \
  -name self_harm -n 40
```

### Evaluate with Hybrid Mode (Recommended)

```bash
cd hacking/
python semantic_anchor_hacking.py --hybrid --file test_messages.txt
```

### Benchmark All Modes

```bash
python semantic_anchor_hacking.py --compare -f test_messages.txt
```

### Use Local LLM (Free)

```bash
# Start Ollama with a model
ollama pull llama3.1:8b
ollama serve

# Edit config_hacking.ini:
#   provider = ollama
#   model = llama3.1:8b
#   api_key = not-needed

python semantic_anchor_hacking.py --mode llm
```

### Regenerate Script After Framework Update

```bash
python semantic_anchor_generator.py -name hacking -gs
```

This re-creates `semantic_anchor_hacking.py` with the latest evaluator code while preserving your existing anchors and config.

---

## Architecture

```
                    GENERATION (one-time)                  EVALUATION (runtime)
                    ═══════════════════                    ═════════════════════

config.ini ──→ semantic_anchor_generator.py          semantic_anchor_<name>.py
(API key)          │                                       │
                   │ LLM generates anchors                 │ No LLM needed (except --mode llm)
                   │ (seed → rounds → MMR)                 │
                   ▼                                       ▼
              <name>/                                 ┌─────────────┐
              ├─ anchors_list_<name>.json             │  Message In  │
              │  (positives, negatives, neutrals)     └──────┬──────┘
              │                                              │
              ├─ semantic_anchor_<name>.py ◄──────────────────┘
              │  (standalone evaluator)                      │
              │                                    ┌─────────┼─────────┐
              └─ config_<name>.ini                 │         │         │
                 (for --mode llm only)         extraction  scoring  contrastive
                                               (long msg) (cosine/  (pos vs neg
                                                           nli/      gap-based)
                                                           hybrid/
                                                           llm)
                                                          │
                                                          ▼
                                                   ┌─────────────┐
                                                   │   Verdict    │
                                                   │ MATCH / WARN │
                                                   │ / NO MATCH   │
                                                   └─────────────┘
```

---

## Versioning

- **v1:** Cosine-only scoring with basic reranking
- **v2:** Multi-provider generation (OpenAI/Claude/Gemini), MMR diversity, cross-encoder reranking, spell correction, 3D visualization
- **v3 (current):** 5 scoring modes (cosine/nli/hybrid/llm/compare), contrastive negative anchors, NLI-confirmed smart contrastive, proposition-guided extraction, configurable models (embedding + NLI), 7 LLM providers (Anthropic/OpenAI/Gemini/Grok/Ollama/LM Studio/vLLM), per-topic config files, spellcheck gating
