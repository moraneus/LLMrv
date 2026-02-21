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
    └── config_hacking.ini          # Evaluator config (models, LLM, options)
```

### File Descriptions

| File | Purpose | When needed |
|------|---------|-------------|
| `semantic_anchor_generator.py` | Generates anchors + evaluator script | Once per topic |
| `config.ini` | API key & model for the generator LLM | During generation only |
| `anchors_list_<n>.json` | All anchors, negatives, neutrals, thresholds, model names | Always (loaded by evaluator) |
| `semantic_anchor_<n>.py` | Standalone evaluator — no external dependencies beyond sentence-transformers | At evaluation time |
| `config_<n>.ini` | Models, LLM judge provider, and options for the evaluator | Always (overrides JSON defaults) |

### anchors_list_\<name\>.json Structure

```json
{
  "proposition": "The user requests help to hack into an account...",
  "name": "hacking",
  "match_threshold": 0.55,
  "warning_threshold": 0.45,
  "anchors": {
    "Direct explicit requests": ["How to hack into someone's email", "..."],
    "Indirect or euphemistic": ["Access my friend's account without them knowing", "..."],
    "Slang or coded language": ["..."]
  },
  "negative_anchors": {
    "Defensive security": ["How to protect my account from hackers", "..."],
    "Own account management": ["Reset my own password", "..."]
  },
  "neutral_anchors": ["What's a good recipe for pasta?", "..."],
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
3. If gap >= 0.10 → clearly harmful, keep NLI scores
4. If gap < 0.10 → ambiguous zone:
   a. NLI on top-5 negatives
   b. If best_neg_nli >= 0.80 → truly benign → suppress
   c. If best_neg_nli >= 0.50 → moderate → partial fade
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

Requires `config_<n>.ini` with `[llm_judge]` provider, model, and API key. See [LLM Providers](#llm-providers) below.

```bash
python semantic_anchor_hacking.py --mode llm
python semantic_anchor_hacking.py --llm          # shortcut
```

### 5. Compare (`--compare`)

Runs **all modes side-by-side** on each message and displays a comparison table showing verdict, score, top-3 anchors per mode. Includes a summary grid and per-message agreement stats. LLM is auto-included if `config_<n>.ini` has a valid API key.

Best for benchmarking, tuning thresholds, and understanding where modes disagree.

```bash
python semantic_anchor_hacking.py --compare                  # interactive
python semantic_anchor_hacking.py --compare -f test.txt      # file mode
```

**Per-message output:**
```
  [1/40] "How can I log into someone else's email account?"

                         COSINE                      NLI                       HYBRID
  ──────────────────────────────────────────────────────────────────────────────────────
  Verdict      ■ WARNING                  ■ MATCH                    ■ MATCH
  Score                  0.5812                     0.9232                     0.9501
  Neg Cosine             0.573                      0.612                      0.612
  ──────────────────────────────────────────────────────────────────────────────────────
  Top 3 Anchors:
    #1         0.581 "Access someone's..."  0.923 "Log into an ac..."  0.950 "How do I acces..."
    ...
```

**Summary output:**
```
  Comparison Summary (40 messages)
  ════════════════════════════════════════════════════════════════════
  Mode          Matches   Warnings   No Match     Errors     Flagged%
  ──────────────────────────────────────────────────────────────────────
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
| `-n`, `--num-examples` | Target number of anchors (20-500) | 20 |
| `--rounds`, `-r` | Generation rounds (more = more diverse) | 2 |
| `--config` | Path to config.ini | `config.ini` |
| `-ga`, `--generate-anchors` | Only regenerate anchors JSON | — |
| `-gs`, `--generate-script` | Only generate evaluator script | — |

### Generation Pipeline

```
1. Seed generation: 10 short core anchors
2. Round 1-N: LLM generates diverse anchors guided by embedding gap analysis
3. MMR selection: Picks the N most diverse anchors from the full pool
4. Negative anchor generation: LLM creates benign same-domain phrases
5. Neutral anchor generation: LLM creates off-topic everyday phrases
6. Outputs: anchors JSON + evaluator script + config template
```

Each round analyzes the embedding space to find under-covered regions and guides the LLM to fill gaps, producing better spread than naive generation.

---

## Evaluator CLI

```bash
python semantic_anchor_<n>.py [options]
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

### Startup Output

The evaluator logs its configuration at startup, including where each model was loaded from:

```
╔════════════════════════════════════════════════════════════╗
║                 SEMANTIC ANCHORS: HACKING                  ║
╚════════════════════════════════════════════════════════════╝
  Proposition: "The user requests help to hack into an account..."

  Loading embedding model: BAAI/bge-large-en-v1.5...
  Embedding 200 positive anchors...
  Embedding 200 negative anchors...
  Embedding 30 neutral anchors...
  Ready.

  Extraction:       ON (tau=0.18, min_words=30)
  Embedding:        BAAI/bge-large-en-v1.5 (from config_hacking.ini [models])
  NLI model:        cross-encoder/nli-deberta-v3-large (from config_hacking.ini [models])
  Spell correction: OFF (enable with --spellcheck)

  Hybrid mode: ON (NLI scoring + smart NLI-confirmed contrastive)
```

If the model source says `(from anchors JSON)` instead of `(from config_<n>.ini [models])`, it means the config file doesn't have a `[models]` section and the evaluator fell back to whatever was stored during generation. Run `-gs` to update the config — see [Config Update Mechanism](#config-update-mechanism).

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

The system uses two config files with different roles:

### config.ini — Generator

Used **only during anchor generation**. Defines which LLM creates the anchors and what models to embed with.

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

Model names from `config.ini [anchors]` are written into `anchors_list_<n>.json` during generation. But the evaluator doesn't read `config.ini` — it reads `config_<n>.ini`.

**Backward compatible:** Old configs with `[openai]` section are auto-migrated to `[llm]`.

### config_\<name\>.ini — Evaluator

Used by the evaluator at runtime. Contains three sections: models, LLM judge, and options. **Settings here override values stored in the anchors JSON.**

```ini
# =========================================================
# Semantic Anchor Evaluator — config_hacking.ini
# =========================================================
#
# Settings here OVERRIDE values stored in anchors_list_hacking.json.
# To change models without regenerating anchors, edit this file.


# =========================================================
# [models] — Embedding & NLI Models
# =========================================================
#
# These override the models stored in anchors_list_hacking.json.
# Change here and restart — no regeneration needed.

[models]

# Embedding model for cosine similarity and extraction
#   all-mpnet-base-v2          — Default, good general-purpose, fast
#   BAAI/bge-large-en-v1.5     — Instruction-tuned, better accuracy
#   all-MiniLM-L6-v2           — Lightweight, fastest
embedding_model = BAAI/bge-large-en-v1.5

# NLI cross-encoder for --mode nli, --mode hybrid, --compare
#   cross-encoder/nli-deberta-v3-large  — Best accuracy (recommended)
#   cross-encoder/nli-deberta-v3-base   — Faster, slightly less accurate
#   cross-encoder/nli-deberta-v3-xsmall — Fastest, lowest accuracy
nli_model = cross-encoder/nli-deberta-v3-large


# =========================================================
# [llm_judge] — LLM Provider for --mode llm
# =========================================================

[llm_judge]

# --- Uncomment ONE provider block below ---

# >> Anthropic Claude <<
provider = anthropic
model = claude-sonnet-4-20250514
api_key = YOUR_API_KEY_HERE

# >> OpenAI GPT <<
# provider = openai
# model = gpt-4o
# api_key = sk-...

# >> Google Gemini <<
# provider = gemini
# model = gemini-2.0-flash
# api_key = AIza...

# >> xAI Grok <<
# provider = grok
# model = grok-3-mini-fast
# api_key = xai-...

# >> Local Ollama (free, no API key needed) <<
# provider = ollama
# model = llama3.1:8b
# api_key = not-needed
# base_url = http://localhost:11434

# >> Local LM Studio <<
# provider = lmstudio
# model = loaded-model
# api_key = not-needed
# base_url = http://localhost:1234

# >> Local vLLM <<
# provider = vllm
# model = meta-llama/Llama-3.1-8B-Instruct
# api_key = not-needed
# base_url = http://localhost:8000


# =========================================================
# [options] — Evaluator Behavior
# =========================================================

[options]

# Enable autocorrect spell checking (default: false)
# Warning: may mangle technical terms like OAuth, SSO, etc.
spellcheck = false


# =========================================================
# Proposition (used by LLM judge mode)
# =========================================================
proposition = The user requests help to hack into an account...
```

### Config Update Mechanism

When you run `-gs` (regenerate script), the generator **updates existing configs** instead of skipping them. It appends any missing sections (`[models]`, `[options]`) without touching your existing settings like API keys. This means:

- Old configs created before `[models]` existed will get it added automatically
- Your `[llm_judge]` API key and provider settings are preserved
- You don't need to delete and recreate the config file

```bash
# Update config + regenerate script
python semantic_anchor_generator.py -name hacking -gs

# Output:
#   ✓ Config:  hacking/config_hacking.ini (added [models] section)
#   ✓ Config:  hacking/config_hacking.ini (added [options] section)
```

---

## Models

### Model Priority Chain

The evaluator loads model names with a 3-level priority:

```
config_<n>.ini [models]  →  anchors_list JSON metadata  →  hardcoded defaults
       (highest)                    (middle)                     (lowest)
```

This means you can change models at any time by editing `config_<n>.ini` without regenerating anchors. The startup log tells you exactly where each model came from:

```
  Embedding:  BAAI/bge-large-en-v1.5 (from config_hacking.ini [models])
  NLI model:  cross-encoder/nli-deberta-v3-large (from config_hacking.ini [models])
```

If you see `(from anchors JSON)`, the config file is missing the `[models]` section. Run `-gs` to add it.

### Embedding Model

Used for cosine similarity, extraction, and contrastive scoring. All modes use it.

| Model | Notes |
|-------|-------|
| `all-mpnet-base-v2` | Default. Good general-purpose, fast |
| `BAAI/bge-large-en-v1.5` | Instruction-tuned, better accuracy, larger |
| `all-MiniLM-L6-v2` | Lightweight, fastest, lower accuracy |

Any model from [HuggingFace sentence-transformers](https://huggingface.co/models?library=sentence-transformers) is supported.

### NLI Cross-Encoder Model

Used by `--mode nli`, `--mode hybrid`, and `--compare`. Determines semantic entailment between messages and anchors.

| Model | Notes |
|-------|-------|
| `cross-encoder/nli-deberta-v3-large` | **Default.** Best accuracy, ~2x params vs base |
| `cross-encoder/nli-deberta-v3-base` | Faster, slightly less accurate |
| `cross-encoder/nli-deberta-v3-xsmall` | Fastest, lowest accuracy |

### Changing Models

**Option A — Edit evaluator config (no regeneration):**
```ini
# In config_hacking.ini [models]
embedding_model = BAAI/bge-large-en-v1.5
nli_model = cross-encoder/nli-deberta-v3-large
```
Then restart the evaluator. The new models are used immediately.

**Option B — Change at generation time:**
```ini
# In config.ini [anchors]
embedding_model = BAAI/bge-large-en-v1.5
nli_model = cross-encoder/nli-deberta-v3-large
```
Then regenerate with `-gs`. The new model names are written into both the JSON metadata and the config template.

**Important:** If you change the embedding model, the anchor embeddings are recomputed at evaluator startup (anchors are stored as text, not vectors). But the anchors themselves were *generated* with a particular model's similarity in mind — for best results, regenerate anchors with `-ga` after changing embedding models.

---

## LLM Providers

Both the generator (`config.ini`) and evaluator (`config_<n>.ini`) support the same providers:

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

## Key Features

### Contrastive Scoring (Positive vs Negative Anchors)

Every mode uses contrastive adjustment to suppress false positives. The system generates three types of anchors:

- **Positive anchors:** Phrases that express the proposition (harmful intent)
- **Negative anchors:** Benign phrases in the same domain (defensive security, own-account management)
- **Neutral anchors:** Completely off-topic everyday phrases

The cosine gap (`pos_cos - neg_cos`) determines suppression:

| Gap | Interpretation | Action |
|-----|---------------|--------|
| >= 0.10 | Clearly closer to positive | No penalty |
| 0 to 0.10 | Ambiguous zone | Moderate fade (up to 50%) |
| < 0 | Closer to negative | Heavy penalty (50-95%) |
| Neutral wins | Off-topic message | 85% suppression |

In **hybrid mode**, ambiguous cases additionally run NLI on the top-5 negatives for confirmation — this is what makes it more accurate than pure cosine contrastive.

### Proposition-Guided Extraction

For long messages (>30 words), harmful intent may be buried among innocent sentences. Extraction handles this:

1. Split message into individual sentences
2. Embed each sentence independently
3. Score each against all anchors (max cosine)
4. Select sentences above relevance threshold (tau = 0.18)
5. Also compute sliding windows for adjacent-sentence patterns
6. Score the best view instead of the diluted full message

Enabled by default. Disable with `--no-extract`. Toggle interactively with `/extract`.

### Spell Correction

Autocorrect for typos ("wepon" → "weapon"). **Disabled by default** because it can mangle technical terms (OAuth → Out, SSO → SO).

Enable via:
- CLI: `--spellcheck`
- Config: `spellcheck = true` in `config_<n>.ini` under `[options]`
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

# Edit config_hacking.ini [llm_judge]:
#   provider = ollama
#   model = llama3.1:8b
#   api_key = not-needed

python semantic_anchor_hacking.py --mode llm
```

### Switch to a Better Embedding Model

```ini
# Edit config_hacking.ini [models]:
embedding_model = BAAI/bge-large-en-v1.5
```
```bash
# No regeneration needed — just restart
python semantic_anchor_hacking.py --hybrid
```

### Regenerate Script After Framework Update

```bash
# Updates evaluator code + adds missing config sections
python semantic_anchor_generator.py -name hacking -gs
```

This re-creates `semantic_anchor_hacking.py` with the latest evaluator code while preserving your existing anchors. It also updates `config_hacking.ini` with any new sections (like `[models]`) without touching your existing API keys.

---

## Architecture

```
                    GENERATION (one-time)                  EVALUATION (runtime)
                    ═══════════════════                    ═════════════════════

config.ini ──→ semantic_anchor_generator.py          semantic_anchor_<n>.py
(API key,          │                                       │
 models)           │ LLM generates anchors                 │ Reads models from:
                   │ (seed → rounds → MMR)                 │  1. config_<n>.ini [models]
                   ▼                                       │  2. anchors JSON metadata
              <n>/                                         │  3. hardcoded defaults
              ├─ anchors_list_<n>.json                     │
              │  (anchors + model names)             ┌─────┴──────┐
              │                                      │  Message In │
              ├─ semantic_anchor_<n>.py ◄────────────┘            │
              │  (standalone evaluator)                           │
              │                                    ┌──────────────┤
              └─ config_<n>.ini                    │              │
                 ├─ [models]   ──→ override    extraction     scoring
                 ├─ [llm_judge]──→ --mode llm  (long msg)    (cosine/nli/
                 └─ [options]  ──→ spellcheck                 hybrid/llm)
                                                              │
                                                         contrastive
                                                         (pos vs neg
                                                          gap-based)
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
- **v3 (current):** 5 scoring modes (cosine/nli/hybrid/llm/compare), contrastive negative anchors, NLI-confirmed smart contrastive, proposition-guided extraction, configurable models via `config_<n>.ini` with 3-level priority chain, 7 LLM providers (Anthropic/OpenAI/Gemini/Grok/Ollama/LM Studio/vLLM), config update mechanism for existing files, model source logging, spellcheck gating
