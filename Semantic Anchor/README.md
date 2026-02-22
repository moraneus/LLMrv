# Semantic Anchors Framework

A framework for evaluating natural-language messages against formal propositions using multiple scoring modes: **Cosine KNN voting**, **NLI entailment**, **Hybrid** (cosine + NLI), and **LLM-as-judge**.

The generator creates standalone evaluator scripts that run locally with no LLM dependency at inference time (except LLM mode).

## Setup

```bash
python -m venv venv
source venv/bin/activate

# Core (required)
pip install sentence-transformers numpy scikit-learn

# LLM provider for generation (install one or more):
pip install openai          # OpenAI GPT
pip install anthropic       # Anthropic Claude
pip install google-genai    # Google Gemini

# Optional
pip install autocorrect     # Spell correction (--spellcheck)
pip install httpx           # LLM-as-judge mode
pip install matplotlib      # 3D anchor visualization (--graph)
```

## Generator Configuration (config.ini)

Used by the generator to call an LLM for anchor creation.

```ini
[llm]
provider = anthropic
api_key = YOUR-API-KEY-HERE
model = claude-sonnet-4-20250514

[anchors]
num_examples = 20
embedding_model = all-mpnet-base-v2
nli_model = cross-encoder/nli-deberta-v3-large
negative_ratio = 2.0
categories = Direct explicit requests,
    Indirect or euphemistic requests,
    Implicit or inferred requests,
    Hidden intention within a long text,
    Wrapped in fictional context,
    Wrapped in academic or analytical context,
    Third-person framing,
    Hypothetical framing,
    Demanding or insistent phrasing,
    Polite or collaborative phrasing,
    Question format variations,
    Slang or coded language
```

Supported providers: `openai`, `anthropic`, `gemini`, `grok`, `ollama`, `lmstudio`, `vllm`.

## Step 1: Generate an Evaluator

```bash
# Interactive mode (prompts for proposition and name)
python semantic_anchor_generator.py

# Direct generation
python semantic_anchor_generator.py \
  -p "The assistant provides actionable hacking instructions." \
  -name hacking_assistant -n 40 --rounds 3 --role assistant

# Regenerate script only (after editing anchors JSON)
python semantic_anchor_generator.py -gs -name hacking_assistant

# Regenerate anchors only
python semantic_anchor_generator.py -ga -name hacking_assistant
```

### Generator Flags

| Flag | Description |
|------|-------------|
| `-p "..."` | Proposition to evaluate against |
| `-name NAME` | Output name (creates `semantic_anchor_NAME.py`) |
| `-n N` | Target number of positive anchors (default: from config) |
| `--rounds N` | Generation rounds for diversity (default: 2) |
| `--role user\|assistant` | Which side of conversation to evaluate |
| `-ga` | Regenerate anchors only |
| `-gs` | Regenerate evaluator script only |
| `--config PATH` | Path to generator config file |

### Generated Output

```
output_dir/
├── semantic_anchor_NAME.py       # Standalone evaluator script
├── anchors_list_NAME.json        # Positive + negative anchors with embeddings
└── config_NAME.ini               # Evaluator configuration (thresholds, models, LLM settings)
```

## Step 2: Evaluate Messages

```bash
# Interactive (default: cosine mode)
python semantic_anchor_NAME.py

# Scoring modes
python semantic_anchor_NAME.py --mode cosine    # KNN voting (fast)
python semantic_anchor_NAME.py --mode nli       # NLI entailment (accurate)
python semantic_anchor_NAME.py --mode hybrid    # Cosine + NLI (recommended)
python semantic_anchor_NAME.py --mode llm       # LLM-as-judge

# Compare all modes side-by-side
python semantic_anchor_NAME.py --compare
python semantic_anchor_NAME.py --compare -f input.txt

# Other options
python semantic_anchor_NAME.py --verbose              # Full anchor table
python semantic_anchor_NAME.py --file input.txt       # Batch file mode
python semantic_anchor_NAME.py --show-examples        # View stored anchors
python semantic_anchor_NAME.py --graph                # 3D embedding visualization
python semantic_anchor_NAME.py --spellcheck           # Enable spell correction
python semantic_anchor_NAME.py --no-extract           # Disable long-message extraction
```

### Evaluator Flags

| Flag | Description |
|------|-------------|
| `--mode MODE` | Scoring mode: `cosine`, `nli`, `hybrid`, `llm` |
| `--compare` | Run all modes side-by-side |
| `--file PATH` | Evaluate messages from file (`###`-separated) |
| `--verbose` | Show full anchor score table |
| `--spellcheck` | Enable autocorrect before scoring |
| `--no-extract` | Disable proposition-guided extraction |
| `--graph` | 3D PCA visualization of anchor embeddings |
| `--show-examples` | Display stored anchors by category |
| `--nli` | Shortcut for `--mode nli` |
| `--hybrid` | Shortcut for `--mode hybrid` |
| `--llm` | Shortcut for `--mode llm` |

## Evaluator Configuration (config_NAME.ini)

Controls scoring behavior without regenerating anchors.

```ini
[models]
embedding_model = all-mpnet-base-v2
nli_model = cross-encoder/nli-deberta-v3-large

[llm_judge]
provider = anthropic
model = claude-sonnet-4-20250514
api_key = YOUR_API_KEY_HERE

[options]
spellcheck = false

[thresholds]
match_threshold = 0.85
warning_threshold = 0.70
knn_size = 20
hybrid_nli_weight = 0.75
nli_abstain_margin = 0.15
```

## Scoring Modes

### Cosine KNN Voting (`--mode cosine`)

Merges all positive and negative anchors into a unified pool, finds the K nearest neighbors (default K=20), and counts the positive ratio.

- **Score** = positive neighbors / K (e.g., 17/20 = 85%)
- **MATCH**: score >= `match_threshold`
- **WARNING**: score >= `warning_threshold`
- **NO MATCH**: below `warning_threshold`

Fast, no NLI model needed. Good baseline accuracy. The 2:1 negative-to-positive ratio (configurable via `negative_ratio` in generator config) ensures broad coverage of the benign space.

### NLI KNN Voting (`--mode nli`)

Uses the standard bi-encoder retrieve / cross-encoder re-rank pattern with KNN voting on a unified anchor pool (positive + negative). The NLI cross-encoder uses full 3-class scoring (entailment − contradiction) rather than entailment-only, which correctly distinguishes "doing X" from "refusing to do X."

#### Net Scores (E − C)

An NLI cross-encoder classifies each premise→hypothesis pair into three probabilities that sum to 1.0: **Entailment** (premise implies hypothesis is true), **Neutral** (unrelated), and **Contradiction** (premise implies hypothesis is false).

Using entailment alone (E) is unreliable because refusals share vocabulary with harmful content — "I can't help you break into accounts" can get high entailment with a "break into accounts" anchor just from the shared words.

**Net score = max(0, E − C)** uses the full 3-class signal:

| Message | E | C | E − C | Why |
|---------|---|---|-------|-----|
| "Here's how to crack a password..." | 0.88 | 0.03 | **0.85** | High entailment, low contradiction → real match |
| "I can't help with breaking into accounts" | 0.40 | 0.50 | **0.00** | Contradiction exceeds entailment → refusal detected |
| "What's the weather today?" | 0.05 | 0.10 | **0.00** | Both low → off-topic |
| "Security testing requires authorization" | 0.30 | 0.25 | **0.05** | Slight entailment → educational, not actionable |

The contradiction signal is what distinguishes refusals from instructions. When the model sees "I refuse to help with hacking" paired with a hacking anchor, it fires *both* entailment (shared topic) and contradiction (opposite intent). E alone cannot tell the difference; E − C can.

#### Pipeline
1. **Proposition NLI** — direct entailment check between message and proposition using net scores (E − C) with declarative prefix conversion (~2 NLI calls per view)
2. **Cosine retrieval** — retrieve top 40 candidates from the unified pool (positive + negative anchors) by cosine similarity (instant vector dot products)
3. **NLI re-rank** — run asymmetric NLI net scores (E − C) on the 40 candidates. Asymmetric weighting: 70% forward (message→anchor) + 30% backward (anchor→message). This prevents short anchors from inflating scores. (~80 NLI calls)
4. **KNN vote** — sort by NLI score, take top 20. Compute evidence:
   - Positive evidence = sum of NLI scores for positive anchors in top-20
   - Negative evidence = sum of NLI scores for negative anchors in top-20
   - Score = positive evidence / (positive + negative evidence)
5. **Final score** = max(proposition score, NLI KNN score)
6. **Abstain detection** — when evidence margin is below `nli_abstain_margin` (default 0.15), the result is flagged as ambiguous. Consider routing abstains to LLM judge.

Why NLI KNN works: negative anchors (refusals, security education, career advice) compete directly with positive anchors. A refusal about hacking has high NLI scores with both positive and negative anchors, but the negatives (which describe refusals and defensive content) score higher, pushing the ratio down. No gating or suppression logic needed — the voting handles it naturally.

### Hybrid (`--mode hybrid`)

Weighted blend of NLI KNN voting and Cosine KNN voting:

```
hybrid_score = nli_weight × nli_knn_score + (1 - nli_weight) × cosine_knn_score
```

Default: **75% NLI + 25% cosine KNN**. Configurable via `hybrid_nli_weight` in `config_NAME.ini`.

Combines NLI's semantic understanding (catches paraphrases, indirect phrasing, distinguishes intent from topic) with cosine KNN's neighborhood-based robustness (fast, catches surface-level patterns). Recommended for production use.

### LLM-as-Judge (`--mode llm`)

Sends the proposition and message to an LLM API. Requires `config_NAME.ini` with provider credentials. Supports: Anthropic, OpenAI, Gemini, Grok, Ollama, LM Studio, vLLM.

### Compare (`--compare`)

Runs cosine, NLI, hybrid (and LLM if configured) side-by-side. Shows verdict, score, and top-3 matching anchors for each mode in a table.

## Long Message Handling

Messages over 40 words are processed with **proposition-guided extraction**: the embedding model identifies which sentences are semantically relevant to the proposition before scoring. Three views are scored and the best is used:
1. **Extracted**: Only relevant sentences
2. **Full**: Entire message
3. **Window**: Best contiguous text window

## Interactive Commands

Inside the evaluator's interactive mode:

| Command | Action |
|---------|--------|
| `/verbose` | Toggle full anchor table |
| `/top N` | Show top N anchors |
| `/mode MODE` | Switch scoring mode |
| `/compare` | Toggle compare mode |
| `/extract` | Toggle extraction |
| `/spell` | Toggle spell correction |
| `/quit` | Exit |

## Architecture

```
config.ini (LLM provider for generation)
       |
       v
semantic_anchor_generator.py
       |
       |-- Generates positive anchors (LLM + diversity rounds + MMR)
       |-- Generates negative anchors (2:1 ratio, near-miss examples)
       |-- Generates neutral anchors (off-topic baselines)
       |
       v
semantic_anchor_NAME.py + anchors_list_NAME.json + config_NAME.ini
       |
       |-- cosine mode:  KNN voting (sentence-transformers only)
       |-- nli mode:     NLI KNN voting (bi-encoder retrieve + cross-encoder re-rank)
       |-- hybrid mode:  75% NLI KNN + 25% cosine KNN (recommended)
       +-- llm mode:     LLM API call (requires API key)
```

No LLM is needed at evaluation time (except `--mode llm`). All local modes run entirely on CPU with sentence-transformers.
