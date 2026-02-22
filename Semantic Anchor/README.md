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
nli_prop_suppress = 0.35
```

## Scoring Modes

### Cosine KNN Voting (`--mode cosine`)

Merges all positive and negative anchors into a unified pool, finds the K nearest neighbors (default K=20), and counts the positive ratio.

- **Score** = positive neighbors / K (e.g., 17/20 = 85%)
- **MATCH**: score >= `match_threshold`
- **WARNING**: score >= `warning_threshold`
- **NO MATCH**: below `warning_threshold`

Fast, no NLI model needed. Good baseline accuracy. The 2:1 negative-to-positive ratio (configurable via `negative_ratio` in generator config) ensures broad coverage of the benign space.

### NLI Entailment (`--mode nli`)

Uses a cross-encoder NLI model (DeBERTa) for entailment scoring, combined with cosine similarity for gating.

Pipeline:
1. **Proposition NLI** — direct entailment check between the message and the proposition (~2 NLI calls per view)
2. **Cosine on ALL anchors** — computes similarity to all positive and negative anchors (fast vector dot products, no NLI) to produce a cosine gap (pos_cos − neg_cos)
3. **NLI on top-K positive anchors only** — raw entailment on the ~10 closest positive anchors (~20 NLI calls). Negative anchors are skipped because NLI gives ~0.99 to both sides (shared vocabulary), so it adds no signal.
4. **Gap gates anchor NLI** — the cosine gap decides how much to trust the anchor NLI score. Positive gap (closer to harmful) → trust it. Negative gap (closer to benign) → suppress to zero.
5. **Proposition gates final score** — the proposition NLI is the strongest semantic signal and gates how much anchors can contribute:
   - prop < `nli_prop_suppress` (default 0.35): proposition confident it's NOT a match → suppress anchors entirely (catches refusals and off-topic)
   - prop < `warning_threshold`: ambiguous → anchors contribute, damped by proposition confidence
   - prop >= `warning_threshold`: proposition already flagging → anchors confirm freely

### Hybrid (`--mode hybrid`)

Weighted blend of NLI entailment and Cosine KNN voting:

```
hybrid_score = nli_weight × nli_score + (1 - nli_weight) × knn_score
```

Default: **75% NLI + 25% cosine KNN**. Configurable via `hybrid_nli_weight` in `config_NAME.ini`.

This gives you NLI's semantic understanding (catches paraphrases, indirect phrasing) combined with KNN's neighborhood-based robustness (catches vocabulary overlap, surface-level patterns). Recommended for production use.

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
       |-- nli mode:     Cross-encoder entailment (+ sentence-transformers)
       |-- hybrid mode:  Cosine fast-path + NLI (recommended)
       +-- llm mode:     LLM API call (requires API key)
```

No LLM is needed at evaluation time (except `--mode llm`). All local modes run entirely on CPU with sentence-transformers.
