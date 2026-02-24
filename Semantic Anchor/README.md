# Semantic Anchors Framework

A framework for evaluating natural-language messages against formal propositions using multiple scoring modes: **Cosine KNN**, **NLI KNN**, **Hybrid** (merged cosine + NLI KNN), and **LLM-as-judge**.

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
hard_positive_ratio = 0.3
hard_negative_ratio = 0.3
orthogonal_axes = true
mmr_anneal = true
adversarial_filter = true
variance_threshold = 0.15
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
python semantic_anchor_NAME.py --mode nli       # NLI KNN voting (accurate)
python semantic_anchor_NAME.py --mode hybrid    # Merged cosine + NLI KNN (recommended)
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
# KNN voting thresholds (all modes use simple counting):
#   score <= 0.50  →  NO MATCH
#   score >  0.50  →  WARNING
#   score >  0.70  →  MATCH
match_threshold = 0.70
warning_threshold = 0.50

# KNN neighborhood size (default: 20)
knn_size = 20

# NLI cosine pre-filter size (default: 40)
nli_retrieve_k = 40

# NLI vote neighborhood size (default: 20)
nli_vote_k = 20

# NLI asymmetric forward weight (default: 0.7)
# 0.7 = 70% forward (message->anchor) + 30% backward (anchor->message)
nli_fwd_weight = 0.7

# NLI abstain margin (default: 0.15)
# When vote margin is below this, flag as abstain
nli_abstain_margin = 0.15
```

## Scoring Modes

All modes use the same principle: **KNN voting**. Find the K nearest neighbors, count how many are positive. `score = positive_count / total_count`.

### Thresholds

| Score | Verdict |
|-------|---------|
| ≤ 50% (e.g. 10/20) | NO MATCH |
| > 50% to ≤ 70% (e.g. 11-14/20) | WARNING |
| > 70% (e.g. 15+/20) | MATCH |

Configurable via `match_threshold` and `warning_threshold` in `config_NAME.ini`.

### Cosine KNN (`--mode cosine`)

```
Message → embed → find K=20 nearest by cosine → count positive / total
```

Merges positive and negative anchors into one pool, sorts by cosine similarity, takes top-K, counts. Fast (embedding model only).

### NLI KNN (`--mode nli`)

```
Message → embed → cosine pre-filter top 40 → NLI re-rank → take top 20 → count positive / total
```

Uses cosine to retrieve candidates, then a cross-encoder (NLI) to re-rank by entailment strength. The re-ranking determines which 20 make the final vote, but scoring is still a simple count.

Why NLI KNN works: negative anchors compete directly with positive anchors in the vote. A refusal about hacking has high NLI scores with both sides, but the negatives score higher in re-ranking, displacing positives from the top-K. No gating or suppression needed — the voting handles it naturally.

### Hybrid (`--mode hybrid`, recommended)

```
Cosine KNN → 20 voters
NLI KNN    → 20 voters
Merge (deduplicate) → up to 40 unique voters → count positive / total
```

Takes the top-20 from cosine and the top-20 from NLI, merges into a union set (removing duplicates), and counts how many are positive. Combines cosine's spatial signal with NLI's semantic re-ranking while keeping scoring simple.

### LLM-as-Judge (`--mode llm`)

Sends the proposition and message to an LLM API. Requires `config_NAME.ini` with provider credentials. Supports: Anthropic, OpenAI, Gemini, Grok, Ollama, LM Studio, vLLM.

### Compare (`--compare`)

Runs cosine, NLI, hybrid (and LLM if configured) side-by-side with verdict, score, and top-3 anchors per mode.

## Long Message Handling

Messages over 40 words use **proposition-guided extraction**: the embedding model identifies relevant sentences before scoring. Three views are scored and the best is used:
1. **Extracted**: Only relevant sentences
2. **Full**: Entire message
3. **Window**: Best contiguous text window

## Interactive Commands

| Command | Action |
|---------|--------|
| `/verbose` | Toggle full anchor table |
| `/top N` | Show top N anchors |
| `/mode MODE` | Switch scoring mode |
| `/compare` | Toggle compare mode |
| `/extract` | Toggle extraction |
| `/spell` | Toggle spell correction |
| `/quit` | Exit |

## Automatic Subtopic Decomposition

When a proposition covers multiple behavior types (e.g., "financial fraud" = chargebacks + laundering + tax evasion + forgery), the generator **automatically decomposes into subtopics** with balanced coverage.

1. **Decomposition**: LLM identifies 3-8 distinct subtopics
2. **Per-subtopic seeds**: Dedicated seeds guarantee initial coverage
3. **Coverage audit**: Anchors classified to nearest subtopic after MMR
4. **Gap-filling**: Underrepresented subtopics get targeted generation
5. **Applies to both positive and negative anchors**

Without decomposition, KNN fails on underrepresented subtopics because the K nearest neighbors come from other subtopics. With decomposition, each gets proportional allocation.

## Adversarial Semantic Mapping

Five strategies for near-perfect anchor distribution, all enabled by default.

1. **Orthogonal Axes** (`orthogonal_axes = true`): LLM identifies 8-12 domain-specific semantic axes merged with static categories for comprehensive coverage.

2. **Annealing MMR** (`mmr_anneal = true`): Decaying λ from 0.8→0.2. Captures core intent first, then fills boundary fringes.

3. **Adversarial Filtering** (`adversarial_filter = true`): Scores hard negative candidates against KNN. Prioritizes those producing highest false-positive rates.

4. **Syntactic Jittering** (always on): 20/20/20/20/20 distribution across imperative, hypothetical, analytical, slang, and narrative structures.

5. **Variance Monitoring** (`variance_threshold = 0.15`): Detects near-identical examples within categories and regenerates with higher diversity.

## Architecture

```
config.ini (LLM provider)
       |
       v
semantic_anchor_generator.py
       |-- Subtopic decomposition + orthogonal axes
       |-- Positive anchors (per-subtopic + annealing MMR)
       |-- Negative anchors (boundary-aware mirrors)
       |-- Hard positives + hard negatives (adversarial-filtered)
       |-- Variance monitoring + gap-filling
       |-- Neutral anchors (off-topic baselines)
       |
       v
semantic_anchor_NAME.py + anchors_list_NAME.json + config_NAME.ini
       |
       |-- cosine:  KNN count (top-K by cosine similarity)
       |-- nli:     KNN count (cosine pre-filter → NLI re-rank → top-K)
       |-- hybrid:  KNN count (union of cosine + NLI top-K)
       +-- llm:     LLM API call
```

No LLM needed at evaluation time (except `--mode llm`). All local modes run on CPU.

## Tuning Guide

### Broad Propositions (many subtopics)

1. Subtopic decomposition is automatic
2. Increase `nli_retrieve_k` to 60–80 so the pre-filter doesn't exclude relevant subtopics
3. Generate more anchors (`-n 40`+) — each subtopic needs dedicated coverage
4. Hard positives fill boundary gaps, hard negatives reduce false positives

### Narrow Propositions (single topic)

1. Default `nli_retrieve_k = 40` is fine
2. Fewer anchors suffice (`-n 20`)
