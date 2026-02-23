# Semantic Anchors Framework

A framework for evaluating natural-language messages against formal propositions using multiple scoring modes: **Cosine KNN voting**, **NLI entailment**, **Hybrid** (cosine + NLI with adaptive weighting), and **LLM-as-judge**.

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
nli_retrieve_k = 40
nli_vote_k = 20
nli_fwd_weight = 0.7
adaptive_hybrid = true
```

### Threshold Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `match_threshold` | 0.85 | Score >= this → MATCH |
| `warning_threshold` | 0.70 | Score >= this → WARNING |
| `knn_size` | 20 | Cosine KNN neighborhood size |
| `hybrid_nli_weight` | 0.75 | Base NLI weight in hybrid mode (cosine = 1 - this) |
| `nli_abstain_margin` | 0.15 | Below this evidence margin → abstain |
| `nli_retrieve_k` | 40 | Cosine pre-filter size for NLI re-ranking |
| `nli_vote_k` | 20 | KNN vote neighborhood after NLI re-ranking |
| `nli_fwd_weight` | 0.7 | Forward (message→anchor) weight in asymmetric NLI |
| `adaptive_hybrid` | true | Dynamically shift weights based on NLI confidence |

**Tuning for broad propositions**: If your proposition covers many subtopics (e.g., "financial fraud" spanning money laundering, tax evasion, forgery, etc.), increase `nli_retrieve_k` to 60–80 so the cosine pre-filter doesn't exclude relevant-subtopic anchors that are far in embedding space from the input.

## Anchor Types

The generator creates four types of anchors, each serving a distinct role in the KNN voting architecture:

### Positive Anchors
Standard examples that clearly match the proposition. Generated with diversity across categories (direct, euphemistic, fictional framing, etc.) and selected via MMR for maximum spread.

### Hard Positives (Boundary-Straddling)
Examples that genuinely match the proposition but are **phrased to sound benign or legitimate**. These are the most critical anchors for NLI KNN accuracy.

**Why they matter**: Without hard positives, polite or professional harmful messages (e.g., "How can I do a chargeback on a purchase I'm happy with?") get classified as benign because they are closer to negative anchors in embedding space than to standard positives. Hard positives occupy the boundary region between the positive and negative anchor clouds, ensuring harmful-but-polite messages have enough positive neighbors to win the KNN vote.

**Generation pipeline**:
1. The LLM sees both positive (obviously harmful) and negative (obviously benign) examples
2. It generates examples that genuinely match the proposition but mimic the phrasing style of negatives
3. A second round uses the best hard positives (those closest to negatives in embedding space) as models for further generation
4. Off-topic results are filtered, and MMR selects the most diverse final set

Configured via `hard_positive_ratio` in `config.ini` (default: 0.3 = 30% of target positive count).

### Negative Anchors (Contrastive)
Examples that use similar vocabulary to the proposition but have clearly legitimate intent. These compete directly with positives in KNN voting to reduce false positives. Generated at a 2:1 ratio to positives (configurable via `negative_ratio`).

### Neutral Anchors
Completely off-topic everyday messages used as a baseline. If a message is closer to neutral anchors than to any domain anchors, it is clearly off-topic.

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
2. **Cosine retrieval** — retrieve top candidates from the unified pool (positive + negative anchors) by cosine similarity (configurable via `nli_retrieve_k`, default 40)
3. **NLI re-rank** — run asymmetric NLI net scores (E − C) on the candidates. Asymmetric weighting configurable via `nli_fwd_weight` (default 70% forward + 30% backward). This prevents short anchors from inflating scores.
4. **KNN vote** — sort by NLI score, take top `nli_vote_k` (default 20). Evidence-weighted voting:
   - pos_evidence = sum of NLI scores for positive anchors in top-K
   - neg_evidence = sum of NLI scores for negative anchors in top-K
   - nli_knn_score = pos_evidence / (pos_evidence + neg_evidence)
5. **Final score** = max(proposition score, NLI KNN score)
6. **Abstain detection** — when evidence margin is below `nli_abstain_margin` (default 0.15), the result is flagged as ambiguous. Consider routing abstains to LLM judge.

Why NLI KNN works: negative anchors (refusals, security education, career advice) compete directly with positive anchors. A refusal about hacking has high NLI scores with both positive and negative anchors, but the negatives (which describe refusals and defensive content) score higher, pushing the ratio down. No gating or suppression logic needed — the voting handles it naturally.

### Hybrid with Adaptive Weighting (`--mode hybrid`)

Weighted blend of NLI KNN voting and Cosine KNN voting with **adaptive confidence-based weighting**:

```
hybrid_score = nli_weight × nli_knn_score + cos_weight × cosine_knn_score
```

#### Fixed Mode (`adaptive_hybrid = false`)

Uses constant weights. Default: **75% NLI + 25% cosine KNN** (configurable via `hybrid_nli_weight`).

#### Adaptive Mode (`adaptive_hybrid = true`, default)

Dynamically adjusts the NLI/cosine blend based on how confident NLI is for each message:

**NLI confidence** = how decisive the NLI KNN vote was (0 = split 50/50, 1 = unanimous):

| NLI Confidence | Weight Shift | Rationale |
|----------------|-------------|-----------|
| Low (< 0.3) | NLI 75% → 40%, Cosine 25% → 60% | NLI is confused (vote near 50/50), cosine neighborhood is more reliable |
| Medium (0.3–0.7) | No change (75/25) | Standard blend |
| High (> 0.7) | NLI 75% → 85%, Cosine 25% → 15% | NLI is decisive, trust it more |

**Max-lift**: If either mode independently exceeds the match/warning threshold but the blend doesn't, the hybrid score gets a floor boost (30% of the gap). This prevents one uncertain mode from dragging down a strong signal from the other.

**Why adaptive helps**: For broad propositions (e.g., "financial fraud" spanning many subtopics), NLI often produces near-50/50 votes because the positive anchors from *other* subtopics score poorly. Cosine KNN handles this better since it just measures neighborhood density. Adaptive weighting automatically detects this and shifts toward cosine, recovering accuracy that fixed 75/25 weighting would lose.

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
       |-- Generates hard positives (boundary-straddling, sound benign but match)
       |-- Generates negative anchors (2:1 ratio, near-miss examples)
       |-- Generates neutral anchors (off-topic baselines)
       |
       v
semantic_anchor_NAME.py + anchors_list_NAME.json + config_NAME.ini
       |
       |-- cosine mode:  KNN voting (sentence-transformers only)
       |-- nli mode:     NLI KNN voting (bi-encoder retrieve + cross-encoder re-rank)
       |-- hybrid mode:  Adaptive NLI/cosine blend (recommended)
       +-- llm mode:     LLM API call (requires API key)
```

No LLM is needed at evaluation time (except `--mode llm`). All local modes run entirely on CPU with sentence-transformers.

## Tuning Guide

### Broad Propositions (many subtopics)

If your proposition covers diverse territory (e.g., "financial fraud" = money laundering + tax evasion + forgery + chargeback fraud + insider trading):

1. **Increase `nli_retrieve_k`** to 60–80 — ensures cosine pre-filter doesn't exclude relevant-subtopic anchors
2. **Keep `adaptive_hybrid = true`** — automatically shifts weight toward cosine when NLI is uncertain about broad subtopics
3. **Generate more anchors** (`-n 40` or higher) — each subtopic needs dedicated positive coverage
4. **Hard positives help most here** — they fill the boundary gaps where polite/professional harmful requests live

### Narrow Propositions (single topic)

If your proposition is focused (e.g., "the user requests weapon construction instructions"):

1. **Default `nli_retrieve_k = 40`** is fine — anchor cloud is compact
2. **NLI works well** — default 75/25 hybrid blend is appropriate
3. **Fewer anchors suffice** (`-n 20` is often enough)
