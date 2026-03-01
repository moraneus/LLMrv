# Semantic Anchors Framework

Evaluates natural-language messages against formal propositions using KNN voting across multiple scoring modes: **Cosine KNN**, **NLI KNN**, **Hybrid** (merged cosine + NLI KNN), and **LLM** (direct LLM match check).

The generator creates standalone evaluator scripts that run locally with no LLM dependency at inference time (except LLM mode).

## Setup

```bash
python -m venv venv
source venv/bin/activate

# Core
pip install sentence-transformers numpy scikit-learn

# LLM provider for generation (one or more):
pip install openai anthropic google-genai

# Optional
pip install autocorrect     # Spell correction (--spellcheck)
pip install matplotlib      # Visualization (--graph)
pip install umap-learn      # Better graph projections (recommended)
pip install scipy           # Distance fidelity metric for --graph
```

## Generator Configuration (config.ini)

```ini
[llm]
provider = anthropic
api_key = YOUR-API-KEY-HERE
model = claude-sonnet-4-20250514

[anchors]
num_examples = 20             # auto-adjusted upward based on category count
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

### Role-Specific Categories

Categories are resolved with priority: `categories_{role}` > `categories` (generic fallback). When `--role user` is set, the generator looks for `categories_user` first; when `--role assistant`, it looks for `categories_assistant`. If neither exists, `categories` is used.

Built-in defaults (used when no config.ini override exists):

- **`categories_user`** (28 categories) — focused on *how people ask*: direct/indirect/euphemistic requests, fictional/academic/hypothetical framing, question format variations, slang, adversarial phrasing, multi-step requests, constraint-based requests, etc.
- **`categories_assistant`** (21 categories) — focused on *how the AI responds*: informational, instructional, analytical, creative, conversational, emotional support, refusal/boundary-setting, risk mitigation, structured data outputs, step-by-step formatting, etc.

Override in config.ini:

```ini
[anchors]
categories_user = Direct explicit requests,
    Indirect or euphemistic requests,
    Hypothetical framing
categories_assistant = Informational responses,
    Refusal or boundary setting responses,
    Risk mitigation responses
```

Supported providers: `openai`, `anthropic`, `gemini`, `grok`, `openrouter`, `ollama`, `lmstudio`, `vllm`.

**OpenRouter** lets you access 200+ models (Claude, GPT, Gemini, Llama, Mistral, etc.) through a single API key. Get your key at [openrouter.ai/keys](https://openrouter.ai/keys), then set `provider = openrouter` and use model names like `anthropic/claude-sonnet-4`, `google/gemini-2.5-flash`, `meta-llama/llama-4-scout`, etc.

## Step 1: Generate an Evaluator

```bash
python semantic_anchor_generator.py \
  -p "The assistant provides actionable hacking instructions." \
  -name hacking_assistant -n 40 --rounds 3 --role assistant
```

| Flag | Description |
|------|-------------|
| `-p "..."` | Proposition to evaluate against |
| `-name NAME` | Output name (creates `NAME/` folder) |
| `-n N` | Target positive anchors (auto-adjusted upward if needed) |
| `--rounds N` | Generation rounds (default: 2) |
| `--role user\|assistant` | Which side of conversation to evaluate |
| `-ga` | Regenerate anchors only |
| `-gs` | Regenerate evaluator script only |
| `--train N` | Anchor Manifold Optimization: train embeddings on N synthetic edge cases |

Output:

```
NAME/
├── semantic_anchor_NAME.py        # Standalone evaluator
├── anchors_list_NAME.json         # Anchors with metadata
├── config_NAME.ini                # Evaluator thresholds and models
├── training_report_NAME.txt       # (after --train) K-fold results + accuracy log
└── optimized_anchors_NAME.npz     # (after --train) Optimized embeddings
```

## Step 2: Evaluate Messages

```bash
python semantic_anchor_NAME.py                        # Interactive (cosine)
python semantic_anchor_NAME.py --mode hybrid          # Recommended mode
python semantic_anchor_NAME.py --compare -f input.txt # All modes side-by-side
python semantic_anchor_NAME.py --auto-calibrate labeled.txt  # ROC calibration
python semantic_anchor_NAME.py --graph                # Distribution visualization
```

| Flag | Description |
|------|-------------|
| `--mode MODE` | `cosine`, `nli`, `hybrid`, `llm` |
| `--compare` | All modes side-by-side |
| `--file PATH` | Batch mode (`###`-separated messages) |
| `--auto-calibrate FILE` | Optimize thresholds from labeled data |
| `--graph` | 2D projection + similarity histograms |
| `--verbose` | Full anchor score table |
| `--spellcheck` | Autocorrect before scoring |
| `--no-extract` | Disable long-message extraction |

## Scoring Modes

All KNN modes work the same way at a high level: given an incoming message, find its K nearest anchors from the combined positive + negative pool, then count how many of those K neighbors are positive. `score = positive_count / K`. The modes differ in *how* they measure "nearest."

| Score | Verdict |
|-------|---------|
| ≤ warning_threshold (default 50%) | NO MATCH |
| > warning to ≤ match_threshold (default 70%) | WARNING |
| > match_threshold | MATCH |

Thresholds are configurable in `config_NAME.ini` and can be auto-calibrated from labeled data with `--auto-calibrate`.

### Cosine KNN (default)

```
Message → embed → cosine similarity against all anchors → top K=20 → count positive / K
```

Both the message and every anchor are converted to dense vectors using a sentence embedding model (e.g. `all-mpnet-base-v2`). Cosine similarity measures the angle between two vectors — 1.0 means identical direction, 0.0 means orthogonal, -1.0 means opposite. The message is compared against every anchor in the pool, the K=20 most similar anchors are selected, and the score is the fraction that are positive.

Cosine captures *topical* similarity — messages that talk about the same subject using similar vocabulary will score high. It's fast (just dot products), needs no external API, and works well when positive and negative anchors are topically distinct. It struggles when a benign message uses the same terminology as a harmful one (e.g. "how to secure a network against hackers" vs "how to hack a network"), because cosine sees both as close in embedding space.

### NLI KNN

```
Message → cosine top 40 → NLI cross-encoder scores (E−C) → re-rank → top K=20 → count positive / K
```

Natural Language Inference (NLI) is a classification task: given two sentences, determine if the first *entails* the second, *contradicts* it, or is *neutral*. A cross-encoder model (e.g. `nli-deberta-v3-large`) processes the message paired with each anchor and outputs three probabilities: entailment (E), contradiction (C), and neutral (N).

The ranking score is `E − C`: anchors where the message strongly entails the anchor's meaning rise to the top; anchors that contradict the message get pushed down, even if they share vocabulary. This handles the weakness of cosine — a message about "defending against hacking" will *contradict* a positive anchor about "performing hacking," pushing it down despite high cosine similarity.

Since NLI scoring is expensive (one cross-encoder forward pass per anchor), cosine pre-filters to the top `nli_retrieve_k` candidates (default 40) before NLI re-ranking. Increase to 60–80 for broad propositions where relevant anchors might not be in the cosine top 40.

### Hybrid (recommended)

```
Message → cosine top 20 ──┐
                           ├─→ agreement-weighted score → points / pool_size
Message → NLI top 20 ─────┘
```

Runs cosine KNN and NLI KNN independently, each selecting their own top K/2 neighbors. Then combines with agreement weighting:

| Anchor appears in | Points | Reasoning |
|--------------------|--------|-----------|
| Both cosine + NLI positive | 3 pts | Strong signal — both systems agree it's relevant |
| One system positive only | 1 pt | Moderate signal — one system found it relevant |
| One system negative only | 0 pts | Weak signal — only one system disagrees |
| Both cosine + NLI negative | −1 pt | Penalty — both systems agree it's irrelevant |

`score = max(0, points) / pool_size`. With default `hybrid_pool_size = 40`: 20 cosine + 20 NLI voters, always scored out of 40. Displayed as `14cos + 11nli +5dup −2neg = 28/40 (70%)`.

This catches cases where cosine is fooled by vocabulary overlap (NLI corrects it) and cases where NLI misreads intent on a short message (cosine corrects it). Set `hybrid_pool_size = 80–100` for broad propositions.

### LLM

Sends the proposition and message directly to an LLM API and asks whether the message matches. This is the most flexible mode — the LLM can reason about context, intent, and nuance that embedding models miss — but it requires API credentials, costs money per call, adds latency, and results are non-deterministic. Useful as a validation baseline when developing anchor sets.

### Compare

Runs all available modes side-by-side on the same message with verdicts, scores, and top anchors for each. Useful for understanding where modes agree and disagree.

## Generation Pipeline

```
1. Subtopic decomposition        → identify distinct aspects of proposition
2. Orthogonal axes generation    → domain-specific semantic dimensions
3. Positive anchor generation    → cluster-constrained MMR selection
4. Negative anchor generation    → dynamic false-positive categories
5. Hard positives + negatives    → boundary-straddling examples
6. Subtopic KNN parity           → enforce minimum anchors per subtopic
7. Boundary targeting             → logistic regression → weak zone filling
8. Hard example mining            → synthetic queries → failure-targeted fixes
9. Neutral anchors                → off-topic baselines
```

### Maximal Marginal Relevance (MMR)

MMR is a selection algorithm that balances relevance and diversity. At each step, candidates are scored as:

```
MMR(c) = λ × similarity(c, target) − (1−λ) × max_similarity(c, already_selected)
```

The first term favors anchors close to the proposition; the second penalizes anchors too similar to ones already picked. λ controls the tradeoff: 1.0 = pure relevance (near-duplicates), 0.0 = pure diversity (random spread), 0.5–0.7 = typical sweet spot. Without MMR, the LLM might generate 20 anchors that are minor paraphrases of the same idea — MMR forces each successive pick to cover a different angle.

**MMR annealing** (`mmr_anneal = true`): Starts λ high (favoring relevance to grab the strongest anchors first) and gradually lowers it (favoring diversity to fill gaps in later picks).

**Cluster-constrained MMR**: Standard MMR can still allow micro-clusters of 5+ similar anchors. The cluster-constrained variant adds k-means clustering over all candidates and enforces `max_per_cluster` during selection — once a cluster hits its cap, remaining members are skipped regardless of their MMR score. Falls back to standard MMR if scikit-learn is unavailable.

### Subtopic KNN Parity

After generation, audits each subtopic's anchor count against `equal_share × 0.9` (90% of equal distribution). Subtopics below threshold get targeted gap-fill anchors with cosine diversity filtering (>85% similarity rejected) and retry-until-full loops (up to 5 rounds per subtopic).

**Iterative balancing**: Gap-fill runs in a loop (up to 3 rounds). After filling underrepresented subtopics, the total anchor count increases, shifting `equal_share` upward — which can push previously-passing subtopics below the new threshold. The loop re-audits and re-fills until all subtopics pass or no new diverse anchors can be generated. This applies to all three gap-fill sites: the positive pipeline, negative pipeline, and `enforce_subtopic_parity`.

Display labels: `← UNDERREPRESENTED` (< 60% equal share), `← LOW` (< 90% equal share).

### Boundary Targeting

Trains logistic regression on current positive/negative embeddings, identifies samples in the bottom 25% by decision margin, clusters them into distinct weak zones, then generates anchors specifically for each zone. Replaces static ratios with geometric feedback — anchors go where the boundary is weakest.

### Hard Example Mining

Self-improving loop: generates 200 synthetic test queries (half harmful, half benign), scores them against the current anchor set with KNN, collects false positives and false negatives, then generates targeted fix anchors for each failure pattern.

### Dynamic Hard Negative Categories

Hard negative categories are generated per-proposition by the LLM, not hardcoded. The LLM identifies 8-10 domain-specific false-positive scenarios (safety education, entertainment, legal regulations, storage/handling, historical collecting, etc.) with examples before generating actual hard negatives.

### ROC Auto-Calibration

```bash
python semantic_anchor_NAME.py --auto-calibrate labeled.txt
```

Reads a labeled file, runs KNN scoring on each message, scans all threshold combinations, and selects the pair maximizing F1 score. Updates `config_NAME.ini` automatically. Labeled file format:

```
This is a harmful message [MATCH]
###
This is a benign message [CLEAN]
```

### Anchor Manifold Optimization (AMO)

```bash
python semantic_anchor_generator.py -name NAME --train 200
```

Treats anchor embeddings as trainable parameters and optimizes them via gradient descent on adversarial synthetic data. Transforms the evaluator from a static vector lookup into a specialized detector that learns exactly where your decision boundary lies.

**How it works:**

1. **Adversarial pair generation**: The LLM generates N examples as two adversarial types:
   - **STEALTH** (positive label): Harmful intent disguised with benign vocabulary — academic framing, fictional wrapping, professional jargon, buried multi-step requests. Designed to evade detection.
   - **TRAP** (negative label): Benign intent loaded with dangerous-sounding keywords — security researchers, compliance auditors, victims describing incidents, bug bounty disclosures. Designed to trigger false positives.
   - **Role-aware categories**: Examples are generated from the correct POV. With `--role user`, the LLM generates user messages distributed across user categories (direct requests, euphemistic, hypothetical, etc.). With `--role assistant`, it generates AI responses distributed across response categories (informational, instructional, refusal, etc.). Categories come from `config.ini`.
   - **Zero-overlap rule**: The full existing anchor set is passed to the LLM with explicit instructions to generate 100% novel scenarios.
   - **Guaranteed target count**: A while-loop runs until the exact target is met — there is no round cap. Only stall detection (3 consecutive rounds producing 0 new examples) can stop it early. Each retry uses a different creative direction (angle rotation) and shows a random sample of already-generated examples to the LLM.
2. **Embedding diversity filter**: After generation, all examples are embedded and within each class, near-duplicate paraphrases (cosine similarity > `synthetic_diversity_threshold`, default 0.95) are removed. If removals reduce count below target, regenerates and re-checks (up to 3 rounds).
3. **K-fold cross-validation**: Stratified k-fold (default 5 folds) trains independently on each fold to find the optimal epoch count. The epoch with the highest mean validation accuracy across all folds is selected. This replaces a single train/dev split — every example serves as validation exactly once, eliminating noisy checkpoint selection from small dev sets.
4. **Soft top-K KNN scoring**: Differentiable approximation of the evaluator's hard KNN. Selects the K nearest anchors (configurable via `train_knn_k`, default 20 to match the evaluator), then applies softmax-weighted voting: `score = Σ(label_i × softmax(sim_i / τ))`. Temperature τ (default 0.05) must be low so the closest neighbors dominate; high τ (0.5+) flattens weights to uniform, producing vanishing gradients.
5. **Optimization**: Adam optimizer with BCE loss and cosine LR annealing nudges anchor vectors — positive anchors move toward stealth examples, negative anchors move toward trap examples.
6. **Cosine anchoring constraint**: After each epoch, any anchor whose cosine similarity to its original position drops below `min_similarity` (default 0.85) is projected back. Prevents anchors from becoming abstract math points that lose semantic meaning.
7. **Final training**: Trains on ALL data (no held-out set) for exactly the epoch count selected by k-fold CV. This maximizes use of every synthetic example.

**AMO status indicator**: The evaluator prints `AMO: ACTIVE (using trained embeddings)` or `AMO: OFF (using original embeddings)` at startup.

**Configuration** (in `config.ini` `[training]` section — NOT in `config_NAME.ini`):

```ini
[training]
# --- Optimizer ---
learning_rate = 0.001       # Adam step size (cosine annealing decays to 10% of this)
epochs = 20                 # Max passes per fold (k-fold selects actual count)
temperature = 0.05          # Softmax sharpness: MUST be low (<0.1) for gradient signal.
                            #   0.05 = closest 2-3 neighbors dominate the vote.
                            #   0.5+ = near-uniform weights → scores converge to label mean
                            #          → gradients vanish → nothing trains.
regularization = 0.01       # L2 drift penalty (soft spring toward original position).
                            #   0.1+ = too strong, anchors can't move enough.
                            #   0.001 = very free movement (cosine constraint is the safety net).
batch_size = 8              # Examples per gradient update
train_knn_k = 20            # Soft top-K neighborhood size (match evaluator's knn_size)

# --- Cross-validation ---
kfold = 5                   # Folds for epoch selection (higher = more reliable, slower)

# --- Drift constraints ---
drift_limit = 0.15          # Max cosine drift from original (soft reference)
min_similarity = 0.85       # Hard floor: project back if below this

# --- Synthetic data quality ---
synthetic_diversity_threshold = 0.95  # Remove paraphrases above this cosine similarity

# --- Synthetic generation ---
diversity_temp = 0.8        # LLM creativity for edge cases (0.0-1.0)
adversarial_depth = high    # How tricky: low / medium / high
context_variety = true      # Force varied formats: email, code, chat, academic, etc.
```

**Output:**

```
NAME/
├── optimized_anchors_NAME.npz   # Updated embedding vectors
└── training_report_NAME.txt     # K-fold results + final accuracy
```

The evaluator automatically loads `optimized_anchors_NAME.npz` when present — no regeneration needed. Delete the `.npz` file to revert to original embeddings. Thresholds in `config_NAME.ini` are never modified by AMO training; to recalibrate after training, use `--auto-calibrate` with labeled data.

## Visualization (`--graph`)

Two-panel display:

**Left**: 2D projection using UMAP (preferred), t-SNE, or PCA fallback. Shows positive (blue), negative (black), proposition (red star). Includes **distance fidelity** (Spearman r between true cosine distances and projected 2D distances) — tells you how trustworthy the layout is. Fidelity > 0.8 means the 2D plot reliably represents high-dimensional relationships.

**Right**: Pairwise cosine similarity histograms (pos↔pos, pos↔neg, neg↔neg) with threshold lines. Shows class separation directly in the native embedding space with no dimensionality reduction artifacts.

## Evaluator Configuration (config_NAME.ini)

```ini
[thresholds]
match_threshold = 0.70
warning_threshold = 0.50
knn_size = 20
nli_retrieve_k = 40          # cosine pre-filter top 40, then NLI re-rank
nli_vote_k = 20
nli_fwd_weight = 0.7
nli_abstain_margin = 0.15
hybrid_pool_size = 40        # 20 cosine + 20 NLI = merged voters

[models]
embedding_model = all-mpnet-base-v2
nli_model = cross-encoder/nli-deberta-v3-large

[llm_judge]
provider = anthropic
model = claude-sonnet-4-20250514
api_key = YOUR_API_KEY_HERE

[options]
spellcheck = false
```

Note: AMO training parameters (`[training]` section) live in the generator's `config.ini`, not in `config_NAME.ini`. The evaluator config only contains inference-time settings.

## Tuning Guide

**Anchor count (`-n`)**: Auto-computed from `2 × total_categories`. With 12 static + 10 orthogonal axes = 22 categories, the minimum is 44. If you pass `-n 20`, the generator auto-adjusts upward and prints the reason. You can override upward (e.g., `-n 60`) but not below the minimum.

**Broad propositions** (many subtopics): The auto-computed `-n` already accounts for category count. Increase `nli_retrieve_k = 60-80` and `hybrid_pool_size = 80-100` for better coverage. Subtopic parity, boundary targeting, and hard mining handle distribution automatically.

**Narrow propositions**: Auto-computed `-n` still applies (category coverage matters regardless of subtopic count). Usually lands around 40-50.

**High false positives**: Run `--auto-calibrate` with labeled data. Boundary targeting and hard mining reduce FPs during generation; calibrated thresholds handle residual cases.

**High false negatives**: Increase `-n` beyond the auto-computed minimum. Check subtopic coverage in generation output — underrepresented subtopics are the usual cause.
