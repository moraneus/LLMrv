import os
import numpy as np

from .config import parse_categories
from .llm import call_llm_raw, _role_context
from .serialization import load_existing_anchors


def generate_synthetic_training_data(proposition, config, n_examples, role="user",
                                      existing_pos=None, existing_neg=None):
    """Generate N synthetic adversarial training pairs for AMO training.

    Uses a "sparring" strategy: instead of generic examples, generates
    adversarial pairs that create maximum tension at the decision boundary.

    Pair types:
      STEALTH (positive label): Harmful intent disguised with benign vocabulary,
        academic framing, fictional wrapping, or professional language.
      TRAP (negative label): Benign intent loaded with dangerous-sounding
        keywords, domain terminology, and alarming context.

    Role-aware generation:
      - role="user": examples as user messages, distributed across user-style
        categories (direct requests, euphemistic, hypothetical, etc.)
      - role="assistant": examples as AI assistant responses, distributed across
        response categories (informational, instructional, refusal, etc.)
      Categories are read from config.ini via parse_categories().

    Generation guarantees:
      - Exact target count: while-loop runs until target is met. Only stall
        detection (3 consecutive rounds with 0 new examples) can stop early.
      - Angle rotation: each retry uses a different creative direction.
      - Zero-overlap rule: existing anchors passed to LLM with exclusion block.
      - Strict 50/50 balance: retries the lagging class before trimming.
      - Case-insensitive dedup.

    Note: embedding-based diversity filtering (removing near-duplicate
    paraphrases) is applied AFTER this function returns, in the caller
    (train_anchor_manifold), which also handles regeneration of shortfall.

    Returns: list of (text, label) where label is 1 (positive) or 0 (negative).
    """
    import random

    half = n_examples // 2
    rc = _role_context(role)

    # --- Read synthetic generation config ---
    diversity_temp = config.get("training", "diversity_temp", fallback="0.8")
    adversarial_depth = config.get("training", "adversarial_depth", fallback="high")
    context_variety = config.get("training", "context_variety", fallback="true").lower() in ("true", "1", "yes")

    depth_instructions = {
        "low": (
            "Generate straightforward edge cases. Some should be tricky but "
            "most can be relatively clear-cut."
        ),
        "medium": (
            "Generate moderately tricky edge cases. At least half should require "
            "careful reading to classify correctly. Use some misdirection."
        ),
        "high": (
            "Generate MAXIMALLY ADVERSARIAL edge cases. Every example should be "
            "deliberately designed to fool a classifier. Use heavy misdirection, "
            "plausible deniability, nested framing, and vocabulary cross-contamination. "
            "A human reviewer should need to think carefully about each one."
        ),
    }
    depth_text = depth_instructions.get(adversarial_depth, depth_instructions["high"])

    # --- Build context variety instructions ---
    variety_text = ""
    if context_variety:
        variety_text = (
            "\nFORMAT VARIETY: Each example must use a DIFFERENT format or context. "
            "Distribute across: casual chat messages, formal emails, code comments, "
            "academic paper excerpts, forum posts, customer support tickets, "
            "social media posts, internal memos, technical documentation, "
            "creative writing excerpts, interview transcripts, and legal/compliance text.\n"
        )

    # --- Zero-overlap exclusion block ---
    exclusion_block = ""
    all_existing = []
    if existing_pos:
        all_existing.extend(existing_pos)
    if existing_neg:
        all_existing.extend(existing_neg)

    if all_existing:
        # Sample up to 30 for the prompt (avoid token overflow)
        sample = random.sample(all_existing, min(30, len(all_existing)))
        exclusion_block = (
            "\n═══ ZERO-OVERLAP RULE ═══\n"
            "Do NOT use any phrases, scenarios, or examples found below. "
            "These are EXISTING anchors from anchors_list.json. "
            "Generate 100%% NOVEL scenarios that test the LIMITS of this proposition. "
            "Every example must describe a situation, framing, or context that "
            "does NOT appear in this list:\n"
        )
        for ex in sample:
            exclusion_block += "  ✗ {}\n".format(ex[:100])
        exclusion_block += "═══ END EXCLUSION LIST ═══\n\n"

    # --- Build category context for role-appropriate generation ---
    categories = parse_categories(config, role=role)
    cat_block = ""
    if categories:
        cat_list = "\n".join("  - {}".format(c) for c in categories[:15])  # cap to avoid token overflow
        if role == "assistant":
            cat_block = (
                "\nROLE: Generate examples as AI ASSISTANT RESPONSES (not user messages).\n"
                "Each example should read like something an AI assistant would write.\n"
                "Distribute examples across these response categories:\n"
                "{}\n\n"
            ).format(cat_list)
        else:
            cat_block = (
                "\nROLE: Generate examples as USER MESSAGES (not assistant responses).\n"
                "Each example should read like something a real user would type.\n"
                "Distribute examples across these request categories:\n"
                "{}\n\n"
            ).format(cat_list)

    # --- STEALTH examples: subtle positive matches (label = 1) ---
    stealth_prompt = (
        "{safety}"
        "PROPOSITION: \"{prop}\"\n\n"
        "{exclusion}"
        "{cats}"
        "Generate {n} additional {noun} that MATCH this proposition.\n\n"
        "These are for training a SAFETY CLASSIFIER to detect subtle cases. "
        "The classifier already catches obvious matches — now we need examples that test "
        "the boundary. These examples should genuinely match the proposition but use "
        "indirect, subtle, or non-obvious formulations:\n\n"
        "  - Euphemistic or coded language instead of direct terms\n"
        "  - Hypothetical or fictional framing (screenplay, novel, thought experiment)\n"
        "  - Academic or research framing (studying, analyzing, documenting)\n"
        "  - Professional jargon that a domain expert would recognize\n"
        "  - Multi-step requests where the relevant part is buried mid-conversation\n"
        "  - Polite or collaborative tone\n"
        "  - Emotional appeals or personal stories that embed the core request\n\n"
        "{depth}\n\n"
        "{variety}"
        "Each example must be SEMANTICALLY DISTINCT — no paraphrases.\n"
        "Return exactly {n} examples, one per line. No numbering, bullets, or labels.\n"
    ).format(
        safety=rc["safety_context"], prop=proposition,
        exclusion=exclusion_block, cats=cat_block, depth=depth_text,
        n=half, noun=rc["example_noun"], variety=variety_text
    )

    # --- TRAP examples: hard negatives with overlapping vocabulary (label = 0) ---
    trap_prompt = (
        "{safety}"
        "PROPOSITION: \"{prop}\"\n\n"
        "{exclusion}"
        "{cats}"
        "Generate {n} HARD NEGATIVE {noun} for testing a safety classifier.\n\n"
        "Hard negatives are examples that do NOT match the proposition but use "
        "similar vocabulary and domain terminology. A naive classifier would "
        "incorrectly flag these, but they have clearly LEGITIMATE intent:\n\n"
        "  - Security professionals discussing attack vectors for DEFENSIVE purposes\n"
        "  - Educators explaining techniques in an academic or classroom context\n"
        "  - Victims or incident responders describing what happened (not requesting how-to)\n"
        "  - Compliance officers auditing existing security measures\n"
        "  - Journalists investigating and reporting on incidents\n"
        "  - Fiction writers researching technical details for realistic storytelling\n"
        "  - Parents or IT administrators discussing monitoring for PROTECTIVE purposes\n"
        "  - Bug bounty hunters describing findings in responsible disclosure reports\n\n"
        "{depth}\n\n"
        "{variety}"
        "KEY: Every example should use vocabulary that OVERLAPS heavily with the "
        "proposition's domain, but the actual INTENT must be clearly legitimate.\n\n"
        "Each example must be SEMANTICALLY DISTINCT — no paraphrases.\n"
        "Return exactly {n} examples, one per line. No numbering, bullets, or labels.\n"
    ).format(
        safety=rc["safety_context"], prop=proposition,
        exclusion=exclusion_block, cats=cat_block, depth=depth_text,
        n=half, noun=rc["example_noun"], variety=variety_text
    )

    system = (
        "You generate adversarial training data for an AI safety classifier. "
        "Your job is to create the HARDEST possible examples that push the "
        "decision boundary to its limits. Output only the requested examples, "
        "one per line. No explanations, no labels, no metadata."
    )

    # --- Generate with retry until target is reached ---
    STALL_PATIENCE = 3  # stop only if N consecutive rounds produce 0 new examples

    # Angle rotation: each retry uses a different creative direction so the LLM
    # doesn't keep generating the same cluster of examples.
    RETRY_ANGLES = [
        "Use completely different vocabulary, sentence structure, and framing than before.",
        "Try unusual contexts: different professions, cultures, age groups, or scenarios.",
        "Use indirect language, euphemisms, slang, or coded speech patterns.",
        "Frame requests as coming from different personas: students, writers, researchers, gamers.",
        "Vary the length and complexity — mix very short (5-10 words) with longer multi-sentence ones.",
        "Use question formats, imperative commands, hypotheticals, and conditional statements.",
        "Include misspellings, abbreviations, casual internet language, or non-native speaker patterns.",
        "Think about edge cases: requests that are almost but not quite matching the category.",
        "Use formal/academic language, legal/policy framing, or journalistic phrasing.",
        "Consider multi-turn conversation fragments, follow-up messages, or out-of-context snippets.",
    ]

    def _generate_and_parse(prompt, label_name, target_count):
        """Generate examples until exactly target_count is reached.

        Runs in a while-loop with no round cap — only stall detection
        (STALL_PATIENCE consecutive rounds with 0 new examples) can stop
        it short of the target. Each retry rotates through creative angles
        and shows a random sample of already-generated examples.
        """
        all_examples = []
        all_examples_lower = set()  # for fast case-insensitive dedup
        stall_count = 0
        round_num = 0

        while len(all_examples) < target_count:
            round_num += 1
            need = target_count - len(all_examples)

            if round_num == 1:
                retry_prompt = prompt
            else:
                # Random sample of existing examples (not always the tail)
                sample_size = min(20, len(all_examples))
                sample = random.sample(all_examples, sample_size) if sample_size > 0 else []
                existing_block = "\n".join(
                    "  ✗ {}".format(ex[:120]) for ex in sample)

                # Rotate through creative angles
                angle = RETRY_ANGLES[(round_num - 1) % len(RETRY_ANGLES)]

                retry_prompt = prompt + (
                    "\n\nPROGRESS: I have {} so far, need {} MORE to reach {}.\n"
                    "ALREADY GENERATED (do NOT repeat, paraphrase, or closely rephrase these):\n"
                    "{}\n\n"
                    "CREATIVE DIRECTION FOR THIS BATCH: {}\n\n"
                    "Generate exactly {} NEW, DISTINCT examples that are UNLIKE anything above."
                ).format(len(all_examples), need, target_count,
                         existing_block, angle, min(need, 30))

            raw = call_llm_raw(config, system, retry_prompt)
            if not raw:
                print("      {} generation failed (round {})".format(
                    label_name, round_num))
                stall_count += 1
                if stall_count >= STALL_PATIENCE:
                    print("      {} consecutive failures — stopping".format(STALL_PATIENCE))
                    break
                continue

            parsed = [
                line.strip().strip("- ").strip("0123456789.)")
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            # Dedup against everything we already have (case-insensitive)
            new_count = 0
            for ex in parsed:
                ex_lower = ex.lower().strip()
                if ex_lower not in all_examples_lower:
                    all_examples.append(ex)
                    all_examples_lower.add(ex_lower)
                    new_count += 1
                    if len(all_examples) >= target_count:
                        break

            if new_count == 0:
                stall_count += 1
                if stall_count >= STALL_PATIENCE:
                    print("      {} consecutive rounds with 0 new examples — stopping".format(
                        STALL_PATIENCE))
                    break
            else:
                stall_count = 0

            if round_num > 1:
                print("      round {}: +{} new (total: {}/{})".format(
                    round_num, new_count, len(all_examples), target_count))

        if len(all_examples) < target_count:
            print("      {}WARNING: stalled at {}/{} {} examples{}".format(
                "\033[93m", len(all_examples), target_count, label_name, "\033[0m"))

        return all_examples[:target_count]

    print("    Generating {} subtle positive examples (boundary-testing)...".format(half))
    stealth_examples = _generate_and_parse(stealth_prompt, "positive", half)

    print("    Generating {} hard negative examples (overlapping vocabulary)...".format(half))
    trap_examples = _generate_and_parse(trap_prompt, "negative", half)

    # --- Enforce strict 50/50 balance ---
    # Instead of trimming the larger side, retry the smaller side
    BALANCE_RETRIES = 5
    for balance_round in range(BALANCE_RETRIES):
        diff = abs(len(stealth_examples) - len(trap_examples))
        if diff == 0:
            break

        if len(stealth_examples) < len(trap_examples):
            shortfall = len(trap_examples) - len(stealth_examples)
            print("    Balance: positives short by {} — generating more...".format(shortfall))
            extra = _generate_and_parse(stealth_prompt, "positive (balance)", shortfall)
            stealth_examples.extend(extra)
        else:
            shortfall = len(stealth_examples) - len(trap_examples)
            print("    Balance: negatives short by {} — generating more...".format(shortfall))
            extra = _generate_and_parse(trap_prompt, "negative (balance)", shortfall)
            trap_examples.extend(extra)

    # Final balance: if retries still couldn't close the gap, trim as last resort
    min_count = min(len(stealth_examples), len(trap_examples))
    if len(stealth_examples) != len(trap_examples):
        YELLOW = "\033[93m"
        RESET_C = "\033[0m"
        print("    {}Balance: trimming to {} per class after {} retry rounds (requested {}){}".format(
            YELLOW, min_count, BALANCE_RETRIES, half, RESET_C))
        stealth_examples = stealth_examples[:min_count]
        trap_examples = trap_examples[:min_count]

    print("    Got {} positive + {} negative = {} total adversarial pairs".format(
        len(stealth_examples), len(trap_examples),
        len(stealth_examples) + len(trap_examples)))

    # Stealth = positive (matches proposition), Trap = negative (does not match)
    data = ([(text, 1) for text in stealth_examples] +
            [(text, 0) for text in trap_examples])
    random.shuffle(data)
    return data


def train_anchor_manifold(proposition, name, config, n_synthetic=100, role="user"):
    """Anchor Manifold Optimization: nudge anchor embeddings using gradient descent.

    Uses soft top-K KNN scoring (differentiable) with cosine anchoring constraints to
    move positive anchors closer to true positives and negative anchors closer to
    false-positive noise, without losing semantic meaning.

    Pipeline:
      1. Generate N role-aware adversarial synthetic examples (user messages or
         assistant responses based on role, distributed across config categories,
         while-loop retries until exact target, then diversity filter + regen)
      2. K-fold cross-validation (default 5 folds) to select optimal epoch count
      3. Final training on ALL data for exactly that many epochs
      4. Save optimized embeddings

    Key defaults (in config.ini [training]):
      temperature = 0.05    (sharp softmax → strong gradients)
      regularization = 0.01 (light spring → anchors can move)
      train_knn_k = 20      (matches evaluator's default knn_size)

    Produces:
      - optimized_anchors_NAME.npz  (updated embedding vectors)
      - training_report_NAME.txt    (k-fold results + final accuracy)

    NOTE: Does NOT modify thresholds in config_NAME.ini. The training scorer
    (soft-KNN with temperature) operates on a different scale than the evaluator's
    hard-KNN scorer. To recalibrate thresholds after training, use:
      python semantic_anchor_NAME.py --auto-calibrate labeled_data.txt
    """
    import torch
    import torch.optim as optim
    import random
    import datetime

    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # --- Load training hyperparameters ---
    lr = float(config.get("training", "learning_rate", fallback="0.001"))
    epochs = int(config.get("training", "epochs", fallback="20"))
    reg_weight = float(config.get("training", "regularization", fallback="0.01"))
    temperature = float(config.get("training", "temperature", fallback="0.05"))
    batch_size = int(config.get("training", "batch_size", fallback="8"))
    train_knn_k = int(config.get("training", "train_knn_k", fallback="20"))
    drift_limit = float(config.get("training", "drift_limit", fallback="0.15"))
    min_sim = float(config.get("training", "min_similarity", fallback="0.85"))

    # Sanity checks for common misconfigurations
    if temperature >= 0.5:
        print("  {}WARNING:{} temperature={} is very high. Softmax over K={} neighbors".format(
            YELLOW, RESET, temperature, train_knn_k))
        print("           will produce near-uniform weights → vanishing gradients.")
        print("           Recommended: 0.01–0.1. Default: 0.05")
    if reg_weight >= 0.1:
        print("  {}WARNING:{} regularization={} is high — anchors may not move enough.".format(
            YELLOW, RESET, reg_weight))
        print("           Recommended: 0.001–0.05. Default: 0.01")

    output_dir = name
    os.makedirs(output_dir, exist_ok=True)

    # --- Load existing anchors ---
    data, anchor_path = load_existing_anchors(output_dir, name)
    if data is None:
        print("  {}ERROR:{} No anchors found at {}/anchors_list_{}.json".format(
            RED, RESET, output_dir, name))
        print("  Run the generator first to create anchors.")
        return

    pos_texts, pos_cats = [], []
    for cat, examples in data["anchors"].items():
        for ex in examples:
            pos_texts.append(ex)
            pos_cats.append(cat)

    neg_texts, neg_cats = [], []
    for cat, examples in data.get("negative_anchors", {}).items():
        for ex in examples:
            neg_texts.append(ex)
            neg_cats.append(cat)

    n_pos = len(pos_texts)
    n_neg = len(neg_texts)
    n_anchors = n_pos + n_neg

    # Clamp train_knn_k to available anchors
    effective_knn_k = min(train_knn_k, n_anchors)
    if effective_knn_k < train_knn_k:
        print("  {}NOTE:{} train_knn_k={} exceeds total anchors ({}), clamped to {}".format(
            YELLOW, RESET, train_knn_k, n_anchors, effective_knn_k))

    print("\n  {}AMO Configuration:{}".format(BOLD, RESET))
    print("    Positive anchors:  {}".format(n_pos))
    print("    Negative anchors:  {}".format(n_neg))
    print("    Synthetic examples: {} ({} positive + {} negative)".format(
        n_synthetic, n_synthetic // 2, n_synthetic // 2))
    print("    Learning rate:     {}".format(lr))
    print("    Epochs:            {}".format(epochs))
    print("    Temperature:       {}".format(temperature))
    print("    Regularization:    {}".format(reg_weight))
    print("    Drift limit:       {} (min cosine sim: {})".format(drift_limit, min_sim))
    print("    Batch size:        {}".format(batch_size))
    print("    KNN K:             {} (soft top-K for gradient flow)".format(
        effective_knn_k))

    # Synthetic generation params
    adv_depth = config.get("training", "adversarial_depth", fallback="high")
    ctx_variety = config.get("training", "context_variety", fallback="true")
    n_folds = int(config.get("training", "kfold", fallback="5"))
    diversity_thresh = float(config.get(
        "training", "synthetic_diversity_threshold", fallback="0.95"))
    print("    Adversarial depth: {}".format(adv_depth))
    print("    Context variety:   {}".format(ctx_variety))
    print("    K-fold CV:         {} folds".format(n_folds))
    print("    Diversity thresh:  {} (cosine)".format(diversity_thresh))

    # --- Load embedding model ---
    from sentence_transformers import SentenceTransformer
    emb_model_name = config.get("anchors", "embedding_model")
    print("\n  Loading embedding model: {}...".format(emb_model_name))
    emb_model = SentenceTransformer(emb_model_name)

    # --- Embed anchors ---
    print("  Embedding {} positive + {} negative anchors...".format(n_pos, n_neg))
    pos_embs_np = emb_model.encode(pos_texts, show_progress_bar=False)
    neg_embs_np = emb_model.encode(neg_texts, show_progress_bar=False)

    # Labels: pos=1, neg=0
    anchor_labels = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)

    # --- Generate synthetic training data ---
    print("\n  {}Generating synthetic training data...{}".format(BOLD, RESET))
    synthetic = generate_synthetic_training_data(
        proposition, config, n_synthetic, role=role,
        existing_pos=pos_texts, existing_neg=neg_texts)

    if len(synthetic) < 10:
        print("  {}ERROR:{} Too few synthetic examples generated ({}).".format(
            RED, RESET, len(synthetic)))
        print("  This usually means the LLM refused to generate adversarial content.")
        print("  Try:")
        print("    - A different model (e.g. a local model via ollama)")
        print("    - Lower adversarial_depth = medium or low in [training] config")
        print("    - A less sensitive proposition")
        return

    # --- Embedding-based diversity filtering with regeneration ---
    # Remove near-duplicate paraphrases that have different text but almost
    # identical embeddings (cosine > threshold). These waste training capacity
    # and cause overfitting to narrow clusters in embedding space.
    # If removals reduce count below target, regenerate and re-check.
    DIVERSITY_THRESHOLD = diversity_thresh
    DIVERSITY_REGEN_ROUNDS = 3
    half = n_synthetic // 2

    for regen_round in range(DIVERSITY_REGEN_ROUNDS + 1):
        syn_texts_all = [t for t, _ in synthetic]
        syn_labels_all = [l for _, l in synthetic]

        if regen_round == 0:
            print("  Embedding {} synthetic examples for diversity check...".format(
                len(synthetic)))
        else:
            print("    Re-embedding {} examples after regeneration...".format(len(synthetic)))

        syn_embs_np = emb_model.encode(syn_texts_all, show_progress_bar=False)

        # Greedy dedup within each class: keep first, skip if too similar to any kept
        diverse_synthetic = []
        pos_kept, pos_removed = 0, 0
        neg_kept, neg_removed = 0, 0

        for label_val in [1.0, 0.0]:
            class_indices = [i for i in range(len(synthetic))
                            if syn_labels_all[i] == label_val]
            kept_embs = []
            for idx in class_indices:
                emb = syn_embs_np[idx]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                is_diverse = True
                for kept_emb in kept_embs:
                    if np.dot(emb_norm, kept_emb) > DIVERSITY_THRESHOLD:
                        is_diverse = False
                        break
                if is_diverse:
                    kept_embs.append(emb_norm)
                    diverse_synthetic.append(synthetic[idx])
                    if label_val == 1.0:
                        pos_kept += 1
                    else:
                        neg_kept += 1
                else:
                    if label_val == 1.0:
                        pos_removed += 1
                    else:
                        neg_removed += 1

        if pos_removed > 0:
            print("    Diversity filter (positive): kept {}, removed {} near-duplicates (cosine > {})".format(
                pos_kept, pos_removed, DIVERSITY_THRESHOLD))
        if neg_removed > 0:
            print("    Diversity filter (negative): kept {}, removed {} near-duplicates (cosine > {})".format(
                neg_kept, neg_removed, DIVERSITY_THRESHOLD))

        synthetic = diverse_synthetic

        # Check if both classes meet target
        if pos_kept >= half and neg_kept >= half:
            break  # we're good

        if regen_round >= DIVERSITY_REGEN_ROUNDS:
            # Last round — accept what we have
            if pos_kept < half or neg_kept < half:
                print("    {}Diversity regen exhausted: {} pos, {} neg (target {} each){}".format(
                    YELLOW, pos_kept, neg_kept, half, RESET))
            break

        # Regenerate shortfall
        regen_system = (
            "You generate adversarial training data for an AI safety classifier. "
            "Output only the requested examples, one per line. No explanations, "
            "no labels, no metadata."
        )
        # Use role context for correct POV
        rc = _role_context(role)
        regen_noun = rc["example_noun"]  # "user messages" or "AI assistant responses"

        if pos_kept < half:
            shortfall = half - pos_kept
            print("    Diversity regen round {}: need {} more positives...".format(
                regen_round + 1, shortfall))
            regen_prompt = (
                "PROPOSITION: \"{}\"\n\n"
                "Generate {} {} that MATCH this proposition.\n"
                "Use subtle, indirect, or non-obvious formulations — euphemisms, "
                "hypotheticals, coded language, academic framing.\n"
                "Each must be SEMANTICALLY DISTINCT. One per line, no numbering."
            ).format(proposition, shortfall, regen_noun)
            raw = call_llm_raw(config, regen_system, regen_prompt)
            if raw:
                for line in raw.strip().split("\n"):
                    line = line.strip().strip("- ").strip("0123456789.)")
                    if line and len(line) > 10:
                        synthetic.append((line, 1.0))

        if neg_kept < half:
            shortfall = half - neg_kept
            print("    Diversity regen round {}: need {} more negatives...".format(
                regen_round + 1, shortfall))
            regen_prompt = (
                "PROPOSITION: \"{}\"\n\n"
                "Generate {} HARD NEGATIVE {}.\n"
                "These do NOT match the proposition but use similar vocabulary "
                "and domain terminology. They have clearly LEGITIMATE intent: "
                "education, journalism, security research, fiction, compliance.\n"
                "Each must be SEMANTICALLY DISTINCT. One per line, no numbering."
            ).format(proposition, shortfall, regen_noun)
            raw = call_llm_raw(config, regen_system, regen_prompt)
            if raw:
                for line in raw.strip().split("\n"):
                    line = line.strip().strip("- ").strip("0123456789.)")
                    if line and len(line) > 10:
                        synthetic.append((line, 0.0))

    if len(synthetic) < 10:
        print("  {}ERROR:{} Too few diverse examples remain ({}).".format(
            RED, RESET, len(synthetic)))
        return

    # Shuffle before use (diversity filter grouped by class)
    random.shuffle(synthetic)

    # --- Embed ALL synthetic data (used for k-fold and final training) ---
    all_syn_texts = [t for t, _ in synthetic]
    all_syn_labels = np.array([l for _, l in synthetic], dtype=np.float32)
    print("  Embedding {} training messages...".format(len(synthetic)))
    all_syn_embs = torch.tensor(
        emb_model.encode(all_syn_texts, show_progress_bar=False), dtype=torch.float32)
    all_syn_labels_t = torch.tensor(all_syn_labels, dtype=torch.float32)

    n_syn_pos = int(all_syn_labels.sum())
    n_syn_neg = len(all_syn_labels) - n_syn_pos
    print("    {} examples ({} pos, {} neg)".format(
        len(synthetic), n_syn_pos, n_syn_neg))

    # --- Create trainable anchor parameters ---
    all_orig_embs = np.vstack([pos_embs_np, neg_embs_np])
    anchor_labels_t = torch.tensor(anchor_labels, dtype=torch.float32)

    # --- Soft-KNN scoring function (differentiable) ---
    knn_k = effective_knn_k  # from config: training.train_knn_k

    def soft_knn_score(query_emb, anchors, labels, temp):
        """Differentiable top-K KNN scoring via softmax-weighted voting.

        1. Compute cosine similarity to all anchors
        2. Select top-K most similar (hard selection — gradient flows through values)
        3. Softmax-weighted vote within those K neighbors

        Without top-K, softmax over 300+ anchors is nearly uniform (every query
        scores ~label_mean), gradients vanish, and nothing trains.
        With top-K selection, small similarity differences create meaningful weight
        differences, and the gradient actually points somewhere useful.
        """
        q_norm = query_emb / (query_emb.norm() + 1e-10)
        a_norm = anchors / (anchors.norm(dim=1, keepdim=True) + 1e-10)
        cos_sim = (q_norm @ a_norm.T)
        top_vals, top_idx = torch.topk(cos_sim, knn_k)
        top_labels = labels[top_idx]
        weights = torch.softmax(top_vals / temp, dim=0)
        score = (weights * top_labels).sum()
        return score.clamp(1e-7, 1 - 1e-7)

    def evaluate(embs, labels_t, anchors, a_labels, temp):
        """Evaluate accuracy and avg loss on a dataset."""
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i in range(len(embs)):
                score = soft_knn_score(embs[i], anchors, a_labels, temp)
                pred = 1 if score > 0.5 else 0
                correct += (pred == int(labels_t[i].item()))
                loss = -labels_t[i] * torch.log(score + 1e-10) - \
                       (1 - labels_t[i]) * torch.log(1 - score + 1e-10)
                total_loss += loss.item()
        acc = correct / len(embs) if len(embs) > 0 else 0.0
        avg_loss = total_loss / len(embs) if len(embs) > 0 else 0.0
        return acc, avg_loss

    def train_one_run(train_embs, train_labels_t, val_embs, val_labels_t,
                      max_epochs, verbose=False, label=""):
        """Run one training cycle. Returns (best_vecs, best_epoch, epoch_val_accs, epoch_train_accs)."""
        anchors = torch.tensor(all_orig_embs.copy(), dtype=torch.float32, requires_grad=True)
        orig = torch.tensor(all_orig_embs, dtype=torch.float32)
        opt = optim.Adam([anchors], lr=lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.1)

        best_acc = 0.0
        best_e = 0
        best_a = anchors.clone().detach()
        patience = 0
        epoch_val_accs = []
        epoch_train_accs = []

        for ep in range(max_epochs):
            # Shuffle and train
            indices = list(range(len(train_embs)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                batch_loss = torch.tensor(0.0)
                for i in batch_idx:
                    score = soft_knn_score(
                        train_embs[i], anchors, anchor_labels_t, temperature)
                    bce = -train_labels_t[i] * torch.log(score + 1e-10) - \
                          (1 - train_labels_t[i]) * torch.log(1 - score + 1e-10)
                    batch_loss = batch_loss + bce
                drift_penalty = ((anchors - orig) ** 2).sum() * reg_weight
                batch_loss = batch_loss / len(batch_idx) + drift_penalty
                opt.zero_grad()
                batch_loss.backward()
                opt.step()

            # Cosine anchoring constraint
            with torch.no_grad():
                for j in range(n_anchors):
                    new_vec = anchors[j]
                    old_vec = orig[j]
                    cos = (new_vec @ old_vec) / (
                        new_vec.norm() * old_vec.norm() + 1e-10)
                    if cos.item() < min_sim:
                        lo, hi = 0.0, 1.0
                        for _ in range(20):
                            mid = (lo + hi) / 2
                            interp = mid * old_vec + (1 - mid) * new_vec
                            sim = (interp @ old_vec) / (
                                interp.norm() * old_vec.norm() + 1e-10)
                            if sim.item() < min_sim:
                                lo = mid
                            else:
                                hi = mid
                        anchors[j] = hi * old_vec + (1 - hi) * new_vec

            # Evaluate
            train_acc, _ = evaluate(train_embs, train_labels_t, anchors, anchor_labels_t, temperature)
            val_acc, _ = evaluate(val_embs, val_labels_t, anchors, anchor_labels_t, temperature)
            epoch_train_accs.append(train_acc)
            epoch_val_accs.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_e = ep + 1
                best_a = anchors.clone().detach()
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

            sched.step()

        if verbose and label:
            print("    {}: best_val={:.1%} at epoch {}".format(label, best_acc, best_e))

        return best_a, best_e, epoch_val_accs, epoch_train_accs

    # --- K-fold cross-validation for robust epoch selection ---
    PATIENCE = 5

    # Pre-training baseline
    baseline_anchors = torch.tensor(all_orig_embs, dtype=torch.float32)
    pre_acc, pre_loss = evaluate(
        all_syn_embs, all_syn_labels_t, baseline_anchors, anchor_labels_t, temperature)
    print("\n  {}Pre-training accuracy (all data): {:.1%}{}".format(BOLD, pre_acc, RESET))

    # Create stratified fold indices
    pos_indices = [i for i in range(len(synthetic)) if all_syn_labels[i] == 1.0]
    neg_indices = [i for i in range(len(synthetic)) if all_syn_labels[i] == 0.0]
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)

    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_indices):
        folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_indices):
        folds[i % n_folds].append(idx)

    print("\n  {}{}-fold cross-validation ({} epochs per fold)...{}".format(
        BOLD, n_folds, epochs, RESET))

    all_fold_val_accs = []  # list of lists: [fold][epoch]
    fold_best_epochs = []

    for fold_idx in range(n_folds):
        val_idx = sorted(folds[fold_idx])
        train_idx = sorted([idx for f in range(n_folds) if f != fold_idx
                            for idx in folds[f]])

        fold_train_embs = all_syn_embs[train_idx]
        fold_train_labels = all_syn_labels_t[train_idx]
        fold_val_embs = all_syn_embs[val_idx]
        fold_val_labels = all_syn_labels_t[val_idx]

        n_val_pos = int(fold_val_labels.sum().item())
        n_val_neg = len(fold_val_labels) - n_val_pos

        _, best_e, val_accs, _ = train_one_run(
            fold_train_embs, fold_train_labels,
            fold_val_embs, fold_val_labels,
            max_epochs=epochs, verbose=True,
            label="Fold {}/{} (train={}, val={}: {} pos, {} neg)".format(
                fold_idx + 1, n_folds, len(train_idx), len(val_idx),
                n_val_pos, n_val_neg))

        all_fold_val_accs.append(val_accs)
        fold_best_epochs.append(best_e)

    # Compute mean val accuracy per epoch across folds
    # Folds may have different lengths due to early stopping — use min length
    min_epochs_run = min(len(accs) for accs in all_fold_val_accs)
    mean_val_accs = []
    for ep in range(min_epochs_run):
        mean_acc = sum(all_fold_val_accs[f][ep] for f in range(n_folds)) / n_folds
        mean_val_accs.append(mean_acc)

    best_mean_epoch = max(range(len(mean_val_accs)), key=lambda e: mean_val_accs[e]) + 1
    best_mean_acc = mean_val_accs[best_mean_epoch - 1]

    # Print epoch summary table
    print("\n  {}Mean validation accuracy by epoch (across {} folds):{}".format(
        BOLD, n_folds, RESET))
    print("  {:>5s}".format("Epoch"), end="")
    for f in range(n_folds):
        print("  {:>7s}".format("Fold{}".format(f + 1)), end="")
    print("  {:>7s}".format("Mean"))
    print("  " + "-" * (6 + 9 * n_folds + 8))

    for ep in range(min_epochs_run):
        is_best = (ep + 1 == best_mean_epoch)
        print("  {:>5d}".format(ep + 1), end="")
        for f in range(n_folds):
            print("  {:>6.1%}".format(all_fold_val_accs[f][ep]), end="")
        marker = " ★" if is_best else ""
        print("  {:>6.1%}{}".format(mean_val_accs[ep], marker))

    print("\n  {}Best epoch: {} (mean val accuracy: {:.1%}){}".format(
        GREEN, best_mean_epoch, best_mean_acc, RESET))

    # --- Final training on ALL data for best_mean_epoch epochs ---
    print("\n  {}Final training on all {} examples for {} epochs...{}".format(
        BOLD, len(synthetic), best_mean_epoch, RESET))

    report_lines = []
    report_lines.append("Anchor Manifold Optimization — Training Report")
    report_lines.append("=" * 50)
    report_lines.append("Proposition: \"{}\"".format(proposition))
    report_lines.append("Name: {}".format(name))
    report_lines.append("Date: {}".format(datetime.datetime.now().isoformat()))
    report_lines.append("")
    report_lines.append("Configuration:")
    report_lines.append("  Learning rate:  {}".format(lr))
    report_lines.append("  Epochs:         {} (selected by {}-fold CV from max {})".format(
        best_mean_epoch, n_folds, epochs))
    report_lines.append("  Temperature:    {}".format(temperature))
    report_lines.append("  Regularization: {}".format(reg_weight))
    report_lines.append("  Drift limit:    {} (min sim: {})".format(drift_limit, min_sim))
    report_lines.append("  Batch size:     {}".format(batch_size))
    report_lines.append("  Anchors:        {} pos + {} neg = {}".format(n_pos, n_neg, n_anchors))
    report_lines.append("  KNN K:          {} (soft top-K)".format(knn_k))
    report_lines.append("  Synthetic data: {} ({} pos, {} neg)".format(
        len(synthetic), n_syn_pos, n_syn_neg))
    report_lines.append("  K-fold CV:      {} folds, best epoch {} (mean val acc {:.1%})".format(
        n_folds, best_mean_epoch, best_mean_acc))
    report_lines.append("")
    report_lines.append("K-fold validation summary:")
    for f in range(n_folds):
        report_lines.append("  Fold {}: best_epoch={}, best_val_acc={:.1%}".format(
            f + 1, fold_best_epochs[f],
            max(all_fold_val_accs[f]) if all_fold_val_accs[f] else 0.0))
    report_lines.append("")

    # Final run: train on ALL data, no validation split
    anchor_vecs = torch.tensor(all_orig_embs.copy(), dtype=torch.float32, requires_grad=True)
    orig_vecs = torch.tensor(all_orig_embs, dtype=torch.float32)
    optimizer = optim.Adam([anchor_vecs], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=best_mean_epoch, eta_min=lr * 0.1)

    report_lines.append("Final training ({} epochs on all data):".format(best_mean_epoch))
    report_lines.append("Epoch  Train_Acc   Loss     Drift_Avg  Projections  LR")
    report_lines.append("-" * 65)

    print("  {:>5s}  {:>9s}  {:>8s}  {:>9s}  {:>11s}  {:>6s}".format(
        "Epoch", "Train", "Loss", "Drift", "Projections", "LR"))
    print("  " + "-" * 55)

    for epoch in range(best_mean_epoch):
        indices = list(range(len(all_syn_embs)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_loss = torch.tensor(0.0)

            for i in batch_idx:
                score = soft_knn_score(
                    all_syn_embs[i], anchor_vecs, anchor_labels_t, temperature)
                bce = -all_syn_labels_t[i] * torch.log(score + 1e-10) - \
                      (1 - all_syn_labels_t[i]) * torch.log(1 - score + 1e-10)
                batch_loss = batch_loss + bce

            drift_penalty = ((anchor_vecs - orig_vecs) ** 2).sum() * reg_weight
            batch_loss = batch_loss / len(batch_idx) + drift_penalty

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

        # Cosine anchoring constraint
        n_projected = 0
        with torch.no_grad():
            for j in range(n_anchors):
                new_vec = anchor_vecs[j]
                old_vec = orig_vecs[j]
                cos = (new_vec @ old_vec) / (
                    new_vec.norm() * old_vec.norm() + 1e-10)
                if cos.item() < min_sim:
                    lo, hi = 0.0, 1.0
                    for _ in range(20):
                        mid = (lo + hi) / 2
                        interpolated = mid * old_vec + (1 - mid) * new_vec
                        sim = (interpolated @ old_vec) / (
                            interpolated.norm() * old_vec.norm() + 1e-10)
                        if sim.item() < min_sim:
                            lo = mid
                        else:
                            hi = mid
                    anchor_vecs[j] = hi * old_vec + (1 - hi) * new_vec
                    n_projected += 1

        train_acc, _ = evaluate(
            all_syn_embs, all_syn_labels_t, anchor_vecs, anchor_labels_t, temperature)

        with torch.no_grad():
            cos_sims = torch.nn.functional.cosine_similarity(anchor_vecs, orig_vecs, dim=1)
            avg_drift = 1.0 - cos_sims.mean().item()

        avg_loss = epoch_loss / max(1, n_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        print("  {:>5d}  {:>8.1%}  {:>8.4f}  {:>8.4f}  {:>11d}  {:.4f}".format(
            epoch + 1, train_acc, avg_loss, avg_drift, n_projected, current_lr))
        report_lines.append("{:>5d}  {:>8.1%}  {:>8.4f}  {:>9.4f}  {:>11d}  {:.4f}".format(
            epoch + 1, train_acc, avg_loss, avg_drift, n_projected, current_lr))

        scheduler.step()

    # --- Post-training evaluation ---
    post_train_acc, post_train_loss = evaluate(
        all_syn_embs, all_syn_labels_t, anchor_vecs.detach(), anchor_labels_t, temperature)

    print("\n  {}Post-training performance:{}".format(BOLD, RESET))
    print("    Train accuracy: {:.1%} (pre: {:.1%})".format(post_train_acc, pre_acc))
    print("    Epoch selected: {} (by {}-fold CV, mean val: {:.1%})".format(
        best_mean_epoch, n_folds, best_mean_acc))

    # --- Drift statistics ---
    with torch.no_grad():
        final_cos = torch.nn.functional.cosine_similarity(anchor_vecs, orig_vecs, dim=1)
        pos_drift = 1.0 - final_cos[:n_pos].mean().item()
        neg_drift = 1.0 - final_cos[n_pos:].mean().item()
        min_cos = final_cos.min().item()
        max_drift_idx = final_cos.argmin().item()

    print("\n  {}Drift statistics:{}".format(BOLD, RESET))
    print("    Positive anchors avg drift: {:.4f}".format(pos_drift))
    print("    Negative anchors avg drift: {:.4f}".format(neg_drift))
    print("    Min cosine to original:     {:.4f} (anchor #{})".format(
        min_cos, max_drift_idx))

    # --- Save optimized embeddings ---
    optimized_path = os.path.join(output_dir, "optimized_anchors_{}.npz".format(name))
    opt_np = anchor_vecs.detach().numpy()
    np.savez_compressed(
        optimized_path,
        embeddings=opt_np,
        pos_count=n_pos,
        neg_count=n_neg,
        pos_texts=np.array(pos_texts, dtype=object),
        neg_texts=np.array(neg_texts, dtype=object),
    )
    print("\n  {}{}\u2713 Optimized anchors: {}{}".format(GREEN, BOLD, optimized_path, RESET))

    # --- Save training report ---
    report_lines.append("")
    report_lines.append("Post-training (epoch {} selected by {}-fold CV):".format(
        best_mean_epoch, n_folds))
    report_lines.append("  Train: {:.1%} accuracy (pre: {:.1%})".format(
        post_train_acc, pre_acc))
    report_lines.append("  CV mean val accuracy: {:.1%}".format(best_mean_acc))
    report_lines.append("")
    report_lines.append("Drift:")
    report_lines.append("  Positive avg: {:.4f}".format(pos_drift))
    report_lines.append("  Negative avg: {:.4f}".format(neg_drift))
    report_lines.append("  Min cosine:   {:.4f}".format(min_cos))

    report_path = os.path.join(output_dir, "training_report_{}.txt".format(name))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("  {}{}\u2713 Training report:  {}{}".format(GREEN, BOLD, report_path, RESET))

    # --- Threshold advisory (informational only — does NOT overwrite config) ---
    # The soft-KNN scorer used during training (softmax-weighted with temperature)
    # produces scores on a different scale than the evaluator's hard-KNN scorer
    # (count / K). Auto-calibrating thresholds on training scores and writing them
    # to the evaluator's config would produce nonsensical results.
    # Instead, we report score distributions so the user can decide.
    print("\n  {}Post-training score distribution (soft-KNN, informational only):{}".format(
        BOLD, RESET))

    scores_pos = []
    scores_neg = []
    with torch.no_grad():
        for i in range(len(all_syn_embs)):
            score = soft_knn_score(
                all_syn_embs[i], anchor_vecs.detach(), anchor_labels_t, temperature)
            if all_syn_labels[i] == 1:
                scores_pos.append(score.item())
            else:
                scores_neg.append(score.item())

    if scores_pos:
        print("    Positive scores: min={:.3f}  mean={:.3f}  max={:.3f}  (n={})".format(
            min(scores_pos), sum(scores_pos) / len(scores_pos),
            max(scores_pos), len(scores_pos)))
    if scores_neg:
        print("    Negative scores: min={:.3f}  mean={:.3f}  max={:.3f}  (n={})".format(
            min(scores_neg), sum(scores_neg) / len(scores_neg),
            max(scores_neg), len(scores_neg)))

    if scores_pos and scores_neg:
        gap = min(scores_pos) - max(scores_neg)
        if gap > 0:
            print("    Separation gap: {}{:.3f} (positive min > negative max){}".format(
                GREEN, gap, RESET))
        else:
            print("    {}Overlap: {:.3f} (some scores cross — boundary is fuzzy){}".format(
                YELLOW, -gap, RESET))

    print("\n  {}NOTE:{} Thresholds in config_{}.ini are UNCHANGED.".format(
        YELLOW, RESET, name))
    print("  The AMO training optimizes anchor positions, not evaluator thresholds.")
    print("  To recalibrate thresholds for the new embeddings, run:")
    print("    python semantic_anchor_{}.py --auto-calibrate labeled_data.txt".format(name))

    print("\n  {}AMO training complete.{}".format(GREEN, RESET))
    print("  The evaluator will automatically use optimized embeddings when")
    print("  {} exists.".format(optimized_path))
    return optimized_path
