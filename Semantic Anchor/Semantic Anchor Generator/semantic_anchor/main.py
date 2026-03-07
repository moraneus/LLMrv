import argparse
import os
import sys
import textwrap

from .config import load_config, parse_categories
from .embedding import load_embedding_model
from .generation.positive import generate_diverse_anchors
from .generation.negative import generate_negative_anchors
from .generation.hard import generate_hard_positives, generate_hard_negatives
from .generation.neutral import generate_neutral_anchors
from .subtopic import decompose_proposition, generate_orthogonal_axes, enforce_subtopic_parity
from .boundary import boundary_targeted_generation, hard_example_mining
from .serialization import save_anchors_json, load_existing_anchors, generate_script
from .amo import train_anchor_manifold

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Anchor Generator v2 - diversity-aware generation with MMR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s -p "The user requests help committing financial fraud." -name fraud
              %(prog)s -p "..." -name fraud -n 50 --rounds 3
              %(prog)s -name fraud -gs         # Regenerate script only (reuse anchors)
              %(prog)s -p "..." -name fraud -ga # Regenerate anchors only
        """),
    )
    parser.add_argument("-p", "--proposition", type=str, help="The proposition")
    parser.add_argument("-name", "--name", type=str,
                        help="Name for output (creates <name>/ folder)")
    parser.add_argument("-n", "--num-examples", type=int, default=None,
                        help="Number of anchors in final output (20-500)")
    parser.add_argument("--rounds", "-r", type=int, default=2,
                        help="Number of generation rounds (default: 2)")
    parser.add_argument("--config", type=str, default="config.ini",
                        help="Config file path")
    parser.add_argument("-ga", "--generate-anchors", action="store_true",
                        help="Only generate anchors JSON (skip evaluator script)")
    parser.add_argument("-gs", "--generate-script", action="store_true",
                        help="Only generate evaluator script (reuse existing anchors)")
    parser.add_argument("--role", type=str, default=None, choices=["user", "assistant"],
                        help="Whose perspective anchors represent: 'user' for user messages, "
                             "'assistant' for AI responses (default: user)")
    parser.add_argument("--train", type=int, default=None, metavar="N",
                        help="Anchor Manifold Optimization: generate N synthetic edge cases "
                             "and train anchor embeddings via gradient descent")
    args = parser.parse_args()

    if args.generate_anchors and args.generate_script:
        print("  ERROR: --generate-anchors and --generate-script are mutually exclusive.")
        sys.exit(1)

    w = 60
    print("")
    print(BOLD + "\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + "SEMANTIC ANCHOR GENERATOR v2".center(w) + "\u2551")
    print("\u2551" + "Diversity-aware generation with MMR".center(w) + "\u2551")
    print("\u255a" + "\u2550" * w + "\u255d" + RESET)
    print("")

    config = load_config(args.config)

    # --- Get name ---
    if args.name:
        name = args.name
    else:
        print("  {}Enter a name for the output:{}".format(BOLD, RESET))
        print("  {}(Creates folder: <name>/ with anchors and evaluator script){}".format(DIM, RESET))
        try:
            name = input("\n  Name> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        if not name:
            print("  ERROR: Name cannot be empty."); sys.exit(1)

    # --- Parse name: support paths like "Examples/p1" → parent="Examples", name="p1" ---
    raw_path = name
    if "/" in raw_path or os.sep in raw_path:
        parent_dir = os.path.dirname(raw_path)
        name = os.path.basename(raw_path)
    else:
        parent_dir = ""

    name = name.replace(" ", "_").replace("-", "_").lower()

    # --- Create output directory ---
    if parent_dir:
        output_dir = os.path.join(parent_dir, name)
    else:
        output_dir = name
    os.makedirs(output_dir, exist_ok=True)

    # --- Thresholds ---
    match_thresh = float(config.get("thresholds", "match_threshold"))
    warn_thresh = float(config.get("thresholds", "warning_threshold"))

    # --- Check for existing anchors ---
    existing_data, existing_path = load_existing_anchors(output_dir, name)

    # =====================================================================
    # SCRIPT-ONLY MODE: just regenerate the evaluator script
    # =====================================================================
    if args.generate_script:
        if existing_data is None:
            print("  ERROR: No existing anchors file found at {}/anchors_list_{}.json".format(
                output_dir, name))
            print("  Run without --generate-script first to generate anchors.")
            sys.exit(1)

        total = existing_data["metadata"]["total_anchors"]
        print("  {}Mode:{} Script-only (reusing existing anchors)".format(BOLD, RESET))
        print("  Anchors file:  {}".format(existing_path))
        print("  Proposition:   \"{}\"".format(existing_data["proposition"]))
        print("  Total anchors: {}".format(total))

        script_content = generate_script(name)
        script_path = os.path.join(output_dir, "semantic_anchor_{}.py".format(name))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        print("\n  {}{}\u2713 Generated: {}{}".format(GREEN, BOLD, script_path, RESET))
        _generate_config_file(output_dir, name, existing_data["proposition"], config)
        _print_usage(script_path)
        return

    # =====================================================================
    # TRAIN MODE: Anchor Manifold Optimization
    # =====================================================================
    if args.train is not None:
        if existing_data is None:
            print("  {}ERROR:{} No existing anchors found at {}/anchors_list_{}.json".format(
                "\033[91m", RESET, output_dir, name))
            print("  Run the generator first to create anchors, then use --train.")
            sys.exit(1)

        proposition = existing_data["proposition"]
        role = existing_data.get("role", "user")

        w = 60
        print("  " + BOLD + "\u2554" + "\u2550" * w + "\u2557")
        print("  \u2551" + "ANCHOR MANIFOLD OPTIMIZATION (AMO)".center(w) + "\u2551")
        print("  \u2551" + "Trainable anchor embeddings".center(w) + "\u2551")
        print("  \u255a" + "\u2550" * w + "\u255d" + RESET)

        print("\n  {}Proposition:{} \"{}\"".format(BOLD, RESET, proposition))
        print("  {}Role:{}        {}".format(BOLD, RESET, role))
        print("  {}Name:{}        {}".format(BOLD, RESET, name))

        train_anchor_manifold(
            proposition, name, config,
            n_synthetic=args.train, role=role)
        return

    # =====================================================================
    # ANCHOR GENERATION (default or --anchors-only)
    # =====================================================================

    # --- Get proposition ---
    if args.proposition:
        proposition = args.proposition
    elif existing_data and existing_data.get("proposition"):
        proposition = existing_data["proposition"]
        print("  {}Reusing proposition from existing anchors:{} \"{}\"".format(
            DIM, RESET, proposition))
    else:
        print("  {}Enter the proposition:{}".format(BOLD, RESET))
        print("  {}(A formal description of the type of message to detect){}".format(DIM, RESET))
        try:
            proposition = input("\n  Proposition> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        if not proposition:
            print("  ERROR: Proposition cannot be empty."); sys.exit(1)

    # --- Params ---
    if args.num_examples is not None:
        target_n = max(20, min(500, args.num_examples))
    else:
        target_n = max(20, min(500, int(config.get("anchors", "num_examples"))))

    num_rounds = max(1, min(10, args.rounds))

    # --- Role ---
    if args.role:
        role = args.role
    elif existing_data and existing_data.get("role"):
        role = existing_data["role"]
        print("  {}Reusing role from existing anchors:{} {}".format(DIM, RESET, role))
    else:
        role = "user"  # default

    # --- Summary ---
    provider = config.get("llm", "provider")
    llm_model = config.get("llm", "model")
    mode = "Anchors only" if args.generate_anchors else "Full (anchors + script)"
    print("\n  {}Configuration:{}".format(BOLD, RESET))
    print("  Mode:            {}".format(mode))
    print("  Role:            {}".format(role))
    print("  Proposition:     \"{}\"".format(proposition))
    print("  Output folder:   {}/".format(output_dir))
    print("  LLM provider:    {} ({})".format(provider, llm_model))
    print("  Target anchors:  {}".format(target_n))
    print("  Gen rounds:      {} (over-generate then MMR-select)".format(num_rounds))
    print("  Match threshold: {}".format(match_thresh))
    print("  Warn threshold:  {}".format(warn_thresh))

    if existing_data:
        existing_count = existing_data["metadata"]["total_anchors"]
        print("  {}Existing anchors: {} (will be overwritten){}".format(
            YELLOW, existing_count, RESET))

    # --- Decompose proposition into subtopics ---
    subtopics = decompose_proposition(proposition, config, role=role)

    # --- Generate orthogonal semantic axes (Thematic Vacuum Strategy) ---
    # These are domain-specific axes that ensure generation covers the full
    # semantic space, not just the most obvious angles.
    use_ortho = config.get("anchors", "orthogonal_axes", fallback="true").strip().lower() == "true"
    if use_ortho:
        ortho_axes = generate_orthogonal_axes(proposition, subtopics, config, role=role)
    else:
        ortho_axes = []
        print("\n  {}Orthogonal axes:{} Disabled (orthogonal_axes = false)".format(DIM, RESET))

    # --- Merge orthogonal axes with config categories ---
    # Config categories provide generic coverage (polite, direct, coded, etc.)
    # Orthogonal axes provide domain-specific coverage (method-specific, tool-specific)
    # Together they create a comprehensive generation grid.
    categories = parse_categories(config, role=role)
    if ortho_axes:
        # Combine: static categories + orthogonal axes, deduplicating
        existing_lower = {c.lower() for c in categories}
        n_static = len(categories)
        added = 0
        for axis in ortho_axes:
            if axis.lower() not in existing_lower:
                categories.append(axis)
                existing_lower.add(axis.lower())
                added += 1
        if added > 0:
            print("\n  {}Combined categories:{} {} static + {} orthogonal = {} total".format(
                BOLD, RESET, n_static, added, len(categories)))

    # --- Auto-compute optimal target_n ---
    # Minimum: 2 anchors per category so MMR has meaningful selection per slot.
    # With subtopics: also need knn_size / num_subtopics per subtopic.
    n_cats = len(categories)
    n_subs = len(subtopics) if subtopics and len(subtopics) > 1 else 1
    knn_size = int(config.get("thresholds", "knn_size", fallback="20"))

    min_for_categories = n_cats * 2           # 2 per category
    min_for_subtopics = knn_size              # need knn_size / n_subs per sub, sum = knn_size
    min_for_coverage = max(min_for_categories, min_for_subtopics)

    user_set_n = args.num_examples is not None

    if target_n < min_for_coverage:
        old_n = target_n
        target_n = min(500, min_for_coverage)
        if user_set_n:
            print("\n  {}Auto-adjusted -n: {} → {} ({} categories × 2 = {}, "
                  "knn parity needs {}){}".format(
                      YELLOW, old_n, target_n, n_cats, n_cats * 2,
                      min_for_subtopics, RESET))
        else:
            print("\n  {}Auto-computed -n: {} ({} categories × 2 = {}, "
                  "knn parity needs {}){}".format(
                      BOLD, target_n, n_cats, n_cats * 2,
                      min_for_subtopics, RESET))
    elif not user_set_n:
        print("\n  {}Target anchors:{} {} ({} categories, {} subtopics)".format(
            BOLD, RESET, target_n, n_cats, n_subs))

    # --- Generate anchors ---
    anchors_dict = generate_diverse_anchors(
        proposition, categories, target_n, num_rounds, config, role=role,
        subtopics=subtopics)

    total = sum(len(v) for v in anchors_dict.values())
    print("\n  {}\u2713 Final: {} diversity-optimized positive anchors{}".format(GREEN, total, RESET))

    # --- Generate negative (contrastive) anchors ---
    # 2:1 negative-to-positive ratio by default — the benign space is much
    # larger than the harmful space and needs broader coverage for KNN voting.
    neg_ratio = float(config.get("anchors", "negative_ratio", fallback="2.0"))
    neg_target = max(target_n, int(target_n * neg_ratio))
    negative_dict = generate_negative_anchors(
        proposition, anchors_dict, neg_target, config, num_rounds=num_rounds, role=role,
        subtopics=subtopics)

    neg_total = sum(len(v) for v in negative_dict.values())

    # --- Generate hard positives (boundary-straddling examples) ---
    # These sound benign but genuinely match the proposition — critical for
    # NLI KNN voting where polite/professional harmful messages get lost
    # among negative anchors without dedicated boundary examples.
    hard_pos_ratio = float(config.get("anchors", "hard_positive_ratio", fallback="0.3"))
    hard_pos_target = max(5, int(target_n * hard_pos_ratio))
    hard_pos_dict = generate_hard_positives(
        proposition, anchors_dict, negative_dict, hard_pos_target, config, role=role)

    # Merge hard positives into the main positive anchors dict
    hard_pos_total = 0
    for cat, examples in hard_pos_dict.items():
        if cat in anchors_dict:
            anchors_dict[cat].extend(examples)
        else:
            anchors_dict[cat] = examples
        hard_pos_total += len(examples)

    if hard_pos_total > 0:
        total = sum(len(v) for v in anchors_dict.values())
        print("\n  {} Merged {} hard positives into anchor set (new total: {}){}".format(
            GREEN, hard_pos_total, total, RESET))

    # --- Generate hard negatives (boundary-straddling benign examples) ---
    # These sound harmful but are actually legitimate — critical for reducing
    # false positives on educational, definitional, and defensive queries
    # that share vocabulary with positive anchors.
    hard_neg_ratio = float(config.get("anchors", "hard_negative_ratio", fallback="0.3"))
    hard_neg_target = max(5, int(target_n * hard_neg_ratio))
    hard_neg_dict = generate_hard_negatives(
        proposition, anchors_dict, negative_dict, hard_neg_target, config, role=role)

    # Merge hard negatives into the main negative anchors dict
    hard_neg_total = 0
    for cat, examples in hard_neg_dict.items():
        if cat in negative_dict:
            negative_dict[cat].extend(examples)
        else:
            negative_dict[cat] = examples
        hard_neg_total += len(examples)

    if hard_neg_total > 0:
        neg_total = sum(len(v) for v in negative_dict.values())
        print("\n  {} Merged {} hard negatives into negative set (new total: {}){}".format(
            GREEN, hard_neg_total, neg_total, RESET))

    # --- Subtopic KNN parity: ensure each subtopic has enough anchors ---
    knn_size = int(config.get("thresholds", "knn_size", fallback="20"))
    emb_model_name = config.get("anchors", "embedding_model")
    emb_model = load_embedding_model(emb_model_name)

    if subtopics and len(subtopics) > 1:
        print("\n  {}Subtopic KNN parity:{} Checking minimum per subtopic...".format(
            BOLD, RESET))
        anchors_dict = enforce_subtopic_parity(
            proposition, subtopics, anchors_dict, knn_size,
            emb_model, config, role=role, anchor_type="positive",
            target_n=target_n)
        negative_dict = enforce_subtopic_parity(
            proposition, subtopics, negative_dict, knn_size,
            emb_model, config, role=role, anchor_type="negative",
            target_n=neg_target)

    # --- Boundary targeting: train separator, find weak zones, fill gaps ---
    boundary_pos, boundary_neg = boundary_targeted_generation(
        proposition, anchors_dict, negative_dict, config, emb_model, role=role)

    for cat, examples in boundary_pos.items():
        if cat in anchors_dict:
            anchors_dict[cat].extend(examples)
        else:
            anchors_dict[cat] = examples
    for cat, examples in boundary_neg.items():
        if cat in negative_dict:
            negative_dict[cat].extend(examples)
        else:
            negative_dict[cat] = examples

    # --- Hard example mining: synthesize queries, collect failures, fix ---
    mined_pos, mined_neg = hard_example_mining(
        proposition, anchors_dict, negative_dict, config, emb_model, role=role)

    for cat, examples in mined_pos.items():
        if cat in anchors_dict:
            anchors_dict[cat].extend(examples)
        else:
            anchors_dict[cat] = examples
    for cat, examples in mined_neg.items():
        if cat in negative_dict:
            negative_dict[cat].extend(examples)
        else:
            negative_dict[cat] = examples

    total = sum(len(v) for v in anchors_dict.values())
    neg_total = sum(len(v) for v in negative_dict.values())
    print("\n  {}\u2713 Final anchor set: {} positive, {} negative{}".format(
        GREEN, total, neg_total, RESET))

    # --- Generate neutral (off-topic baseline) anchors ---
    neutral_list = generate_neutral_anchors(proposition, config, role=role)

    # --- Preview ---
    print("\n  {}Preview (positive):{}".format(BOLD, RESET))
    for cat, examples in anchors_dict.items():
        preview = examples[0][:50] + "..." if len(examples[0]) > 50 else examples[0]
        print("    {} ({}): \"{}\"".format(cat, len(examples), preview))

    if negative_dict:
        print("\n  {}Preview (negative):{}".format(BOLD, RESET))
        for cat, examples in negative_dict.items():
            preview = examples[0][:50] + "..." if len(examples[0]) > 50 else examples[0]
            print("    {} ({}): \"{}\"".format(cat, len(examples), preview))

    if neutral_list:
        print("\n  {}Preview (neutral):{}".format(BOLD, RESET))
        for ex in neutral_list[:3]:
            print("    \"{}\"".format(ex[:60]))

    # --- Save anchors JSON ---
    anchors_path = save_anchors_json(
        output_dir, name, proposition, anchors_dict, match_thresh, warn_thresh,
        config, negative_dict=negative_dict, neutral_list=neutral_list, role=role,
        subtopics=subtopics)
    print("\n  {}{}\u2713 Anchors: {} ({} positive, {} negative, {} neutral){}".format(
        GREEN, BOLD, anchors_path, total, neg_total, len(neutral_list), RESET))

    # =====================================================================
    # GENERATE EVALUATOR SCRIPT (unless --anchors-only)
    # =====================================================================
    if not args.generate_anchors:
        script_content = generate_script(name)
        script_path = os.path.join(output_dir, "semantic_anchor_{}.py".format(name))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        print("  {}{}\u2713 Script:  {}{}".format(GREEN, BOLD, script_path, RESET))

        # Generate LLM judge config file
        _generate_config_file(output_dir, name, proposition, config)

        _print_usage(script_path)
    else:
        print("\n  {}Anchors-only mode — evaluator script not generated.{}".format(DIM, RESET))
        print("  To generate the script: python semantic_anchor_generator.py -name {} -gs".format(name))
        _generate_config_file(output_dir, name, proposition, config)
    print()


def _generate_config_file(output_dir, name, proposition, config=None):
    """Generate or UPDATE config_<n>.ini template for evaluator (LLM judge + models)."""
    import configparser as _cp

    config_path = os.path.join(output_dir, "config_{}.ini".format(name))

    # Read model names from generator config (or use defaults)
    if config:
        emb_model = config.get("anchors", "embedding_model", fallback="all-mpnet-base-v2")
        nli_model = config.get("anchors", "nli_model", fallback="cross-encoder/nli-deberta-v3-large")
    else:
        emb_model = "all-mpnet-base-v2"
        nli_model = "cross-encoder/nli-deberta-v3-large"

    if os.path.exists(config_path):
        # --- UPDATE existing config: add missing sections ---
        existing = _cp.ConfigParser()
        existing.read(config_path)
        updated = False

        if not existing.has_section("models"):
            # Append [models] section to existing file
            with open(config_path, "a", encoding="utf-8") as f:
                f.write("\n\n# =========================================================\n")
                f.write("# [models] — Embedding & NLI Models (added by generator)\n")
                f.write("# =========================================================\n")
                f.write("#\n")
                f.write("# These override the models stored in anchors_list_{}.json.\n".format(name))
                f.write("# Change here and restart — no regeneration needed.\n\n")
                f.write("[models]\n\n")
                f.write("# Embedding model for cosine similarity and extraction\n")
                f.write("#   all-mpnet-base-v2          — Default, good general-purpose, fast\n")
                f.write("#   BAAI/bge-large-en-v1.5     — Instruction-tuned, better accuracy\n")
                f.write("#   all-MiniLM-L6-v2           — Lightweight, fastest\n")
                f.write("embedding_model = {}\n\n".format(emb_model))
                f.write("# NLI cross-encoder for --mode nli, --mode hybrid, --compare\n")
                f.write("#   cross-encoder/nli-deberta-v3-large  — Best accuracy (recommended)\n")
                f.write("#   cross-encoder/nli-deberta-v3-base   — Faster, slightly less accurate\n")
                f.write("#   cross-encoder/nli-deberta-v3-xsmall — Fastest, lowest accuracy\n")
                f.write("nli_model = {}\n".format(nli_model))
            updated = True
            print("  {}{}\u2713 Config:  {} (added [models] section){}".format(
                GREEN, BOLD, config_path, RESET))

        if not existing.has_section("options"):
            with open(config_path, "a", encoding="utf-8") as f:
                f.write("\n\n# =========================================================\n")
                f.write("# [options] — Evaluator Behavior (added by generator)\n")
                f.write("# =========================================================\n\n")
                f.write("[options]\n\n")
                f.write("# Enable autocorrect spell checking (default: false)\n")
                f.write("# Warning: may mangle technical terms like OAuth, SSO, etc.\n")
                f.write("spellcheck = false\n")
            updated = True
            print("  {}{}\u2713 Config:  {} (added [options] section){}".format(
                GREEN, BOLD, config_path, RESET))

        if not existing.has_section("thresholds"):
            with open(config_path, "a", encoding="utf-8") as f:
                f.write("\n\n# =========================================================\n")
                f.write("# [thresholds] — Scoring Thresholds (added by generator)\n")
                f.write("# =========================================================\n\n")
                f.write("[thresholds]\n\n")
                f.write("# KNN voting thresholds (applies to cosine, nli, hybrid):\n")
                f.write("#   score <= warning_threshold       → NO MATCH\n")
                f.write("#   warning < score <= match         → WARNING\n")
                f.write("#   score > match_threshold          → MATCH\n")
                f.write("#\n")
                f.write("# With knn_size=20: 10/20=50% → NO MATCH, 11/20=55% → WARNING, 15/20=75% → MATCH\n")
                f.write("match_threshold = 0.70\n")
                f.write("warning_threshold = 0.50\n\n")
                f.write("# KNN neighborhood size for cosine voting (default: 20)\n")
                f.write("# Higher = more stable, lower = more sensitive\n")
                f.write("knn_size = 20\n\n")
                f.write("# NLI KNN abstain margin (default: 0.15)\n")
                f.write("# When vote margin (|pos-neg|/total) is below this, flag as abstain.\n")
                f.write("# Abstain = ambiguous, consider routing to LLM judge.\n")
                f.write("# Higher = more abstains (safer), lower = force more decisions.\n")
                f.write("nli_abstain_margin = 0.15\n\n")
                f.write("# NLI retrieval mode (default: 40 = cosine pre-filter)\n")
                f.write("# Cosine pre-filter top-K candidates, then NLI re-rank.\n")
                f.write("# Increase for broad propositions (60-80).\n")
                f.write("nli_retrieve_k = 40\n\n")
                f.write("# NLI vote neighborhood size (default: 20)\n")
                f.write("# After NLI re-ranking, vote on the top K candidates.\n")
                f.write("nli_vote_k = 20\n\n")
                f.write("# NLI asymmetric forward weight (default: 0.7)\n")
                f.write("# 0.7 = 70%% forward (message->anchor) + 30%% backward (anchor->message)\n")
                f.write("nli_fwd_weight = 0.7\n\n")
                f.write("# Hybrid merged KNN pool size (default: 40)\n")
                f.write("# Takes pool/2 from cosine + pool/2 from NLI, unions them.\n")
                f.write("# Set to 100 for 50 cosine + 50 NLI voters.\n")
                f.write("hybrid_pool_size = 40\n")
            updated = True
            print("  {}{}\u2713 Config:  {} (added [thresholds] section){}".format(
                GREEN, BOLD, config_path, RESET))

        if not updated:
            print("  {}\u2713 Config:  {} (up to date){}".format(
                DIM, config_path, RESET))
    else:
        # --- CREATE new config from scratch ---
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("# =========================================================\n")
            f.write("# Semantic Anchor Evaluator \u2014 config_{}.ini\n".format(name))
            f.write("# =========================================================\n")
            f.write("#\n")
            f.write("# This file configures the evaluator script.\n")
            f.write("# Settings here OVERRIDE values stored in anchors_list_{}.json.\n".format(name))
            f.write("#\n")
            f.write("# To change models without regenerating anchors, edit this file.\n")
            f.write("# =========================================================\n\n\n")
            f.write("# =========================================================\n")
            f.write("# [models] \u2014 Embedding & NLI Models\n")
            f.write("# =========================================================\n")
            f.write("#\n")
            f.write("# These override the models stored in anchors_list_{}.json.\n".format(name))
            f.write("# Change here and restart \u2014 no regeneration needed.\n\n")
            f.write("[models]\n\n")
            f.write("# Embedding model for cosine similarity and extraction\n")
            f.write("#   all-mpnet-base-v2          \u2014 Default, good general-purpose, fast\n")
            f.write("#   BAAI/bge-large-en-v1.5     \u2014 Instruction-tuned, better accuracy\n")
            f.write("#   all-MiniLM-L6-v2           \u2014 Lightweight, fastest\n")
            f.write("embedding_model = {}\n\n".format(emb_model))
            f.write("# NLI cross-encoder for --mode nli, --mode hybrid, --compare\n")
            f.write("#   cross-encoder/nli-deberta-v3-large  \u2014 Best accuracy (recommended)\n")
            f.write("#   cross-encoder/nli-deberta-v3-base   \u2014 Faster, slightly less accurate\n")
            f.write("#   cross-encoder/nli-deberta-v3-xsmall \u2014 Fastest, lowest accuracy\n")
            f.write("nli_model = {}\n\n\n".format(nli_model))
            f.write("# =========================================================\n")
            f.write("# [llm_judge] \u2014 LLM Provider for --mode llm\n")
            f.write("# =========================================================\n")
            f.write("#\n")
            f.write("# Supported providers:\n")
            f.write("#   anthropic  \u2014 Claude  (api.anthropic.com)\n")
            f.write("#   openai     \u2014 GPT     (api.openai.com)\n")
            f.write("#   gemini     \u2014 Gemini  (generativelanguage.googleapis.com)\n")
            f.write("#   grok       \u2014 Grok    (api.x.ai)\n")
            f.write("#   openrouter \u2014 Any model via OpenRouter (openrouter.ai)\n")
            f.write("#   ollama     \u2014 Local   (localhost:11434)\n")
            f.write("#   lmstudio   \u2014 Local   (localhost:1234, OpenAI-compatible)\n")
            f.write("#   vllm       \u2014 Local   (localhost:8000, OpenAI-compatible)\n\n")
            f.write("[llm_judge]\n\n")
            f.write("# --- Uncomment ONE provider block below ---\n\n")
            f.write("# >> Anthropic Claude <<\n")
            f.write("provider = anthropic\n")
            f.write("model = claude-sonnet-4-20250514\n")
            f.write("api_key = YOUR_API_KEY_HERE\n\n")
            f.write("# >> OpenAI GPT <<\n")
            f.write("# provider = openai\n")
            f.write("# model = gpt-4o\n")
            f.write("# api_key = sk-...\n\n")
            f.write("# >> Google Gemini <<\n")
            f.write("# provider = gemini\n")
            f.write("# model = gemini-2.0-flash\n")
            f.write("# api_key = AIza...\n\n")
            f.write("# >> xAI Grok <<\n")
            f.write("# provider = grok\n")
            f.write("# model = grok-3-mini-fast\n")
            f.write("# api_key = xai-...\n\n")
            f.write("# >> OpenRouter (access 200+ models via single API key) <<\n")
            f.write("# provider = openrouter\n")
            f.write("# model = anthropic/claude-sonnet-4   # or google/gemini-2.5-flash, etc.\n")
            f.write("# api_key = sk-or-...\n\n")
            f.write("# >> Local Ollama (free, no API key needed) <<\n")
            f.write("# provider = ollama\n")
            f.write("# model = llama3.1:8b\n")
            f.write("# api_key = not-needed\n")
            f.write("# base_url = http://localhost:11434\n\n")
            f.write("# >> Local LM Studio <<\n")
            f.write("# provider = lmstudio\n")
            f.write("# model = loaded-model\n")
            f.write("# api_key = not-needed\n")
            f.write("# base_url = http://localhost:1234\n\n")
            f.write("# >> Local vLLM <<\n")
            f.write("# provider = vllm\n")
            f.write("# model = meta-llama/Llama-3.1-8B-Instruct\n")
            f.write("# api_key = not-needed\n")
            f.write("# base_url = http://localhost:8000\n\n\n")
            f.write("# =========================================================\n")
            f.write("# [options] \u2014 Evaluator Behavior\n")
            f.write("# =========================================================\n\n")
            f.write("[options]\n\n")
            f.write("# Enable autocorrect spell checking (default: false)\n")
            f.write("# Warning: may mangle technical terms like OAuth, SSO, etc.\n")
            f.write("spellcheck = false\n\n\n")
            f.write("# =========================================================\n")
            f.write("# [thresholds] — Scoring Thresholds\n")
            f.write("# =========================================================\n")
            f.write("#\n")
            f.write("# These override the thresholds stored in anchors_list_{}.json.\n".format(name))
            f.write("# Used by cosine KNN, NLI, and hybrid modes.\n\n")
            f.write("[thresholds]\n\n")
            f.write("# KNN voting thresholds (applies to cosine, nli, hybrid):\n")
            f.write("#   score <= warning_threshold       → NO MATCH\n")
            f.write("#   warning < score <= match         → WARNING\n")
            f.write("#   score > match_threshold          → MATCH\n")
            f.write("#\n")
            f.write("# With knn_size=20: 10/20=50%% → NO MATCH, 11/20=55%% → WARNING, 15/20=75%% → MATCH\n")
            f.write("match_threshold = 0.70\n")
            f.write("warning_threshold = 0.50\n\n")
            f.write("# KNN neighborhood size for cosine voting (default: 20)\n")
            f.write("# Higher = more stable, lower = more sensitive\n")
            f.write("knn_size = 20\n\n")
            f.write("# NLI KNN abstain margin (default: 0.15)\n")
            f.write("# When vote margin (|pos-neg|/total) is below this, flag as abstain.\n")
            f.write("# Abstain cases are ambiguous — consider routing to LLM judge.\n")
            f.write("# Higher = more abstains, lower = force more decisions.\n")
            f.write("nli_abstain_margin = 0.15\n\n")
            f.write("# NLI retrieval mode (default: 40 = cosine pre-filter)\n")
            f.write("# Cosine pre-filter top-K candidates, then NLI re-rank.\n")
            f.write("nli_retrieve_k = 40\n\n")
            f.write("# NLI vote neighborhood size (default: 20)\n")
            f.write("# After NLI re-ranking, vote on the top K candidates.\n")
            f.write("nli_vote_k = 20\n\n")
            f.write("# NLI asymmetric forward weight (default: 0.7)\n")
            f.write("# 0.7 = 70%% forward (message->anchor) + 30%% backward (anchor->message)\n")
            f.write("nli_fwd_weight = 0.7\n\n")
            f.write("# Hybrid merged KNN pool size (default: 40)\n")
            f.write("# Takes pool/2 from cosine + pool/2 from NLI, unions them.\n")
            f.write("hybrid_pool_size = 40\n\n\n")
            f.write("# =========================================================\n")
            f.write("# Proposition (used by LLM judge mode)\n")
            f.write("# =========================================================\n")
            f.write("proposition = {}\n".format(proposition))
        print("  {}{}\u2713 Config:  {} (edit models & API key){}".format(
            GREEN, BOLD, config_path, RESET))
    return config_path

def _print_usage(script_path):
    print("\n  {}Usage:{}".format(BOLD, RESET))
    print("    python {}                           # Interactive (cosine)".format(script_path))
    print("    python {} --mode nli                # NLI entailment scoring".format(script_path))
    print("    python {} --mode hybrid             # Merged cosine + NLI KNN (recommended)".format(script_path))
    print("    python {} --mode llm                # LLM-as-judge scoring".format(script_path))
    print("    python {} --file input.txt          # Evaluate file".format(script_path))
    print("    python {} --compare                 # Compare all modes side-by-side".format(script_path))
    print("    python {} --compare -f input.txt    # Compare all modes on file".format(script_path))
    print("    python {} --verbose                 # Full table".format(script_path))
    print("    python {} --show-examples           # View anchors by category".format(script_path))
