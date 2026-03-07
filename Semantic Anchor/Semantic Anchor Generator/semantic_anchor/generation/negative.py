import json

import numpy as np

from ..embedding import load_embedding_model, embed_texts, cosine_sim_matrix
from ..llm import call_llm, _role_context
from ..config import parse_categories
from .selection import find_clusters, cluster_constrained_mmr
from ..subtopic import audit_subtopic_coverage, generate_subtopic_gap_fill

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def generate_negative_anchors(proposition, positive_anchors_dict, target_n, config,
                               num_rounds=None, role="user", subtopics=None):
    """
    Generate NEGATIVE (contrastive) anchors — messages that use similar
    terminology to the proposition but have clearly legitimate intent.

    Uses the same diversity pipeline as positive anchors:
      1. Generate negative-specific categories from LLM (fallback: config.ini)
      2. Seed round for core negatives
      3. Round 1 with categories
      4. Rounds 2+N with clustering/gap analysis (2× positive rounds)
      5. Remove negatives too close to positives (> 0.85)
      6. MMR selection for final diverse set

    Negative rounds = 2× positive rounds (negatives need more diversity
    since the benign space is much larger than the harmful space).
    """
    emb_model_name = config.get("anchors", "embedding_model")
    emb_model = load_embedding_model(emb_model_name)
    model_name = config.get("llm", "model")

    # Negative rounds = 2× positive rounds (capped at 8)
    if num_rounds is None:
        neg_rounds = min(8, max(4, (target_n // 40) + 1))
    else:
        neg_rounds = min(8, num_rounds * 2)

    # Collect positive samples for context
    pos_samples = []
    for cat, examples in positive_anchors_dict.items():
        pos_samples.extend(examples[:5])
    pos_sample_text = "\n".join("  - \"{}\"".format(s) for s in pos_samples[:20])

    rc = _role_context(role)

    neg_system = (
        "You are generating NEGATIVE (benign) test inputs for an AI safety classifier. "
        "These are {} that use similar vocabulary to harmful {} but have "
        "clearly legitimate intent. They serve as contrastive examples to reduce false "
        "positives. Output ONLY valid JSON."
    ).format(rc["example_noun_short"], rc["example_noun_short"])

    # =====================================================================
    # Step 0: Generate negative categories from LLM
    # =====================================================================
    print("\n  {}Negative anchors:{} Generating contrastive categories...".format(
        BOLD, RESET))

    # Fallback: use config.ini categories for this role
    config_categories = parse_categories(config, role=role)

    cat_prompt = (
        rc["safety_context"] +
        'PROPOSITION (harmful intent to detect): "{}"\n\n'
        'POSITIVE ANCHOR EXAMPLES ({} that SHOULD be flagged):\n{}\n\n'
        'I need to generate NEGATIVE (benign, legitimate) test {} that use '
        'SIMILAR vocabulary but have CLEARLY innocent intent.\n\n'
        'Generate 5-7 categories of benign {} that would be common false positives '
        'for this proposition. Each category should represent a different REASON '
        'similar terminology might appear with legitimate intent.\n\n'
        'Make the categories SPECIFIC to this proposition\'s domain.\n\n'
        'Output ONLY valid JSON:\n'
        '{{"categories": ["Category 1 — short description", "Category 2 — ...", ...]}}'
    ).format(proposition, rc["example_noun_short"], pos_sample_text,
             rc["example_noun_short"], rc["example_noun_short"])

    cat_result = call_llm(config, neg_system, cat_prompt)

    neg_categories = config_categories  # fallback = config.ini categories
    try:
        if isinstance(cat_result, dict):
            parsed_cats = cat_result
        elif isinstance(cat_result, list):
            # LLM returned a raw list of category strings
            parsed_cats = {"categories": cat_result}
        else:
            parsed_cats = json.loads(str(cat_result).strip())
        if "categories" in parsed_cats and isinstance(parsed_cats["categories"], list):
            llm_cats = [str(c).strip() for c in parsed_cats["categories"] if str(c).strip()]
            if len(llm_cats) >= 3:
                neg_categories = llm_cats
                print("    Generated {} categories from LLM".format(len(neg_categories)))
            else:
                print("    LLM returned too few categories, using config.ini categories")
        else:
            print("    Could not parse categories, using config.ini categories")
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # Last resort: try to extract strings from raw text
        raw = str(cat_result) if cat_result else ""
        extracted = []
        for line in raw.split("\n"):
            line = line.strip().strip("-•*").strip()
            if len(line) > 10 and not line.startswith("{") and not line.startswith("["):
                extracted.append(line)
        if len(extracted) >= 3:
            neg_categories = extracted
            print("    Extracted {} categories from raw LLM output".format(len(neg_categories)))
        else:
            preview = str(cat_result)[:120] if cat_result else "(empty)"
            print("    Category generation failed ({}), using config.ini categories".format(e))
            print("    {}LLM returned: {}{}".format(DIM, preview, RESET))

    for i, cat in enumerate(neg_categories):
        print("      {}. {}".format(i + 1, cat))

    # =====================================================================
    # Step 1: Seed generation — short core negatives
    # =====================================================================
    all_neg_texts = []
    all_neg_cats = []

    per_cat_target = max(5, target_n // len(neg_categories))
    pool_cap = max(300, target_n * 3)

    print("    Target: {} negatives | Rounds: {}{}".format(
        target_n, neg_rounds,
        " (capped from {}x{} positive)".format(
            2, num_rounds) if num_rounds and num_rounds * 2 > neg_rounds else ""))

    neg_seed_count = max(40, min(80, target_n // 5))
    if role == "assistant":
        neg_seed_count = max(neg_seed_count, 50)
    print("\n    {}Seed:{} Generating {} core negative anchors (boundary-aware)...".format(
        BOLD, RESET, neg_seed_count))

    # Collect the actual positive seeds/short examples for mirror generation
    pos_short = []
    for cat, examples in positive_anchors_dict.items():
        for ex in examples:
            if len(ex.split()) <= 15:  # short examples only
                pos_short.append(ex)
    pos_short = pos_short[:20]
    pos_short_text = "\n".join("  - \"{}\"".format(s) for s in pos_short)

    seed_prompt = (
        rc["safety_context"] +
        'PROPOSITION (harmful intent to detect): "{}"\n\n'
        'POSITIVE ANCHORS (harmful — a classifier SHOULD flag these):\n'
        '{}\n\n'
        'YOUR TASK: Generate {} BENIGN {} that are the MIRROR IMAGE of the '
        'positive anchors above.\n\n'
        'For EACH type of positive anchor, there should be a benign counterpart:\n\n'
        'MIRROR STRATEGIES (examples are illustrative — adapt to the proposition above):\n'
        'A) DEFINITIONAL mirrors — turn "how to do X" into "what is X?":\n'
        '   "How do I [harmful action]?" → "What is [harmful action]?"\n'
        '   "Help me [do harmful thing]" → "Define [harmful thing]"\n\n'
        'B) DEFENSIVE mirrors — same topic, protective intent:\n'
        '   "Help me [attack]" → "How to detect [attack]"\n'
        '   "[Harmful action] against someone" → "How to protect against [harmful action]"\n\n'
        'C) REPORTING mirrors — same topic, victim perspective:\n'
        '   "[Do harmful thing]" → "Report [harmful thing]"\n'
        '   "[Harmful action]" → "I\'m a victim of [harmful action], what do I do?"\n\n'
        'D) EDUCATIONAL mirrors — same vocabulary, learning context:\n'
        '   "How to [harmful action]" → "Explain [harmful action] penalties"\n'
        '   "[Do harmful thing]" → "How do authorities catch [harmful thing]?"\n\n'
        'E) LEGITIMATE ACTION mirrors — same verbs, legal context:\n'
        '   "[Action] for harmful purpose" → "[Action] for legitimate purpose"\n\n'
        'F) ENTERTAINMENT / RECOMMENDATION mirrors — same nouns, consumer context:\n'
        '   "[Build harmful thing]" → "Recommend a movie about [thing]"\n'
        '   "[Do harmful action]" → "Best documentary about [topic]?"\n\n'
        'G) SAFETY / STORAGE / HANDLING mirrors — same objects, safe-use context:\n'
        '   "[Make dangerous item]" → "How to safely store [item]?"\n'
        '   "[Build dangerous thing]" → "How to dispose of [thing] safely?"\n\n'
        'H) LEGAL / REGULATORY mirrors — same domain, compliance context:\n'
        '   "[Do illegal action]" → "What are the laws about [action]?"\n'
        '   "[Build restricted item]" → "What permits do I need for [item]?"\n\n'
        'I) HISTORICAL / HOBBY mirrors — same objects, collecting/reenactment:\n'
        '   "[Commit fraud]" → "History of [fraud type]"\n'
        '   "[Make device]" → "[Device] in museum collections"\n\n'
        'J) COMPARISON / REVIEW mirrors — same products, evaluation context:\n'
        '   "[Modify thing for harm]" → "What safety features does [thing] have?"\n'
        '   "[Build dangerous item]" → "Compare [item] brands for safety"\n\n'
        'RULES:\n'
        '- Every negative MUST use keywords that appear in the positive anchors above\n'
        '- Every negative MUST be obviously benign to any reasonable person\n'
        '- Keep them very short: 3-10 words\n'
        '- Cover ALL mirror strategies above\n'
        '{}\n\n'
        'Output ONLY valid JSON:\n'
        '{{"seeds": ["example1", "example2", ...]}}'
    ).format(proposition, pos_short_text, neg_seed_count,
             rc["example_noun_short"], rc["neg_style"])

    seed_result = call_llm(config, neg_system, seed_prompt)

    seeds = []
    try:
        if isinstance(seed_result, dict):
            if "seeds" in seed_result:
                seeds = seed_result["seeds"]
            else:
                for v in seed_result.values():
                    if isinstance(v, list):
                        seeds.extend(v)
        elif isinstance(seed_result, list):
            seeds = seed_result
    except Exception:
        pass

    for s in seeds:
        text = str(s).strip()
        if text and text not in all_neg_texts:
            all_neg_texts.append(text)
            all_neg_cats.append("Core (seed)")

    print("      Got {} seed negatives".format(len(seeds)))

    # =====================================================================
    # Step 2: Round 1 — large generation with categories
    # =====================================================================
    round1_n = int(target_n * 1.5)
    print("\n    {}Round 1:{} Generating {} negatives with {}...".format(
        BOLD, RESET, round1_n, model_name))

    neg_subtopic_section = ""
    has_neg_subtopics = subtopics and len(subtopics) > 1
    if has_neg_subtopics:
        st_lines = "\n".join("  {}. {}".format(i + 1, st) for i, st in enumerate(subtopics))
        neg_subtopic_section = (
            'SUBTOPIC DISTRIBUTION (CRITICAL):\n'
            'The proposition covers these distinct subtopics:\n'
            '{}\n\n'
            'You MUST generate negatives for ALL subtopics.\n'
            'Each subtopic needs benign mirrors that use its specific vocabulary.\n\n'
        ).format(st_lines)

    round1_prompt = (
        rc["safety_context"] +
        'PROPOSITION (harmful intent to detect): "{}"\n\n'
        'EXISTING POSITIVE ANCHORS ({} that SHOULD be flagged):\n{}\n\n'
        'Generate {} NEGATIVE (contrastive) anchor {} distributed across '
        'the categories below.\n\n'
        'These {} must:\n'
        '- Use SIMILAR VOCABULARY and TERMINOLOGY as the positive anchors\n'
        '- Have clearly LEGITIMATE, BENIGN intent\n'
        '- Be "hard negatives" — surface-similar to positives but clearly innocent\n'
        '- A safety system should NOT flag these\n\n'
        'CATEGORIES (distribute evenly):\n{}\n\n'
        '{}'
        'STYLE:\n{}\n\n'
        'CRITICAL RULES:\n'
        '- Use the SAME keywords that appear in positive anchors\n'
        '- Mix lengths: at least 30% very short (3-7 words), 30% medium, rest longer\n\n'
        'Output ONLY valid JSON:\n'
        '{{"categories": {{{}}}}}'
    ).format(
        proposition, rc["example_noun_short"], pos_sample_text, round1_n,
        rc["example_noun_short"], rc["example_noun_short"],
        "\n".join("  {}. {}".format(i + 1, c) for i, c in enumerate(neg_categories)),
        neg_subtopic_section,
        rc["neg_style"],
        ", ".join('"{}": ["ex1", "ex2", ...]'.format(c) for c in neg_categories)
    )

    result = call_llm(config, neg_system, round1_prompt)

    try:
        if isinstance(result, dict):
            parsed = result
        else:
            cleaned = str(result).strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            parsed = json.loads(cleaned)

        if "categories" in parsed:
            parsed = parsed["categories"]

        new_count = 0
        if isinstance(parsed, dict):
            for cat, examples in parsed.items():
                if isinstance(examples, list):
                    for ex in examples:
                        text = str(ex).strip()
                        if text and text not in all_neg_texts:
                            all_neg_texts.append(text)
                            all_neg_cats.append(cat)
                            new_count += 1
        elif isinstance(parsed, list):
            for i, ex in enumerate(parsed):
                text = str(ex).strip()
                if text and text not in all_neg_texts:
                    all_neg_texts.append(text)
                    all_neg_cats.append(neg_categories[i % len(neg_categories)])
                    new_count += 1

        print("      +{} new (total pool: {})".format(new_count, len(all_neg_texts)))

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print("      {}Round 1: parse error: {}{}".format(YELLOW, e, RESET))

    # =====================================================================
    # Step 3: Rounds 2+N — clustering analysis + diversity generation
    # =====================================================================
    for rnd in range(2, neg_rounds + 1):
        if len(all_neg_texts) < 2:
            print("\n    {}Round {}:{} Skipping — need at least 2 examples.".format(
                BOLD, rnd, RESET))
            continue

        if len(all_neg_texts) >= pool_cap:
            print("\n    Pool is full ({}/{}). Stopping generation rounds.".format(
                len(all_neg_texts), pool_cap))
            break

        print("\n    {}Round {}:{} Analyzing negative embedding space...".format(
            BOLD, rnd, RESET))

        neg_embeddings = embed_texts(emb_model, all_neg_texts)

        clusters = find_clusters(neg_embeddings, all_neg_texts, threshold=0.85)
        total_clustered = sum(len(c) for c in clusters)
        print("      Found {} clusters ({} examples too similar)".format(
            len(clusters), total_clustered))

        sim_matrix = cosine_sim_matrix(neg_embeddings)
        np.fill_diagonal(sim_matrix, 0)
        avg_sim = sim_matrix.mean()
        print("      Avg pairwise similarity: {:.4f}".format(avg_sim))

        gap_n = min(pool_cap - len(all_neg_texts), max(10, target_n // 2))
        if gap_n <= 0:
            break

        # Build cluster description for the LLM
        cluster_desc = ""
        for ci, cluster in enumerate(clusters):
            if len(cluster) > 1:
                cluster_desc += "\n    Cluster {} ({} too-similar examples):\n".format(
                    ci + 1, len(cluster))
                for ex in cluster[:3]:
                    cluster_desc += "      - \"{}\"\n".format(ex)
                if len(cluster) > 3:
                    cluster_desc += "      ... and {} more similar ones\n".format(
                        len(cluster) - 3)

        # Sample existing negatives (cluster representatives + random)
        MAX_NEG_IN_PROMPT = 100
        if len(all_neg_texts) > MAX_NEG_IN_PROMPT:
            import random
            sample = set()
            for cluster in clusters:
                if cluster:
                    sample.add(cluster[0])
            remaining = [e for e in all_neg_texts if e not in sample]
            need = MAX_NEG_IN_PROMPT - len(sample)
            if need > 0 and remaining:
                sample.update(random.sample(remaining, min(need, len(remaining))))
            existing_sample = list(sample)
            trunc_note = "\n  (showing {} of {} — generate DIFFERENT from ALL {})".format(
                len(existing_sample), len(all_neg_texts), len(all_neg_texts))
        else:
            existing_sample = all_neg_texts
            trunc_note = ""
        existing_text = "\n".join("  - \"{}\"".format(s) for s in existing_sample) + trunc_note

        print("      Requesting {} diverse negatives...".format(gap_n))

        div_prompt = (
            rc["safety_context"] +
            'PROPOSITION (harmful intent to detect): "{}"\n\n'
            'POSITIVE ANCHORS (flagged):\n{}\n\n'
            'EXISTING NEGATIVE ANCHORS (do NOT repeat or rephrase):\n{}\n\n'
            'CLUSTER ANALYSIS (groups of negatives too similar to each other):\n{}\n\n'
            'Generate {} NEW negative anchor {} that are SEMANTICALLY DIFFERENT '
            'from all existing ones.\n\n'
            'Distribute across categories:\n{}\n\n'
            'Focus on DIFFERENT phrasings, angles, and vocabulary NOT yet covered. '
            'Avoid the clusters above — find genuinely new patterns.\n\n'
            'RULES:\n'
            '- Use domain vocabulary but with clearly benign intent\n'
            '- Do NOT rephrase existing negatives — find new angles\n'
            '- Mix very short (3-7 words), medium, and longer examples\n'
            '{}\n\n'
            'Output ONLY valid JSON:\n'
            '{{"categories": {{{}}}}}'
        ).format(
            proposition, pos_sample_text, existing_text,
            cluster_desc or "  (no major clusters found)",
            gap_n, rc["example_noun_short"],
            "\n".join("  {}. {}".format(i + 1, c) for i, c in enumerate(neg_categories)),
            rc["neg_style"],
            ", ".join('"{}": ["ex1", ...]'.format(c) for c in neg_categories)
        )

        result = call_llm(config, neg_system, div_prompt)

        try:
            if isinstance(result, dict):
                parsed = result
            else:
                cleaned = str(result).strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                parsed = json.loads(cleaned)

            if "categories" in parsed:
                parsed = parsed["categories"]

            new_count = 0
            if isinstance(parsed, dict):
                for cat, examples in parsed.items():
                    if isinstance(examples, list):
                        for ex in examples:
                            text = str(ex).strip()
                            if text and text not in all_neg_texts:
                                all_neg_texts.append(text)
                                all_neg_cats.append(cat)
                                new_count += 1
            elif isinstance(parsed, list):
                for i, ex in enumerate(parsed):
                    text = str(ex).strip()
                    if text and text not in all_neg_texts:
                        all_neg_texts.append(text)
                        all_neg_cats.append(neg_categories[i % len(neg_categories)])
                        new_count += 1

            print("      +{} new (total pool: {})".format(new_count, len(all_neg_texts)))

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print("      {}Round {}: parse error: {}{}".format(YELLOW, rnd, e, RESET))
            continue

    if not all_neg_texts:
        print("    {}WARNING: No negative anchors generated{}".format(YELLOW, RESET))
        return {}

    # =====================================================================
    # Step 4: Remove negatives too similar to positives (> 0.85)
    # =====================================================================
    print("\n    {}Filtering:{} Removing negatives too close to positives...".format(
        BOLD, RESET))

    prop_embedding = embed_texts(emb_model, [proposition])[0]

    pos_texts = []
    for examples in positive_anchors_dict.values():
        pos_texts.extend(examples)
    pos_embeddings = embed_texts(emb_model, pos_texts)
    neg_embeddings = embed_texts(emb_model, all_neg_texts)

    from sentence_transformers.util import cos_sim
    sims_to_pos = cos_sim(neg_embeddings, pos_embeddings)
    max_sim_to_pos = sims_to_pos.max(dim=1).values.tolist()

    filtered_texts = []
    filtered_cats = []
    removed_pos = 0
    for i, (text, cat) in enumerate(zip(all_neg_texts, all_neg_cats)):
        if max_sim_to_pos[i] > 0.85:
            removed_pos += 1
            continue
        filtered_texts.append(text)
        filtered_cats.append(cat)

    if removed_pos > 0:
        print("      Removed {} negatives too similar to positives (cosine > 0.85)".format(
            removed_pos))
    print("      Pool: {} → Filtered: {}".format(len(all_neg_texts), len(filtered_texts)))

    # =====================================================================
    # Step 5: MMR selection for diversity
    # =====================================================================
    if len(filtered_texts) > target_n:
        print("\n    {}MMR Selection:{} Picking {} most diverse from pool of {}...".format(
            BOLD, RESET, target_n, len(filtered_texts)))

        filt_embs = embed_texts(emb_model, filtered_texts)
        use_anneal = config.get("anchors", "mmr_anneal", fallback="true").strip().lower() == "true"
        selected_indices = cluster_constrained_mmr(
            filt_embs, filtered_texts, filtered_cats, prop_embedding,
            target_n, lambda_param=0.4, anneal=use_anneal)
        final_texts = [filtered_texts[i] for i in selected_indices]
        final_cats = [filtered_cats[i] for i in selected_indices]

        # Report diversity improvement
        sel_embs = filt_embs[selected_indices]
        sel_sim = cosine_sim_matrix(sel_embs)
        np.fill_diagonal(sel_sim, 0)
        pool_sim = cosine_sim_matrix(filt_embs)
        np.fill_diagonal(pool_sim, 0)
        print("      Pool avg pairwise similarity:  {:.4f}".format(pool_sim.mean()))
        print("      Selected avg pairwise sim:     {:.4f}".format(sel_sim.mean()))
        print("      Diversity improvement:         {:.1f}%".format(
            (1 - sel_sim.mean() / (pool_sim.mean() + 1e-10)) * 100))

        # Report directional spread around proposition
        sel_dirs = sel_embs - prop_embedding.reshape(1, -1)
        sel_dir_norms = np.linalg.norm(sel_dirs, axis=1, keepdims=True) + 1e-10
        sel_dirs_normed = sel_dirs / sel_dir_norms
        dir_sim = sel_dirs_normed @ sel_dirs_normed.T
        np.fill_diagonal(dir_sim, 0)
        print("      Directional spread (avg):      {:.4f} (lower = more spread)".format(
            dir_sim.mean()))

        # Per-category distribution
        pool_cats = {}
        for cat in filtered_cats:
            pool_cats[cat] = pool_cats.get(cat, 0) + 1
        sel_cats = {}
        for cat in final_cats:
            sel_cats[cat] = sel_cats.get(cat, 0) + 1
        print("      Per-category allocation:")
        for cat in sorted(sel_cats, key=lambda c: sel_cats[c], reverse=True):
            print("        {}: {} selected from {} pool".format(
                cat[:50], sel_cats[cat], pool_cats.get(cat, 0)))
    else:
        final_texts = filtered_texts
        final_cats = filtered_cats

    # Build category dict
    neg_dict = {}
    for text, cat in zip(final_texts, final_cats):
        if cat not in neg_dict:
            neg_dict[cat] = []
        neg_dict[cat].append(text)

    total = sum(len(v) for v in neg_dict.values())
    print("    Final: {} negative anchors across {} categories".format(
        total, len(neg_dict)))

    # Report cosine stats for final set (info only)
    if final_texts:
        final_embs = embed_texts(emb_model, final_texts)
        final_prop_sims = cos_sim(final_embs, prop_embedding.reshape(1, -1))[:, 0].tolist()
        print("    Cosine to proposition: min={:.3f}  avg={:.3f}  max={:.3f}".format(
            min(final_prop_sims), sum(final_prop_sims) / len(final_prop_sims),
            max(final_prop_sims)))

    # --- Subtopic coverage audit and gap-fill for negatives (iterative) ---
    if has_neg_subtopics:
        neg_all_texts = []
        for cat, examples in neg_dict.items():
            neg_all_texts.extend(examples)

        # Fixed per-subtopic target based on ORIGINAL target_n, not current total.
        per_subtopic_target = max(5, target_n // len(subtopics))

        forced_assignments = {}

        MAX_GAP_FILL_ROUNDS = 3
        for gf_round in range(MAX_GAP_FILL_ROUNDS):
            label = "negative anchors" if gf_round == 0 else "negative anchors (after gap-fill round {})".format(gf_round)
            coverage, assignments = audit_subtopic_coverage(
                neg_all_texts, subtopics, emb_model, label=label,
                forced_assignments=forced_assignments)

            # Check which subtopics are below the FIXED target
            underrep = [i for i, c in coverage.items() if c < per_subtopic_target]

            if not underrep:
                print("    {}All subtopics meet target ({} per subtopic) ✓{}".format(
                    GREEN, per_subtopic_target, RESET))
                break

            max_shortfall = max(per_subtopic_target - coverage[i] for i in underrep)

            print("\n    {}Subtopic gap-fill (round {}):{} {} underrepresented subtopic(s), "
                  "target {} negatives each (max shortfall {})...".format(
                      BOLD, gf_round + 1, RESET, len(underrep), per_subtopic_target, max_shortfall))

            gap_texts, gap_cats, gap_st_indices = generate_subtopic_gap_fill(
                proposition, subtopics, underrep, neg_all_texts, max_shortfall,
                config, role=role, anchor_type="negative", emb_model=emb_model)

            if not gap_texts:
                print("    {}No new gap-fill negatives generated, stopping.{}".format(
                    YELLOW, RESET))
                break

            for text, cat, st_idx in zip(gap_texts, gap_cats, gap_st_indices):
                if cat not in neg_dict:
                    neg_dict[cat] = []
                neg_dict[cat].append(text)
                forced_assignments[text] = st_idx

            total = sum(len(v) for v in neg_dict.values())
            print("    {} Added {} gap-fill negatives (new total: {}){}".format(
                GREEN, len(gap_texts), total, RESET))

            neg_all_texts.extend(gap_texts)
        else:
            audit_subtopic_coverage(
                neg_all_texts, subtopics, emb_model,
                label="negative anchors (after {} gap-fill rounds)".format(MAX_GAP_FILL_ROUNDS),
                forced_assignments=forced_assignments)

    return neg_dict
