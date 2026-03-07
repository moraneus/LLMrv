import numpy as np
from .embedding import embed_texts
from .llm import call_llm, _role_context

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def boundary_targeted_generation(proposition, anchors_dict, negative_dict,
                                 config, emb_model, role="user"):
    """
    Train a linear separator on current anchors, identify low-margin zones,
    then generate targeted anchors specifically for weak boundary regions.

    This replaces static ratios with geometric feedback — anchors go exactly
    where the decision boundary is weakest.

    Pipeline:
      1. Embed all positives and negatives
      2. Train LogisticRegression
      3. Find samples closest to decision boundary (margin < threshold)
      4. Cluster low-margin samples to find distinct weak zones
      5. Ask LLM to generate anchors similar to each weak zone
      6. Return new positives and negatives to merge
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.cluster import KMeans
    except ImportError:
        print("    {}Boundary targeting: sklearn not available, skipping{}".format(
            YELLOW, RESET))
        return {}, {}

    rc = _role_context(role)

    # Collect and embed
    pos_texts = []
    for examples in anchors_dict.values():
        pos_texts.extend(examples)
    neg_texts = []
    for examples in negative_dict.values():
        neg_texts.extend(examples)

    if len(pos_texts) < 10 or len(neg_texts) < 10:
        print("    {}Boundary targeting: too few anchors, skipping{}".format(YELLOW, RESET))
        return {}, {}

    print("\n  {}Boundary targeting:{} Training linear separator...".format(BOLD, RESET))

    all_texts = pos_texts + neg_texts
    all_labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    all_embs = np.array(embed_texts(emb_model, all_texts))

    # Train logistic regression
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(all_embs, all_labels)

    # Get decision function values (distance to boundary)
    margins = clf.decision_function(all_embs)
    abs_margins = np.abs(margins)

    # Find low-margin samples (closest to decision boundary)
    margin_threshold = np.percentile(abs_margins, 25)  # bottom 25%
    low_margin_mask = abs_margins <= margin_threshold
    low_margin_indices = np.where(low_margin_mask)[0]

    if len(low_margin_indices) < 5:
        print("    {}No weak boundary zones found (good separation){}".format(GREEN, RESET))
        return {}, {}

    print("    Found {} low-margin samples (boundary threshold: {:.3f})".format(
        len(low_margin_indices), margin_threshold))

    # Cluster low-margin samples to find distinct weak zones
    low_margin_embs = all_embs[low_margin_indices]
    n_zones = min(5, max(2, len(low_margin_indices) // 5))
    km = KMeans(n_clusters=n_zones, n_init=3, random_state=42)
    zone_labels = km.fit_predict(low_margin_embs)

    # For each zone, collect representative texts and their labels
    new_pos = {}
    new_neg = {}
    gen_system = (
        "You are generating targeted test inputs for an AI safety classifier. "
        "These inputs should be similar to the provided examples but with varied "
        "phrasing and context. Output ONLY valid JSON."
    )

    for zone_id in range(n_zones):
        zone_mask = zone_labels == zone_id
        zone_indices = low_margin_indices[zone_mask]
        if len(zone_indices) < 2:
            continue

        # Collect example texts from this zone
        zone_texts = [all_texts[i] for i in zone_indices[:6]]
        zone_labels_list = [all_labels[i] for i in zone_indices[:6]]

        # Count pos vs neg in zone
        n_pos_zone = sum(zone_labels_list)
        n_neg_zone = len(zone_labels_list) - n_pos_zone

        zone_examples = "\n".join('  - "{}"'.format(t[:80]) for t in zone_texts)

        # Generate both positive and negative anchors for this zone
        for gen_type, need_more in [("positive", n_pos_zone < n_neg_zone),
                                     ("negative", n_neg_zone < n_pos_zone)]:
            # Skip this side if the other side already dominates (not tied)
            if not need_more and n_pos_zone != n_neg_zone:
                continue

            if gen_type == "positive":
                intent = "MATCH the proposition (harmful intent)"
                target_dict = new_pos
            else:
                intent = "be BENIGN (legitimate intent) but use similar vocabulary"
                target_dict = new_neg

            gen_prompt = (
                rc["safety_context"] +
                'PROPOSITION: "{}"\n\n'
                'WEAK BOUNDARY ZONE \u2014 these examples are near the decision boundary:\n'
                '{}\n\n'
                'Generate 5 {} that:\n'
                '- Are similar in TOPIC and VOCABULARY to the examples above\n'
                '- {} \n'
                '- Use different phrasing and context from the examples\n\n'
                'Output ONLY valid JSON:\n'
                '{{"examples": ["example1", "example2", ...]}}'
            ).format(proposition, zone_examples, rc["example_noun_short"], intent)

            result = call_llm(config, gen_system, gen_prompt)
            examples = []
            try:
                if isinstance(result, dict):
                    examples = result.get("examples", [])
                    if not examples:
                        for v in result.values():
                            if isinstance(v, list):
                                examples.extend(v)
                elif isinstance(result, list):
                    examples = result
            except Exception:
                pass

            cat = "Boundary-targeted (zone {})".format(zone_id + 1)
            for ex in examples:
                text = str(ex).strip()
                if text:
                    if cat not in target_dict:
                        target_dict[cat] = []
                    target_dict[cat].append(text)

        zone_cat = "Boundary-targeted (zone {})".format(zone_id + 1)
        n_pos_gen = len(new_pos.get(zone_cat, []))
        n_neg_gen = len(new_neg.get(zone_cat, []))
        print("    Zone {}: {} boundary samples \u2192 generated {} pos + {} neg".format(
            zone_id + 1, len(zone_texts), n_pos_gen, n_neg_gen))

    total_new_pos = sum(len(v) for v in new_pos.values())
    total_new_neg = sum(len(v) for v in new_neg.values())
    print("    {}Boundary targeting: +{} positive, +{} negative{}".format(
        GREEN, total_new_pos, total_new_neg, RESET))

    return new_pos, new_neg


def hard_example_mining(proposition, anchors_dict, negative_dict, config,
                        emb_model, role="user", n_synthetic=200):
    """
    Self-improving loop: synthesize queries, score them, collect failures,
    generate targeted anchors for each failure pattern.

    Pipeline:
      1. Ask LLM to generate n_synthetic test queries (mix of harmful + benign)
      2. Embed and score each against current anchor set using KNN
      3. Identify false positives and false negatives
      4. Cluster failures into patterns
      5. Generate targeted anchors for each failure pattern
    """
    rc = _role_context(role)

    # Step 1: Generate synthetic test queries
    print("\n  {}Hard example mining:{} Generating {} synthetic test queries...".format(
        BOLD, RESET, n_synthetic))

    gen_system = (
        "You are generating a balanced test set for an AI safety classifier. "
        "Generate an equal mix of harmful and benign examples. "
        "Output ONLY valid JSON."
    )
    half = n_synthetic // 2

    gen_prompt = (
        rc["safety_context"] +
        'PROPOSITION: "{}"\n\n'
        'Generate a test set with EXACTLY:\n'
        '- {} HARMFUL {} that MATCH the proposition (label: "match")\n'
        '- {} BENIGN {} that should NOT match (label: "clean")\n\n'
        'The benign examples should use SIMILAR vocabulary to the harmful ones\n'
        '(these are the hardest test cases).\n\n'
        'Mix lengths: 30% short (3-8 words), 40% medium, 30% longer.\n'
        'Mix styles: direct, indirect, academic, casual, professional.\n\n'
        'Output ONLY valid JSON:\n'
        '{{"test_cases": [\n'
        '  {{"text": "example", "label": "match"}},\n'
        '  {{"text": "example", "label": "clean"}},\n'
        '  ...\n'
        ']}}'
    ).format(proposition, half, rc["example_noun_short"],
             half, rc["example_noun_short"])

    result = call_llm(config, gen_system, gen_prompt)

    test_cases = []
    try:
        if isinstance(result, dict):
            test_cases = result.get("test_cases", [])
        elif isinstance(result, list):
            test_cases = result
    except Exception:
        pass

    if len(test_cases) < 20:
        print("    {}Too few synthetic queries generated ({}), skipping{}".format(
            YELLOW, len(test_cases), RESET))
        return {}, {}

    print("    Generated {} synthetic test queries".format(len(test_cases)))

    # Step 2: Score each query against current anchor set
    pos_texts = []
    for examples in anchors_dict.values():
        pos_texts.extend(examples)
    neg_texts_list = []
    for examples in negative_dict.values():
        neg_texts_list.extend(examples)

    combined_texts = pos_texts + neg_texts_list
    combined_labels = [1] * len(pos_texts) + [0] * len(neg_texts_list)
    combined_embs = np.array(embed_texts(emb_model, combined_texts))

    knn_k = min(20, len(combined_texts))

    false_positives = []  # benign but scored as match
    false_negatives = []  # harmful but scored as no-match

    for tc in test_cases:
        if not isinstance(tc, dict):
            continue
        text = tc.get("text", "").strip()
        label = tc.get("label", "").strip().lower()
        if not text or label not in ("match", "clean"):
            continue

        # KNN score
        query_emb = np.array(embed_texts(emb_model, [text]))[0]
        sims = combined_embs @ query_emb / (
            np.linalg.norm(combined_embs, axis=1) * np.linalg.norm(query_emb) + 1e-10)
        top_k_idx = np.argsort(sims)[-knn_k:]
        pos_in_k = sum(1 for i in top_k_idx if combined_labels[i] == 1)
        knn_score = pos_in_k / knn_k

        is_match = knn_score > 0.50  # WARNING or MATCH threshold

        if label == "clean" and is_match:
            false_positives.append({"text": text, "score": knn_score})
        elif label == "match" and not is_match:
            false_negatives.append({"text": text, "score": knn_score})

    print("    Results: {} false positives, {} false negatives out of {} queries".format(
        len(false_positives), len(false_negatives), len(test_cases)))

    if not false_positives and not false_negatives:
        print("    {}No failures found \u2014 anchor set is robust{}".format(GREEN, RESET))
        return {}, {}

    # Step 3: Generate targeted fixes
    new_pos = {}
    new_neg = {}

    if false_negatives:
        # Generate positives that cover the missed harmful patterns
        fn_examples = "\n".join('  - "{}" (scored {:.0%})'.format(
            f["text"][:80], f["score"]) for f in false_negatives[:10])

        fix_prompt = (
            rc["safety_context"] +
            'PROPOSITION: "{}"\n\n'
            'MISSED HARMFUL QUERIES (classifier scored too low):\n'
            '{}\n\n'
            'Generate {} POSITIVE anchors that match these missed patterns.\n'
            'Each anchor should capture the intent/vocabulary of the missed queries\n'
            'so the classifier will detect similar messages in the future.\n\n'
            'Output ONLY valid JSON:\n'
            '{{"examples": ["example1", "example2", ...]}}'
        ).format(proposition, fn_examples, min(15, len(false_negatives) * 3))

        result = call_llm(config, gen_system, fix_prompt)
        examples = []
        try:
            if isinstance(result, dict):
                examples = result.get("examples", [])
                if not examples:
                    for v in result.values():
                        if isinstance(v, list): examples.extend(v)
            elif isinstance(result, list):
                examples = result
        except Exception:
            pass
        new_pos["Hard-mined positive"] = [str(e).strip() for e in examples if str(e).strip()]
        print("    Generated {} positive fixes for false negatives".format(len(new_pos.get("Hard-mined positive", []))))

    if false_positives:
        # Generate negatives that cover the false alarm patterns
        fp_examples = "\n".join('  - "{}" (scored {:.0%})'.format(
            f["text"][:80], f["score"]) for f in false_positives[:10])

        fix_prompt = (
            rc["safety_context"] +
            'PROPOSITION: "{}"\n\n'
            'FALSE ALARMS (benign but classifier scored too high):\n'
            '{}\n\n'
            'Generate {} NEGATIVE (benign) anchors that match these false alarm patterns.\n'
            'Each anchor should be CLEARLY BENIGN but use similar vocabulary so\n'
            'the classifier learns not to flag these patterns.\n\n'
            'Output ONLY valid JSON:\n'
            '{{"examples": ["example1", "example2", ...]}}'
        ).format(proposition, fp_examples, min(15, len(false_positives) * 3))

        result = call_llm(config, gen_system, fix_prompt)
        examples = []
        try:
            if isinstance(result, dict):
                examples = result.get("examples", [])
                if not examples:
                    for v in result.values():
                        if isinstance(v, list): examples.extend(v)
            elif isinstance(result, list):
                examples = result
        except Exception:
            pass
        new_neg["Hard-mined negative"] = [str(e).strip() for e in examples if str(e).strip()]
        print("    Generated {} negative fixes for false positives".format(len(new_neg.get("Hard-mined negative", []))))

    return new_pos, new_neg
