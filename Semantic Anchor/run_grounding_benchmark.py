#!/usr/bin/env python3
"""
Batch generation + evaluation runner for grounding_dataset.csv.

For each proposition (p1..p10 by default), this script:
1. Calls semantic_anchor_generator internals to generate anchors.
2. Writes semantic_anchor_<proposition_id>.py.
3. Evaluates the proposition's labeled examples from the dataset.
4. Prints per-example predictions and aggregate accuracy.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

from semantic_anchor_generator import (
    generate_anchors,
    generate_script,
    load_config,
    parse_categories,
)


REQUIRED_COLUMNS = {
    "proposition_id",
    "role",
    "proposition",
    "example_type",
    "example_id",
    "text",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_header(name):
    return (name or "").replace("\ufeff", "").strip().lower()


def proposition_sort_key(pid):
    match = re.fullmatch(r"p(\d+)", pid.lower())
    if match:
        return (0, int(match.group(1)))
    return (1, pid)


def load_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Dataset has no header row.")
        reader.fieldnames = [normalize_header(h) for h in reader.fieldnames]
        missing = REQUIRED_COLUMNS.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(
                "Dataset is missing required columns: {}".format(", ".join(sorted(missing)))
            )
        for row in reader:
            normalized = {normalize_header(k): (v or "").strip() for k, v in row.items()}
            ex_type = normalized["example_type"].lower()
            if ex_type not in {"positive", "negative"}:
                raise ValueError(
                    "Invalid example_type '{}' for example_id '{}'.".format(
                        normalized["example_type"], normalized.get("example_id", "")
                    )
                )
            rows.append(normalized)
    if not rows:
        raise ValueError("Dataset has no rows.")
    return rows


def group_by_proposition(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["proposition_id"]].append(row)
    return grouped


def flatten_anchors(anchors_dict):
    texts = []
    cats = []
    for cat, examples in anchors_dict.items():
        for ex in examples:
            texts.append(ex)
            cats.append(cat)
    return texts, cats


def classify(score, match_threshold):
    return "positive" if score >= match_threshold else "negative"


def score_message(model, text, anchor_embeddings):
    from sentence_transformers.util import cos_sim

    message_emb = model.encode(text, show_progress_bar=False)
    similarities = cos_sim(message_emb, anchor_embeddings)[0].tolist()
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return similarities[best_idx], best_idx


def safe_file_label(text):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic anchor evaluators for each proposition in a grounding "
            "dataset and report classification accuracy."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "grounding_dataset.csv"),
        help="Path to grounding dataset CSV.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "config.ini"),
        help="Path to config.ini used by semantic_anchor_generator.",
    )
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=None,
        help="Number of anchors to generate per proposition (clamped to 20-100).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        help="Decision threshold for positive classification.",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=None,
        help="Optional warning threshold shown in output.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence-transformers model for scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SCRIPT_DIR,
        help="Directory where semantic_anchor_<proposition_id>.py files are written.",
    )
    parser.add_argument(
        "--propositions",
        nargs="+",
        default=None,
        help="Optional subset (e.g., p1 p2 p3). Defaults to all in dataset.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Optional path to write per-example evaluation results as CSV.",
    )
    args = parser.parse_args()

    default_config_path = os.path.join(SCRIPT_DIR, "config.ini")
    if args.config == default_config_path and not os.path.exists(args.config):
        cwd_config = os.path.join(os.getcwd(), "config.ini")
        if os.path.exists(cwd_config):
            args.config = cwd_config

    config = load_config(args.config)
    rows = load_dataset(args.dataset)
    grouped = group_by_proposition(rows)

    if args.propositions:
        requested = {p.strip() for p in args.propositions}
        missing = sorted(requested.difference(grouped.keys()), key=proposition_sort_key)
        if missing:
            print("ERROR: proposition_id not found in dataset: {}".format(", ".join(missing)))
            sys.exit(1)
        proposition_ids = sorted(requested, key=proposition_sort_key)
    else:
        proposition_ids = sorted(grouped.keys(), key=proposition_sort_key)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    categories = parse_categories(config)
    if args.num_examples is not None:
        num_examples = max(20, min(100, args.num_examples))
    else:
        num_examples = max(20, min(100, int(config.get("anchors", "num_examples"))))

    match_threshold = (
        args.match_threshold
        if args.match_threshold is not None
        else float(config.get("thresholds", "match_threshold"))
    )
    warning_threshold = (
        args.warning_threshold
        if args.warning_threshold is not None
        else float(config.get("thresholds", "warning_threshold"))
    )
    embedding_model = (
        args.embedding_model
        if args.embedding_model is not None
        else config.get("anchors", "embedding_model")
    )

    print("== Grounding Benchmark Configuration ==")
    print("Dataset:            {}".format(args.dataset))
    print("Config:             {}".format(args.config))
    print("Propositions:       {}".format(", ".join(proposition_ids)))
    print("Anchors each:       {}".format(num_examples))
    print("Match threshold:    {}".format(match_threshold))
    print("Warning threshold:  {}".format(warning_threshold))
    print("Embedding model:    {}".format(embedding_model))
    print("Output dir:         {}".format(output_dir))
    print("")

    proposition_assets = {}
    failed_propositions = []
    for proposition_id in proposition_ids:
        records = grouped[proposition_id]
        proposition_texts = {r["proposition"] for r in records}
        if len(proposition_texts) != 1:
            print(
                "[{}] ERROR: multiple proposition texts found. Skipping.".format(
                    proposition_id
                )
            )
            failed_propositions.append(proposition_id)
            continue
        proposition = next(iter(proposition_texts))

        print("[{}] Generating anchors...".format(proposition_id))
        try:
            anchors_dict = generate_anchors(proposition, categories, num_examples, config)
            total_anchors = sum(len(v) for v in anchors_dict.values())

            script_name = safe_file_label(proposition_id)
            script_content = generate_script(
                script_name,
                proposition,
                anchors_dict,
                match_threshold,
                warning_threshold,
            )
            script_filename = "semantic_anchor_{}.py".format(script_name)
            script_path = os.path.join(output_dir, script_filename)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
        except SystemExit as e:
            print(
                "[{}] ERROR: generation failed (SystemExit: {}). Skipping.".format(
                    proposition_id, e
                )
            )
            failed_propositions.append(proposition_id)
            continue
        except Exception as e:
            print("[{}] ERROR: generation failed ({}). Skipping.".format(proposition_id, e))
            failed_propositions.append(proposition_id)
            continue

        print(
            "[{}] Wrote {} ({} anchors)".format(
                proposition_id, script_path, total_anchors
            )
        )

        anchor_texts, anchor_categories = flatten_anchors(anchors_dict)
        if not anchor_texts:
            print("[{}] ERROR: No anchors were generated. Skipping.".format(proposition_id))
            failed_propositions.append(proposition_id)
            continue
        proposition_assets[proposition_id] = {
            "records": records,
            "anchor_texts": anchor_texts,
            "anchor_categories": anchor_categories,
        }

    successful_ids = [pid for pid in proposition_ids if pid in proposition_assets]
    if failed_propositions:
        print(
            "\nSkipped propositions due to generation failures: {}".format(
                ", ".join(failed_propositions)
            )
        )
    if not successful_ids:
        print("\nERROR: No propositions were generated successfully. Nothing to evaluate.")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed in this interpreter.")
        print("Run: python -m pip install sentence-transformers")
        sys.exit(1)

    print("\nLoading embedding model: {} ...".format(embedding_model))
    model = SentenceTransformer(embedding_model)

    all_results = []
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "correct": 0, "count": 0}
    per_prop_accuracy = {}

    print("\n== Per-example Results ==")
    for proposition_id in successful_ids:
        records = proposition_assets[proposition_id]["records"]
        anchor_texts = proposition_assets[proposition_id]["anchor_texts"]
        anchor_categories = proposition_assets[proposition_id]["anchor_categories"]

        anchor_embeddings = model.encode(anchor_texts, show_progress_bar=False)

        prop_correct = 0
        print("\n[{}]".format(proposition_id))
        for row in records:
            score, best_idx = score_message(model, row["text"], anchor_embeddings)
            predicted = classify(score, match_threshold)
            truth = row["example_type"].lower()
            is_correct = int(predicted == truth)
            prop_correct += is_correct
            totals["correct"] += is_correct
            totals["count"] += 1

            if truth == "positive" and predicted == "positive":
                totals["tp"] += 1
            elif truth == "negative" and predicted == "negative":
                totals["tn"] += 1
            elif truth == "negative" and predicted == "positive":
                totals["fp"] += 1
            elif truth == "positive" and predicted == "negative":
                totals["fn"] += 1

            zone = "match"
            if score < match_threshold:
                zone = "warning" if score >= warning_threshold else "clean"

            nearest_anchor = anchor_texts[best_idx]
            nearest_category = anchor_categories[best_idx]

            print(
                "  {eid:12s} truth={truth:8s} pred={pred:8s} score={score:.4f} zone={zone:7s} top_cat={cat} correct={ok}".format(
                    eid=row["example_id"],
                    truth=truth,
                    pred=predicted,
                    score=score,
                    zone=zone,
                    cat=nearest_category,
                    ok="yes" if is_correct else "no",
                )
            )

            all_results.append(
                {
                    "proposition_id": proposition_id,
                    "example_id": row["example_id"],
                    "truth": truth,
                    "predicted": predicted,
                    "score": "{:.6f}".format(score),
                    "zone": zone,
                    "correct": str(bool(is_correct)),
                    "proposition": row["proposition"],
                    "nearest_anchor": nearest_anchor,
                    "text": row["text"],
                }
            )

        per_prop_accuracy[proposition_id] = prop_correct / len(records)
        print(
            "  -> proposition accuracy: {}/{} = {:.2%}".format(
                prop_correct, len(records), per_prop_accuracy[proposition_id]
            )
        )

    overall_accuracy = totals["correct"] / totals["count"] if totals["count"] else 0.0

    print("\n== Accuracy Summary ==")
    for proposition_id in successful_ids:
        print("  {}: {:.2%}".format(proposition_id, per_prop_accuracy[proposition_id]))
    print("")
    print("  Overall accuracy: {:.2%}".format(overall_accuracy))
    print(
        "  Confusion matrix: TP={} TN={} FP={} FN={}".format(
            totals["tp"], totals["tn"], totals["fp"], totals["fn"]
        )
    )

    if args.results_csv:
        out_fields = [
            "proposition_id",
            "example_id",
            "truth",
            "predicted",
            "score",
            "zone",
            "correct",
            "proposition",
            "nearest_anchor",
            "text",
        ]
        with open(args.results_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(all_results)
        print("\nSaved per-example results CSV to: {}".format(args.results_csv))


if __name__ == "__main__":
    main()
