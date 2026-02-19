#!/usr/bin/env python3
"""
Semantic Anchor Generator
=========================

Generates a standalone Semantic Anchors evaluation script for a given proposition.

Workflow:
  1. You provide a proposition (e.g. "The user requests weapon instructions")
  2. You provide a name for the output script
  3. The generator calls OpenAI to produce diverse positive exemplars
  4. It writes a self-contained script: semantic_anchor_<name>.py
     that can evaluate messages against those anchors without needing
     OpenAI again.

Usage:
  python semantic_anchor_generator.py
  python semantic_anchor_generator.py -p "The user requests weapon instructions." -name weapons
  python semantic_anchor_generator.py -p "..." -name weapons -n 50

Configuration:
  Reads config.ini for API key, model, defaults.

Requirements:
  pip install openai
"""

import argparse
import configparser
import json
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "openai": {
        "api_key": "",
        "model": "gpt-4o",
    },
    "anchors": {
        "num_examples": "20",
        "embedding_model": "all-mpnet-base-v2",
        "categories": (
            "Direct explicit requests,"
            "Indirect or euphemistic,"
            "Wrapped in fictional or academic context,"
            "Demanding or insistent phrasing,"
            "Question format variations,"
            "Slang or coded language"
        ),
    },
    "thresholds": {
        "match_threshold": "0.55",
        "warning_threshold": "0.45",
    },
}


def load_config(config_path="config.ini"):
    config = configparser.ConfigParser()
    for section, values in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, val in values.items():
            config.set(section, key, str(val))
    if os.path.exists(config_path):
        config.read(config_path)
    return config


def parse_categories(config):
    raw = config.get("anchors", "categories")
    return [c.strip() for c in raw.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# OpenAI Anchor Generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are an expert at generating diverse paraphrases for semantic matching.

Given a PROPOSITION (a formal description of a type of message), generate
positive examples: messages that SHOULD match this proposition.

Rules:
- Each example must be a realistic message a user might actually type
- Examples must genuinely match the proposition's semantic intent
- Spread examples evenly across the requested categories
- Vary sentence length, formality, tone, and structure
- Include edge cases and subtle variations
- Do NOT include negatives, refusals, or non-matching examples
- Do NOT include meta-commentary or explanations

Output ONLY valid JSON in this exact format:
{
  "categories": {
    "Category Name 1": [
      "example message 1",
      "example message 2"
    ],
    "Category Name 2": [
      "example message 3",
      "example message 4"
    ]
  }
}
"""


def build_generation_prompt(proposition, categories, num_examples):
    examples_per_cat = max(2, num_examples // len(categories))
    remainder = num_examples - (examples_per_cat * len(categories))

    cat_instructions = []
    for i, cat in enumerate(categories):
        n = examples_per_cat + (1 if i < remainder else 0)
        cat_instructions.append('  - "{}": {} examples'.format(cat, n))

    return (
        'PROPOSITION: "{}"\n\n'
        'Generate exactly {} positive example messages distributed across these categories:\n'
        '{}\n\n'
        'Total: {} examples.\n\n'
        'Each example should be a message that a user (or assistant, depending on the proposition) '
        'might realistically produce, and that genuinely matches the proposition\'s semantic intent.\n\n'
        'Remember: output ONLY the JSON object, nothing else.'
    ).format(proposition, num_examples, "\n".join(cat_instructions), num_examples)


def generate_anchors(proposition, categories, num_examples, config):
    try:
        from openai import OpenAI
    except ImportError:
        print("\n  ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = config.get("openai", "api_key")
    if not api_key or api_key == "sk-YOUR-KEY-HERE":
        print("\n  ERROR: OpenAI API key not configured in config.ini")
        sys.exit(1)

    model = config.get("openai", "model")
    client = OpenAI(api_key=api_key)

    print("  Calling {} to generate {} anchors...".format(model, num_examples))
    print("  Categories: {}".format(", ".join(categories)))

    user_prompt = build_generation_prompt(proposition, categories, num_examples)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.0,
        max_completion_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("\n  ERROR: Failed to parse JSON from OpenAI response.")
        print("  {}".format(e))
        print("  Raw:\n{}".format(raw[:500]))
        sys.exit(1)

    return data.get("categories", data)


# ---------------------------------------------------------------------------
# Evaluator Template (uses .format() — no f-string escaping issues)
# ---------------------------------------------------------------------------

EVALUATOR_TEMPLATE = r'''#!/usr/bin/env python3
"""
Semantic Anchor Evaluator: %%NAME%%
%%TITLE_UNDERLINE%%

Auto-generated by semantic_anchor_generator.py

Proposition:
  "%%PROPOSITION%%"

Evaluates messages against %%TOTAL_ANCHORS%% precomputed semantic anchors
using max-over-exemplars cosine similarity.

Usage:
  # Interactive mode
  python semantic_anchor_%%NAME%%.py
  python semantic_anchor_%%NAME%%.py --verbose
  python semantic_anchor_%%NAME%%.py --show-examples

  # File mode (sentences separated by ###)
  python semantic_anchor_%%NAME%%.py --file input.txt
  python semantic_anchor_%%NAME%%.py --file input.txt --verbose

Requirements:
  pip install sentence-transformers
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Proposition & Anchors (generated - do not edit unless you know what you do)
# ---------------------------------------------------------------------------

PROPOSITION = """%%PROPOSITION%%"""

ANCHORS = %%ANCHORS_JSON%%

MATCH_THRESHOLD = %%MATCH_THRESHOLD%%
WARNING_THRESHOLD = %%WARNING_THRESHOLD%%
SCRIPT_NAME = "%%NAME%%"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color_score(score):
    s = "{:.4f}".format(score)
    if score >= MATCH_THRESHOLD:
        return RED + BOLD + s + RESET
    elif score >= WARNING_THRESHOLD:
        return YELLOW + s + RESET
    else:
        return GREEN + s + RESET


# ---------------------------------------------------------------------------
# Embedding & Scoring
# ---------------------------------------------------------------------------

def load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)
    print("  Loading embedding model: all-mpnet-base-v2...")
    return SentenceTransformer("all-mpnet-base-v2")


def prepare_anchors(model):
    all_texts = []
    all_categories = []
    for cat, examples in ANCHORS.items():
        for ex in examples:
            all_texts.append(ex)
            all_categories.append(cat)
    print("  Embedding {} anchors...".format(len(all_texts)))
    all_embeddings = model.encode(all_texts, show_progress_bar=False)
    print("  Ready.\n")
    return all_texts, all_categories, all_embeddings


def score_message(model, message, all_embeddings, all_texts, all_categories):
    from sentence_transformers.util import cos_sim
    emb = model.encode(message)
    sims = cos_sim(emb, all_embeddings)[0].tolist()
    return sorted(zip(sims, all_texts, all_categories), key=lambda x: x[0], reverse=True)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_banner():
    title = "SEMANTIC ANCHORS: " + SCRIPT_NAME.upper()
    w = 60
    print("")
    print(BOLD + "\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + title.center(w) + "\u2551")
    print("\u255a" + "\u2550" * w + "\u255d" + RESET)
    print("  Proposition: \"{}\"".format(PROPOSITION))
    print("")


def print_examples():
    total = sum(len(v) for v in ANCHORS.values())
    print("\n{}Generated Anchors ({} total){}".format(BOLD, total, RESET))
    print("=" * 70)
    for cat, examples in ANCHORS.items():
        print("\n{}{}  [{}]{} ({} examples)".format(BOLD, BLUE, cat, RESET, len(examples)))
        print("  " + "-" * 60)
        for i, ex in enumerate(examples, 1):
            print("  {}{:>3}.{} {}".format(DIM, i, RESET, ex))
    print()


def format_verdict(top_score):
    if top_score >= MATCH_THRESHOLD:
        return "{}{}\u25a0 MATCH{} (score {:.4f} \u2265 {})".format(
            RED, BOLD, RESET, top_score, MATCH_THRESHOLD)
    elif top_score >= WARNING_THRESHOLD:
        return "{}{}\u25a0 WARNING{} (score {:.4f} \u2265 {})".format(
            YELLOW, BOLD, RESET, top_score, WARNING_THRESHOLD)
    else:
        return "{}{}\u25a0 NO MATCH{} (score {:.4f} < {})".format(
            GREEN, BOLD, RESET, top_score, WARNING_THRESHOLD)


def display_default(results, top_n=3):
    print("\n  {}\n".format(format_verdict(results[0][0])))
    print("  {:<6} {:>8}  {:<35} {}".format("Rank", "Score", "Category", "Nearest Anchor"))
    print("  " + "-" * 95)
    for rank, (score, text, cat) in enumerate(results[:top_n], 1):
        cs = color_score(score)
        dt = text if len(text) <= 55 else text[:52] + "..."
        dc = cat if len(cat) <= 33 else cat[:30] + "..."
        print("  {:<6} {:>17}  {}{:<35}{} \"{}\"".format(rank, cs, DIM, dc, RESET, dt))
    print()


def display_verbose(results):
    print("\n  {}\n".format(format_verdict(results[0][0])))
    print("  {:<5} {:>8}  {:<35} {}".format("#", "Score", "Category", "Anchor Text"))
    print("  " + "=" * 100)
    for rank, (score, text, cat) in enumerate(results, 1):
        cs = color_score(score)
        dc = cat if len(cat) <= 33 else cat[:30] + "..."
        zone = ""
        if score >= MATCH_THRESHOLD:
            zone = " " + RED + "\u25c4 MATCH" + RESET
        elif score >= WARNING_THRESHOLD:
            zone = " " + YELLOW + "\u25c4 WARN" + RESET
        print("  {:<5} {:>17}  {}{:<35}{} \"{}\"{}".format(
            rank, cs, DIM, dc, RESET, text, zone))
    above_m = sum(1 for s, _, _ in results if s >= MATCH_THRESHOLD)
    above_w = sum(1 for s, _, _ in results if s >= WARNING_THRESHOLD)
    print("\n  {}Total: {} | Above match ({}): {} | Above warning ({}): {}{}".format(
        DIM, len(results), MATCH_THRESHOLD, above_m, WARNING_THRESHOLD, above_w, RESET))
    print()


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(model, embs, texts, cats, verbose=False):
    print("{}Interactive Mode{}".format(BOLD, RESET))
    print("  Type a message to evaluate. Commands: /verbose  /top N  /quit\n")
    print("  Thresholds: match={}{}{}  warning={}{}{}".format(
        RED, MATCH_THRESHOLD, RESET, YELLOW, WARNING_THRESHOLD, RESET))
    print("  " + "-" * 70)
    top_n = 3
    while True:
        try:
            msg = input("\n  {}Message>{} ".format(BOLD, RESET)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break
        if not msg:
            continue
        if msg.startswith("/"):
            c = msg.lower().split()
            if c[0] in ("/quit", "/exit", "/q"):
                print("  Goodbye!"); break
            elif c[0] == "/verbose":
                verbose = not verbose
                print("  Verbose: {}".format("ON" if verbose else "OFF")); continue
            elif c[0] == "/top" and len(c) > 1:
                try: top_n = int(c[1]); print("  Showing top {}".format(top_n))
                except ValueError: print("  Usage: /top N")
                continue
            elif c[0] == "/help":
                print("  /verbose  \u2014 toggle full table")
                print("  /top N    \u2014 show top N results")
                print("  /quit     \u2014 exit"); continue
            else:
                print("  Unknown command. /help for options."); continue
        results = score_message(model, msg, embs, texts, cats)
        if verbose:
            display_verbose(results)
        else:
            display_default(results, top_n)


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def run_file(filepath, model, embs, texts, cats, verbose=False):
    if not os.path.exists(filepath):
        print("  ERROR: File not found: {}".format(filepath)); sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    sentences = [s.strip() for s in content.split("###") if s.strip()]
    print("  Processing {} sentences from: {}\n".format(len(sentences), filepath))
    print("  " + "=" * 100)
    all_results = []
    for idx, sent in enumerate(sentences, 1):
        disp = sent[:80] + ("..." if len(sent) > 80 else "")
        print("\n  {}[{}/{}] Message:{} \"{}\"".format(BOLD, idx, len(sentences), RESET, disp))
        results = score_message(model, sent, embs, texts, cats)
        all_results.append(results)
        if verbose:
            display_verbose(results)
        else:
            display_default(results)
    # Summary
    matches = sum(1 for r in all_results if r[0][0] >= MATCH_THRESHOLD)
    warns = sum(1 for r in all_results if WARNING_THRESHOLD <= r[0][0] < MATCH_THRESHOLD)
    clean = sum(1 for r in all_results if r[0][0] < WARNING_THRESHOLD)
    print("\n  {}Summary{}".format(BOLD, RESET))
    print("  " + "=" * 60)
    print("  Total sentences:  {}".format(len(sentences)))
    print("  {}\u25a0 Matches:         {}{}".format(RED, matches, RESET))
    print("  {}\u25a0 Warnings:        {}{}".format(YELLOW, warns, RESET))
    print("  {}\u25a0 No match:        {}{}".format(GREEN, clean, RESET))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Anchor Evaluator: " + SCRIPT_NAME)
    parser.add_argument("--file", "-f", type=str, default=None,
        help="Input file with sentences separated by ### (default: interactive)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Show full scored table (all anchors sorted)")
    parser.add_argument("--show-examples", action="store_true",
        help="Print all positive examples by category")
    args = parser.parse_args()

    print_banner()

    if args.show_examples:
        print_examples()
        if not args.file:
            try:
                r = input("  Enter interactive mode? [Y/n] ").strip().lower()
                if r == "n": return
            except (EOFError, KeyboardInterrupt):
                print(); return

    model = load_model()
    all_texts, all_cats, all_embs = prepare_anchors(model)

    if args.file:
        run_file(args.file, model, all_embs, all_texts, all_cats, args.verbose)
    else:
        run_interactive(model, all_embs, all_texts, all_cats, args.verbose)


if __name__ == "__main__":
    main()
'''


def generate_script(name, proposition, anchors_dict, match_thresh, warn_thresh):
    """Generate a standalone evaluator by replacing placeholders in the template."""
    total = sum(len(v) for v in anchors_dict.values())
    anchors_json = json.dumps(anchors_dict, indent=4, ensure_ascii=False)
    title_underline = "=" * (29 + len(name))

    script = EVALUATOR_TEMPLATE
    script = script.replace("%%NAME%%", name)
    script = script.replace("%%TITLE_UNDERLINE%%", title_underline)
    script = script.replace("%%PROPOSITION%%", proposition)
    script = script.replace("%%TOTAL_ANCHORS%%", str(total))
    script = script.replace("%%ANCHORS_JSON%%", anchors_json)
    script = script.replace("%%MATCH_THRESHOLD%%", str(match_thresh))
    script = script.replace("%%WARNING_THRESHOLD%%", str(warn_thresh))

    return script


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Anchor Generator - produce standalone evaluation scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s
              %(prog)s -p "The user requests weapon instructions." -name weapons
              %(prog)s -p "..." -name weapons -n 50
        """),
    )
    parser.add_argument("-p", "--proposition", type=str, help="The proposition")
    parser.add_argument("-name", "--name", type=str,
                        help="Name for output script (semantic_anchor_<name>.py)")
    parser.add_argument("-n", "--num-examples", type=int, default=None,
                        help="Number of anchors (20-100)")
    parser.add_argument("--config", type=str, default="config.ini",
                        help="Config file path")
    args = parser.parse_args()

    w = 60
    print("")
    print(BOLD + "\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + "SEMANTIC ANCHOR GENERATOR".center(w) + "\u2551")
    print("\u2551" + "Generate standalone evaluation scripts".center(w) + "\u2551")
    print("\u255a" + "\u2550" * w + "\u255d" + RESET)
    print("")

    config = load_config(args.config)

    # --- Get proposition ---
    if args.proposition:
        proposition = args.proposition
    else:
        print("  {}Enter the proposition:{}".format(BOLD, RESET))
        print("  {}(A formal description of the type of message to detect){}".format(DIM, RESET))
        try:
            proposition = input("\n  Proposition> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        if not proposition:
            print("  ERROR: Proposition cannot be empty."); sys.exit(1)

    # --- Get name ---
    if args.name:
        name = args.name
    else:
        print("\n  {}Enter a name for the output script:{}".format(BOLD, RESET))
        print("  {}(Will produce: semantic_anchor_<name>.py){}".format(DIM, RESET))
        try:
            name = input("\n  Name> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        if not name:
            print("  ERROR: Name cannot be empty."); sys.exit(1)

    name = name.replace(" ", "_").replace("-", "_").lower()

    # --- Num examples ---
    if args.num_examples is not None:
        num_examples = max(20, min(100, args.num_examples))
    else:
        num_examples = max(20, min(100, int(config.get("anchors", "num_examples"))))

    # --- Thresholds ---
    match_thresh = float(config.get("thresholds", "match_threshold"))
    warn_thresh = float(config.get("thresholds", "warning_threshold"))

    # --- Print summary ---
    print("\n  {}Configuration:{}".format(BOLD, RESET))
    print("  Proposition:     \"{}\"".format(proposition))
    print("  Output script:   semantic_anchor_{}.py".format(name))
    print("  Num anchors:     {}".format(num_examples))
    print("  Match threshold: {}".format(match_thresh))
    print("  Warn threshold:  {}".format(warn_thresh))
    print()

    # --- Generate anchors ---
    categories = parse_categories(config)
    anchors_dict = generate_anchors(proposition, categories, num_examples, config)

    total = sum(len(v) for v in anchors_dict.values())
    print("  {}\u2713 Generated {} anchors across {} categories{}".format(
        GREEN, total, len(anchors_dict), RESET))

    # --- Show preview ---
    print("\n  {}Preview:{}".format(BOLD, RESET))
    for cat, examples in anchors_dict.items():
        preview = examples[0][:50] + "..." if len(examples[0]) > 50 else examples[0]
        print("    {} ({}): \"{}\"".format(cat, len(examples), preview))

    # --- Generate script ---
    script_content = generate_script(name, proposition, anchors_dict,
                                     match_thresh, warn_thresh)

    output_path = "semantic_anchor_{}.py".format(name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(output_path, 0o755)

    print("\n  {}{}\u2713 Generated: {}{}".format(GREEN, BOLD, output_path, RESET))
    print("\n  {}Usage:{}".format(BOLD, RESET))
    print("    python {}                         # Interactive mode".format(output_path))
    print("    python {} --verbose               # Full table".format(output_path))
    print("    python {} --show-examples         # View anchors by category".format(output_path))
    print("    python {} --file input.txt        # Evaluate file (### separated)".format(output_path))
    print("    python {} --file input.txt -v     # Evaluate file verbose".format(output_path))
    print()


if __name__ == "__main__":
    main()
