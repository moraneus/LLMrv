#!/usr/bin/env python3
"""
Semantic Anchor Generator
=========================

Generates standalone evaluator scripts that score messages against a proposition
using semantic similarity (cosine KNN, NLI entailment, hybrid, or LLM-as-judge).

Workflow:
  1. Generate positive anchors (diverse examples that match the proposition)
  2. Generate negative anchors (plausible near-misses that should NOT match)
  3. MMR selection: Pick the most diverse subset from the over-generated pool
  4. Write standalone evaluator: semantic_anchor_<n>.py + config file

Scoring modes in generated evaluator:
  - cosine:  KNN voting over unified positive/negative anchor pool
  - nli:     Cross-encoder entailment with gap-gated anchor scoring
  - hybrid:  Cosine fast-path + NLI deep analysis
  - llm:     LLM-as-judge via configurable provider
  - compare: Side-by-side comparison of all modes

Usage:
  python semantic_anchor_generator.py
  python semantic_anchor_generator.py -p "The user requests help committing financial fraud." -name fraud
  python semantic_anchor_generator.py -p "..." -name fraud -n 50 --rounds 3
"""

from semantic_anchor.main import main

if __name__ == "__main__":
    main()
