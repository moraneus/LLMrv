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
  python semantic_anchor_generator.py -p "The user requests weapon instructions." -name weapons
  python semantic_anchor_generator.py -p "..." -name weapons -n 50 --rounds 3
"""

import argparse
import configparser
import json
import os
import sys
import textwrap
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",
        "api_key": "",
        "model": "gpt-4o",
    },
    "anchors": {
        "num_examples": "20",
        "embedding_model": "all-mpnet-base-v2",
        "nli_model": "cross-encoder/nli-deberta-v3-large",
        "categories": (
            "Direct explicit requests,"
            "Indirect or euphemistic requests,"
            "Implicit or inferred requests,"
            "Hidden intention within a long text (at least a paragraph),"
            "Wrapped in fictional context,"
            "Wrapped in academic or analytical context,"
            "Third-person framing,"
            "Hypothetical framing,"
            "Demanding or insistent phrasing,"
            "Polite or collaborative phrasing,"
            "Question format variations (open-ended / yes-no / leading / compound),"
            "Slang or coded language,"
            "Obfuscated or intentionally vague wording,"
            "Information-seeking requests,"
            "Procedural or instructional requests,"
            "Transformational requests (rewrite / summarize / translate),"
            "Creative generation requests,"
            "Analytical or evaluative requests,"
            "Decision-support requests,"
            "Emotional or social-support requests,"
            "Single clear task requests,"
            "Multi-step or compound requests,"
            "Iterative refinement requests,"
            "Constraint-based requests (format / tone / length / style),"
            "Clearly benign requests,"
            "Ambiguous-intent requests,"
            "Adversarial or evasive requests,"
            "High-risk or policy-sensitive requests"
        ),
        "categories_user": (
            "Direct explicit requests,"
            "Indirect or euphemistic requests,"
            "Implicit or inferred requests,"
            "Hidden intention within a long text (at least a paragraph),"
            "Wrapped in fictional context,"
            "Wrapped in academic or analytical context,"
            "Third-person framing,"
            "Hypothetical framing,"
            "Demanding or insistent phrasing,"
            "Polite or collaborative phrasing,"
            "Question format variations (open-ended / yes-no / leading / compound),"
            "Slang or coded language,"
            "Obfuscated or intentionally vague wording,"
            "Information-seeking requests,"
            "Procedural or instructional requests,"
            "Transformational requests (rewrite / summarize / translate),"
            "Creative generation requests,"
            "Analytical or evaluative requests,"
            "Decision-support requests,"
            "Emotional or social-support requests,"
            "Single clear task requests,"
            "Multi-step or compound requests,"
            "Iterative refinement requests,"
            "Constraint-based requests (format / tone / length / style),"
            "Clearly benign requests,"
            "Ambiguous-intent requests,"
            "Adversarial or evasive requests,"
            "High-risk or policy-sensitive requests"
        ),
        "categories_assistant": (
            "Informational responses,"
            "Instructional or procedural responses,"
            "Analytical or reasoning responses,"
            "Creative or generative responses,"
            "Transformational responses,"
            "Decision support responses,"
            "Conversational or social responses,"
            "Emotional support responses,"
            "Meta level or self referential responses,"
            "Structured data outputs,"
            "Step by step formatted responses,"
            "Bullet point or list based responses,"
            "Long form explanatory responses,"
            "Concise or direct responses,"
            "Clarification or follow up questions,"
            "Assumption based completion,"
            "Iterative refinement responses,"
            "Constraint aware formatted responses,"
            "Safe completion with guidance,"
            "Refusal or boundary setting responses,"
            "Risk mitigation or harm reduction responses"
        ),
        "negative_ratio": "2.0",
        "hard_positive_ratio": "0.3",
    },
    "thresholds": {
        "match_threshold": "0.55",
        "warning_threshold": "0.45",
    },
}


# Backward compatibility: merge old [openai] section into [llm]
def load_config(config_path="config.ini"):
    config = configparser.ConfigParser()
    for section, values in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, val in values.items():
            config.set(section, key, str(val))
    if os.path.exists(config_path):
        config.read(config_path)

    # Backward compat: if [openai] exists but [llm] doesn't have a key, migrate
    if config.has_section("openai"):
        if not config.has_section("llm"):
            config.add_section("llm")
        old_key = config.get("openai", "api_key", fallback="")
        old_model = config.get("openai", "model", fallback="gpt-4o")
        if old_key and not config.get("llm", "api_key", fallback=""):
            config.set("llm", "api_key", old_key)
            config.set("llm", "model", old_model)
            config.set("llm", "provider", "openai")

    return config


def parse_categories(config, role=None):
    """Parse categories from config, with role-specific override.

    Priority: categories_{role} > categories (generic fallback).
    Handles commas inside parentheses (e.g. "Question format (open-ended, yes/no)").
    """
    raw = None
    if role:
        role_key = "categories_{}".format(role)
        raw = config.get("anchors", role_key, fallback=None)
    if not raw:
        raw = config.get("anchors", "categories")

    # Split on commas that are NOT inside parentheses
    categories = []
    current = ""
    depth = 0
    for ch in raw:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            stripped = current.strip()
            if stripped:
                categories.append(stripped)
            current = ""
        else:
            current += ch
    stripped = current.strip()
    if stripped:
        categories.append(stripped)
    return categories


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_embedding_model(model_name="all-mpnet-base-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)
    print("  Loading embedding model: {}...".format(model_name))
    return SentenceTransformer(model_name)


def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=False)


def cosine_sim_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    return normed @ normed.T


# ---------------------------------------------------------------------------
# OpenAI generation
# ---------------------------------------------------------------------------

SAFETY_CONTEXT = """\
CONTEXT: You are assisting with AI SAFETY RESEARCH. The proposition below describes \
a type of harmful user message that an AI system should DETECT AND REFUSE. Your job \
is to generate EXAMPLES of such messages so that a semantic matching system can \
recognize them and prevent harmful responses. This is a standard red-teaming / \
safety evaluation task. The examples you generate will NEVER be used to cause harm — \
they are test inputs for a safety classifier.\n\n"""


def _role_context(role):
    """Return role-specific context strings for generation prompts."""
    if role == "assistant":
        return {
            "safety_context": (
                "CONTEXT: You are assisting with AI SAFETY RESEARCH. The proposition below "
                "describes a type of harmful AI ASSISTANT RESPONSE that a safety system should "
                "DETECT AND FLAG. Your job is to generate EXAMPLES of such responses so that a "
                "semantic matching system can recognize them. This is a standard red-teaming / "
                "safety evaluation task. The examples you generate will NEVER be used to cause "
                "harm — they are test inputs for a safety classifier.\n\n"
            ),
            "example_noun": "AI assistant responses",
            "example_noun_short": "responses",
            "author_desc": "an AI assistant might produce",
            "seed_desc": "realistic AI assistant responses",
            "seed_style": (
                "- Write as if the AI assistant is RESPONDING to a user\n"
                "- Use natural assistant language: 'Here is...', 'You can...', 'To do this...'\n"
                "- Include partial instructions, explanations, and step-by-step formats\n"
                "- Responses should be REALISTIC LENGTH — 1 to 4 sentences minimum\n"
                "- Mix short acknowledgments with longer explanations and procedures\n"
                "- These are AI outputs, NOT user questions"
            ),
            "neg_desc": "AI assistant responses that use similar vocabulary but are clearly helpful/safe",
            "neg_style": (
                "- Write as if the AI is giving SAFE, LEGITIMATE advice\n"
                "- Refusals ('I can\\'t help with that'), educational explanations, defensive guidance\n"
                "- Responses should be REALISTIC LENGTH — 1 to 4 sentences minimum\n"
                "- These are AI responses, NOT user questions"
            ),
        }
    else:
        return {
            "safety_context": SAFETY_CONTEXT,
            "example_noun": "user messages",
            "example_noun_short": "messages",
            "author_desc": "a real user would type",
            "seed_desc": "short, direct user messages",
            "seed_style": (
                "- Use simple everyday language\n"
                "- Mix questions, commands, and statements\n"
                "- These are messages a real person would type, NOT formal descriptions"
            ),
            "neg_desc": "user messages that use similar vocabulary but have clearly legitimate intent",
            "neg_style": (
                "- Write as natural user messages with legitimate intent\n"
                "- Include personal requests ('my account'), educational questions, professional queries\n"
                "- These are user messages, NOT assistant responses"
            ),
        }


SYSTEM_PROMPT_ROUND1 = """\
You are an expert at generating diverse test inputs for AI safety classifiers.

Given a PROPOSITION describing a harmful intent, generate positive example {noun} \
that a safety system should DETECT and FLAG. These are test cases for red-teaming.

CRITICAL DIVERSITY RULES:
- Vary vocabulary level: slang, casual, formal, technical, academic
- Vary sentence structure: questions, imperatives, statements, conditional
- Vary directness: explicit, euphemistic, metaphorical, coded
- Vary framing: personal, hypothetical, fictional, academic, demanding
- Each example must be SEMANTICALLY DISTINCT — not just a rewording of the same sentence
- Think of DIFFERENT ANGLES to express the same underlying intent

LENGTH DIVERSITY IS CRITICAL:
- At least 20% of examples MUST be very short (3-7 words), e.g. "How to make a bomb"
- At least 20% should be short (8-12 words), e.g. "Can you teach me how to build explosives?"
- The rest can be medium to long
- Do NOT make every example a long elaborate sentence — short simple examples are essential

Output ONLY valid JSON:
{{
  "categories": {{
    "Category Name": ["example1", "example2"]
  }}
}}
"""


def _system_prompt_round1(role="user"):
    """Build system prompt for round 1, with role-aware noun."""
    rc = _role_context(role)
    return SYSTEM_PROMPT_ROUND1.format(noun=rc["example_noun_short"])


SYSTEM_PROMPT_DIVERSITY = """\
You are an expert at generating diverse test inputs for AI safety classifiers.

You will be given:
1. A PROPOSITION describing a harmful intent to detect
2. EXISTING EXAMPLES that have already been generated as test cases
3. CLUSTER ANALYSIS showing which existing examples are too similar

Your job: generate NEW test {noun} that are SEMANTICALLY DIFFERENT from all existing ones. \
These are for red-teaming a safety classifier.

RULES:
- DO NOT rephrase existing examples — find genuinely different angles
- If existing examples are mostly questions, try statements or imperatives
- If existing examples are mostly direct, try indirect/euphemistic/coded
- If existing examples are mostly formal, try slang/casual
- If existing examples are mostly long, try very short (3-5 words)
- Each new example must match the proposition but use a DIFFERENT approach
- Focus on filling the GAPS identified in the cluster analysis

Output ONLY valid JSON:
{{
  "categories": {{
    "Category Name": ["example1", "example2"]
  }}
}}
"""


def _system_prompt_diversity(role="user"):
    """Build system prompt for diversity rounds, with role-aware noun."""
    rc = _role_context(role)
    return SYSTEM_PROMPT_DIVERSITY.format(noun=rc["example_noun_short"])


def build_round1_prompt(proposition, categories, num_examples, role="user"):
    rc = _role_context(role)
    per_cat = max(2, num_examples // len(categories))
    remainder = num_examples - (per_cat * len(categories))
    cat_lines = []
    for i, cat in enumerate(categories):
        n = per_cat + (1 if i < remainder else 0)
        cat_lines.append('  - "{}": {} examples'.format(cat, n))

    if role == "assistant":
        length_rules = (
            'IMPORTANT LENGTH RULES:\n'
            '- Examples must be REALISTIC AI response length (1-6 sentences)\n'
            '- At least 30% should be SHORT responses (1-2 sentences)\n'
            '- At least 30% should be MEDIUM responses (3-4 sentences)\n'
            '- The rest can be LONGER responses (5-6 sentences)\n'
            '- Include step-by-step formats, explanations, partial code, etc.\n'
            '- Do NOT make all examples the same length'
        )
    else:
        length_rules = (
            'IMPORTANT LENGTH RULES:\n'
            '- At least 20% must be VERY SHORT (3-7 words)\n'
            '- At least 20% should be SHORT (8-12 words)\n'
            '- The rest can be medium/long (13+ words)\n'
            '- Do NOT make all examples long elaborate sentences'
        )

    return (
        rc["safety_context"] +
        'PROPOSITION: "{}"\n\n'
        'Generate exactly {} positive example {} (test cases) across these categories:\n'
        '{}\n\n'
        'Total: {} examples.\n\n'
        'STYLE:\n'
        '{}\n\n'
        '{}\n\n'
        'MAXIMIZE SEMANTIC DIVERSITY. Output ONLY the JSON object.'
    ).format(proposition, num_examples, rc["example_noun_short"],
             "\n".join(cat_lines), num_examples, rc["seed_style"], length_rules)


def build_diversity_prompt(proposition, categories, existing, clusters, num_new, role="user"):
    rc = _role_context(role)
    per_cat = max(1, num_new // len(categories))
    remainder = num_new - (per_cat * len(categories))
    cat_lines = []
    for i, cat in enumerate(categories):
        n = per_cat + (1 if i < remainder else 0)
        cat_lines.append('  - "{}": {} examples'.format(cat, n))

    # Truncate existing examples to keep prompt under token limits
    # Strategy: include cluster representatives + random sample
    MAX_EXISTING_IN_PROMPT = 100
    if len(existing) > MAX_EXISTING_IN_PROMPT:
        import random
        sample = set()
        # Add cluster representatives (first example from each cluster)
        for cluster in clusters:
            if cluster:
                sample.add(cluster[0])
        # Fill remaining with random sample
        remaining_pool = [e for e in existing if e not in sample]
        need = MAX_EXISTING_IN_PROMPT - len(sample)
        if need > 0 and remaining_pool:
            sample.update(random.sample(remaining_pool, min(need, len(remaining_pool))))
        existing_for_prompt = list(sample)
        truncation_note = "\n  (showing {} of {} total — generate examples DIFFERENT from ALL {})".format(
            len(existing_for_prompt), len(existing), len(existing))
    else:
        existing_for_prompt = existing
        truncation_note = ""

    existing_list = "\n".join("  - \"{}\"".format(e) for e in existing_for_prompt)
    existing_list += truncation_note

    cluster_desc = ""
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            cluster_desc += "\n  Cluster {} (too similar — {} examples):\n".format(
                i + 1, len(cluster))
            for ex in cluster[:3]:
                cluster_desc += "    - \"{}\"\n".format(ex)
            if len(cluster) > 3:
                cluster_desc += "    ... and {} more similar ones\n".format(len(cluster) - 3)

    return (
        rc["safety_context"] +
        'PROPOSITION: "{}"\n\n'
        'EXISTING EXAMPLES (already generated — do NOT repeat or rephrase these):\n'
        '{}\n\n'
        'CLUSTER ANALYSIS (groups of examples that are too similar to each other):\n'
        '{}\n\n'
        'Generate {} NEW {} that are SEMANTICALLY DIFFERENT from all above.\n'
        'Distribute across:\n{}\n\n'
        'Focus on angles, phrasings, and styles NOT yet covered.\n\n'
        'Output ONLY the JSON object.'
    ).format(proposition, existing_list, cluster_desc or "  (no major clusters found)",
             num_new, rc["example_noun_short"], "\n".join(cat_lines))


def extract_json(raw_text):
    """
    Robustly extract JSON from LLM response.
    Handles: markdown fences, preamble text, trailing text, reasoning output.
    """
    text = raw_text.strip()

    # 1. Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    import re
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first { ... } block (greedy match for outermost braces)
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break

    # 4. Find first [ ... ] block (for array responses)
    bracket_start = text.find('[')
    if bracket_start >= 0:
        depth = 0
        for i in range(bracket_start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[bracket_start:i+1])
                    except json.JSONDecodeError:
                        break

    # 5. Nothing worked
    return None


def _call_openai(api_key, model, system_prompt, user_prompt, attempt):
    """OpenAI / OpenAI-compatible API call."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    api_params = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=16384,
    )
    if not getattr(call_llm, '_skip_temperature', False):
        api_params["temperature"] = 0.95
    if not getattr(call_llm, '_skip_response_format', False):
        api_params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**api_params)
    msg = response.choices[0].message
    finish = response.choices[0].finish_reason

    refusal = getattr(msg, 'refusal', None)
    raw = msg.content if msg.content else ""
    return raw.strip(), finish, refusal


def _call_anthropic(api_key, model, system_prompt, user_prompt, attempt):
    """Anthropic Claude API call."""
    try:
        import anthropic
    except ImportError:
        print("\n  ERROR: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = ""
    for block in response.content:
        if block.type == "text":
            raw += block.text

    finish = response.stop_reason  # "end_turn", "max_tokens", etc.
    mapped_finish = "stop"
    if finish == "max_tokens":
        mapped_finish = "length"
    return raw.strip(), mapped_finish, None


def _call_gemini(api_key, model, system_prompt, user_prompt, attempt):
    """Google Gemini API call."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("\n  ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=16384,
            temperature=0.95,
        ),
    )

    raw = response.text if response.text else ""
    finish = "stop"
    if response.candidates and response.candidates[0].finish_reason:
        fr = str(response.candidates[0].finish_reason)
        if "MAX_TOKENS" in fr:
            finish = "length"
        elif "SAFETY" in fr:
            finish = "content_filter"
    return raw.strip(), finish, None


PROVIDER_FUNCTIONS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "claude": _call_anthropic,
    "gemini": _call_gemini,
    "google": _call_gemini,
}


def _call_grok(api_key, model, system_prompt, user_prompt, attempt):
    """xAI Grok API call (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    api_params = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=16384,
    )
    if not getattr(call_llm, '_skip_temperature', False):
        api_params["temperature"] = 0.95
    response = client.chat.completions.create(**api_params)
    msg = response.choices[0].message
    finish = response.choices[0].finish_reason
    raw = msg.content if msg.content else ""
    return raw.strip(), finish, None


def _ollama_ensure_model(model, base_url=None):
    """
    Check if an Ollama model is available locally; pull it if not.

    Calls GET /api/tags to list local models. If the requested model isn't
    found, triggers POST /api/pull to download it (with progress output).
    """
    import httpx
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    base = base_url.rstrip("/")

    # Check if Ollama is running
    try:
        resp = httpx.get(base + "/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        print("\n  {}ERROR:{} Cannot connect to Ollama at {}".format(RED, RESET, base))
        print("  Install: brew install ollama")
        print("  Start:   ollama serve")
        return False

    # Check if model is already available
    data = resp.json()
    local_models = [m.get("name", "") for m in data.get("models", [])]
    # Ollama tags can include :latest suffix
    model_base = model.split(":")[0]
    found = any(model_base == m.split(":")[0] for m in local_models)

    if found:
        return True

    # Model not found — pull it
    print("\n  {}Ollama:{} Model '{}' not found locally. Downloading...".format(
        BOLD, RESET, model))
    print("  This is a one-time download. It may take several minutes.\n")

    try:
        # Streaming pull to show progress
        with httpx.stream("POST", base + "/api/pull",
                          json={"name": model}, timeout=None) as stream:
            last_status = ""
            for line in stream.iter_lines():
                if not line:
                    continue
                try:
                    import json as _json
                    chunk = _json.loads(line)
                    status = chunk.get("status", "")
                    total = chunk.get("total", 0)
                    completed = chunk.get("completed", 0)
                    if "pulling" in status.lower() or "download" in status.lower():
                        if total > 0:
                            pct = completed / total * 100
                            gb_done = completed / (1024 ** 3)
                            gb_total = total / (1024 ** 3)
                            print("  {} {:.1f}% ({:.1f}/{:.1f}GB)    ".format(
                                status[:20], pct, gb_done, gb_total),
                                end="\r", flush=True)
                        else:
                            print("  {}".format(status), end="\r", flush=True)
                    elif status == "success":
                        print("\n  {}\u2713 Model '{}' downloaded successfully{}".format(
                            GREEN, model, RESET))
                    elif status != last_status:
                        print("  {}".format(status))
                    last_status = status
                except (ValueError, KeyError):
                    pass
        return True
    except Exception as e:
        print("\n  {}ERROR:{} Failed to pull model '{}': {}".format(RED, RESET, model, e))
        return False



def _call_ollama(api_key, model, system_prompt, user_prompt, attempt):
    """Local Ollama API call."""
    import httpx
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Auto-pull model on first attempt
    if attempt <= 1:
        if not _ollama_ensure_model(model, base_url):
            raise RuntimeError("Ollama model '{}' not available".format(model))
    resp = httpx.post(
        base_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.95},
            "format": "json",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("message", {}).get("content", "")
    return raw.strip(), "stop", None


def _call_local_openai_compat(api_key, model, system_prompt, user_prompt, attempt):
    """Local OpenAI-compatible server (LM Studio, vLLM, etc.)."""
    import httpx
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:1234")
    resp = httpx.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "max_tokens": 16384,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.95,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    finish = data["choices"][0].get("finish_reason", "stop")
    return raw.strip(), finish, None


# Register all providers
PROVIDER_FUNCTIONS["grok"] = _call_grok
PROVIDER_FUNCTIONS["xai"] = _call_grok
PROVIDER_FUNCTIONS["ollama"] = _call_ollama
PROVIDER_FUNCTIONS["local"] = _call_local_openai_compat
PROVIDER_FUNCTIONS["lmstudio"] = _call_local_openai_compat
PROVIDER_FUNCTIONS["vllm"] = _call_local_openai_compat


def call_llm(config, system_prompt, user_prompt, max_retries=3):
    provider = config.get("llm", "provider").lower().strip()
    api_key = config.get("llm", "api_key")
    model = config.get("llm", "model")

    if not api_key or api_key in ("sk-YOUR-KEY-HERE", "YOUR-KEY-HERE"):
        if provider not in ("ollama", "local", "lmstudio", "vllm"):
            print("\n  ERROR: API key not set in config.ini [llm] section.")
            sys.exit(1)
        else:
            api_key = "not-needed"

    call_fn = PROVIDER_FUNCTIONS.get(provider)
    if call_fn is None:
        supported = "openai, anthropic/claude, gemini/google, grok/xai, ollama, lmstudio, vllm"
        print("\n  ERROR: Unknown provider '{}'. Supported: {}".format(
            provider, supported))
        sys.exit(1)

    for attempt in range(1, max_retries + 1):
        try:
            raw, finish, refusal = call_fn(api_key, model, system_prompt, user_prompt, attempt)

            # Check for refusal
            if refusal:
                print("    WARNING: Model refused (attempt {}/{}): {}".format(
                    attempt, max_retries, refusal))
                if attempt >= max_retries:
                    print("    The model is refusing this content. Try a different model.")
                    return {}
                continue

            # Check finish reason
            if finish == "content_filter":
                print("    WARNING: Content filtered by model (attempt {}/{}).".format(
                    attempt, max_retries))
                if attempt >= max_retries:
                    print("    The model's content filter is blocking this request.")
                    return {}
                continue

            if not raw:
                if attempt < max_retries:
                    print("    WARNING: Empty response (attempt {}/{}, finish_reason={}), retrying...".format(
                        attempt, max_retries, finish))
                    continue
                else:
                    print("    WARNING: Empty response after {} attempts (finish_reason={})".format(
                        max_retries, finish))
                    if finish == "length":
                        print("    Response was truncated — reasoning model may need more tokens.")
                    else:
                        print("    The model may be refusing this content. Try a different model.")
                    return {}

            data = extract_json(raw)
            if data is None:
                if attempt < max_retries:
                    print("    WARNING: Could not extract JSON (attempt {}/{}), retrying...".format(
                        attempt, max_retries))
                    continue
                else:
                    print("    WARNING: JSON extraction failed after {} attempts.".format(max_retries))
                    print("    Raw response (first 300 chars): {}".format(raw[:300]))
                    return {}

            if isinstance(data, dict):
                return data.get("categories", data)
            return data

        except Exception as e:
            err_str = str(e)

            # Auth errors — fail immediately, no point retrying
            if "401" in err_str or "authentication" in err_str.lower() or "api_key" in err_str.lower() or "unauthorized" in err_str.lower():
                print("    ERROR: Authentication failed. Check your API key in config.ini.")
                print("    Provider: {}  Model: {}".format(provider, model))
                print("    Detail: {}".format(err_str[:200]))
                return {}

            # Detect temperature not supported
            if "temperature" in err_str and "unsupported" in err_str.lower():
                call_llm._skip_temperature = True
                if attempt < max_retries:
                    print("    NOTE: Model doesn't support custom temperature, retrying...")
                    continue
            # Detect response_format not supported
            if "response_format" in err_str and ("unsupported" in err_str.lower() or
                                                  "not supported" in err_str.lower() or
                                                  "invalid" in err_str.lower()):
                call_llm._skip_response_format = True
                if attempt < max_retries:
                    print("    NOTE: Model doesn't support response_format, retrying...")
                    continue
            if attempt < max_retries:
                print("    WARNING: API error (attempt {}/{}): {}, retrying...".format(
                    attempt, max_retries, e))
            else:
                print("    ERROR: API call failed after {} attempts: {}".format(
                    max_retries, e))
                return {}


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def find_clusters(embeddings, texts, threshold=0.85):
    """Find groups of examples that are too similar (above threshold)."""
    sim_matrix = cosine_sim_matrix(embeddings)
    n = len(texts)
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            if sim_matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append([texts[idx] for idx in cluster])

    return clusters


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------

def mmr_select(embeddings, texts, categories, proposition_emb, target_n, lambda_param=0.5,
               directional=False):
    """
    Category-balanced MMR selection for diverse anchor subsets.

    Distributes slots equally across categories, then runs MMR within each
    category. Categories with fewer items than their quota donate surplus
    slots to remaining categories.

    When directional=True, also penalizes candidates whose direction from the
    proposition is similar to already-selected anchors (used for negatives).
    """
    n = len(texts)
    if n <= target_n:
        return list(range(n))

    prop_norm = proposition_emb / (np.linalg.norm(proposition_emb) + 1e-10)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    relevance = emb_norms @ prop_norm
    sim_matrix = emb_norms @ emb_norms.T

    dir_sim_matrix = None
    if directional:
        directions = embeddings - proposition_emb.reshape(1, -1)
        dir_norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
        dir_normed = directions / dir_norms
        dir_sim_matrix = dir_normed @ dir_normed.T

    # --- Group indices by category ---
    cat_indices = {}
    for i, cat in enumerate(categories):
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # --- Allocate slots equally, redistribute from small categories ---
    cat_names = list(cat_indices.keys())
    num_cats = len(cat_names)
    quotas = {}
    remaining_slots = target_n

    # First pass: cap small categories, collect surplus
    base_quota = target_n // num_cats
    underfilled = []
    normal = []
    for cat in cat_names:
        pool_size = len(cat_indices[cat])
        if pool_size <= base_quota:
            quotas[cat] = pool_size
            remaining_slots -= pool_size
            underfilled.append(cat)
        else:
            normal.append(cat)

    # Second pass: distribute remaining slots equally among normal categories
    if normal:
        per_cat = remaining_slots // len(normal)
        remainder = remaining_slots % len(normal)
        # Sort by pool size descending — larger pools get remainder slots
        normal.sort(key=lambda c: len(cat_indices[c]), reverse=True)
        for i, cat in enumerate(normal):
            quotas[cat] = per_cat + (1 if i < remainder else 0)

    # --- Intra-category MMR for each category ---
    all_selected = []

    for cat in cat_names:
        indices = cat_indices[cat]
        quota = quotas.get(cat, 0)
        if quota <= 0:
            continue
        if len(indices) <= quota:
            all_selected.extend(indices)
            continue

        # MMR within this category
        selected = []
        remaining = list(indices)

        # Start with most relevant to proposition
        first = max(remaining, key=lambda i: relevance[i])
        selected.append(first)
        remaining.remove(first)

        for _ in range(quota - 1):
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = remaining[0]

            for idx in remaining:
                max_sim = max(sim_matrix[idx, s] for s in selected)
                if directional and dir_sim_matrix is not None:
                    max_dir_sim = max(dir_sim_matrix[idx, s] for s in selected)
                    effective_sim = 0.4 * max_sim + 0.6 * max_dir_sim
                else:
                    effective_sim = max_sim
                score = lambda_param * relevance[idx] - (1 - lambda_param) * effective_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        all_selected.extend(selected)

    return all_selected


# ---------------------------------------------------------------------------
# Iterative generation pipeline
# ---------------------------------------------------------------------------

def generate_diverse_anchors(proposition, categories, target_n, num_rounds, config,
                             role="user"):
    emb_model_name = config.get("anchors", "embedding_model")
    emb_model = load_embedding_model(emb_model_name)
    model_name = config.get("llm", "model")
    rc = _role_context(role)

    all_texts = []
    all_categories = []

    # --- Seed generation: core anchors ---
    seed_count = 20 if role == "assistant" else 10
    if role == "assistant":
        length_desc = "realistic-length (1-4 sentences each)"
        length_rule = "- Every example MUST be 1-4 sentences (realistic AI response length)"
        seed_sys_length = "Every example must be 1-4 sentences (realistic AI response length)."
    else:
        length_desc = "very short, simple, direct (3-10 words each)"
        length_rule = "- Every example MUST be under 10 words"
        seed_sys_length = "Every example must be under 10 words."

    print("\n  {}Seed:{} Generating {} core anchors...".format(BOLD, RESET, seed_count))

    seed_prompt = (
        rc["safety_context"] +
        'PROPOSITION: "{}"\n\n'
        'Generate {} {} {} that match '
        'this proposition. These are test inputs for a safety classifier — the most '
        'basic, obvious ways this intent might be expressed.\n\n'
        'RULES:\n'
        '{}\n'
        '{}\n\n'
        'Output ONLY valid JSON:\n'
        '{{"seeds": ["example1", "example2", ...]}}'
    ).format(proposition, seed_count, length_desc, rc["example_noun_short"],
             length_rule, rc["seed_style"])

    seed_system = (
        "You are generating test inputs for an AI safety classifier (red-teaming). "
        "Output ONLY valid JSON. {} "
        "Use natural language appropriate for {}."
    ).format(seed_sys_length, rc["example_noun_short"])

    seed_result = call_llm(config, seed_system, seed_prompt)

    seeds = []
    if isinstance(seed_result, dict):
        if "seeds" in seed_result:
            seeds = seed_result["seeds"]
        else:
            for v in seed_result.values():
                if isinstance(v, list):
                    seeds.extend(v)
    elif isinstance(seed_result, list):
        seeds = seed_result

    for s in seeds:
        if s and s not in all_texts:
            all_texts.append(s)
            all_categories.append("Core (seed)")

    print("    Got {} seed anchors".format(len(seeds)))
    if seeds:
        for s in seeds[:5]:
            print("      \"{}\"{} ".format(s, "" if len(s) <= 50 else "..."))
        if len(seeds) > 5:
            print("      ... and {} more".format(len(seeds) - 5))

    # --- Round 1 (batched by category chunks) ---
    ROUND1_BATCH_SIZE = 7  # categories per LLM call
    round1_n = int(target_n * 1.5)
    cat_batches = [categories[i:i + ROUND1_BATCH_SIZE]
                   for i in range(0, len(categories), ROUND1_BATCH_SIZE)]
    per_batch_n = max(20, round1_n // len(cat_batches))

    print("\n  {}Round 1:{} Generating ~{} anchors in {} batches ({} cats/batch) with {}...".format(
        BOLD, RESET, round1_n, len(cat_batches), ROUND1_BATCH_SIZE, model_name))

    round1_count = 0
    for batch_idx, cat_batch in enumerate(cat_batches):
        batch_n = per_batch_n if batch_idx < len(cat_batches) - 1 else \
            max(per_batch_n, round1_n - round1_count)  # last batch gets remainder

        prompt1 = build_round1_prompt(proposition, cat_batch, batch_n, role=role)
        result = call_llm(config, _system_prompt_round1(role), prompt1)

        batch_count = 0
        if isinstance(result, dict):
            for cat, examples in result.items():
                if isinstance(examples, list):
                    for ex in examples:
                        if isinstance(ex, str) and ex and ex not in all_texts:
                            all_texts.append(ex)
                            all_categories.append(cat)
                            batch_count += 1
        elif isinstance(result, list):
            for i, ex in enumerate(result):
                if isinstance(ex, str) and ex and ex not in all_texts:
                    all_texts.append(ex)
                    all_categories.append(cat_batch[i % len(cat_batch)])
                    batch_count += 1

        round1_count += batch_count
        print("    Batch {}/{}: +{} examples (cats: {})".format(
            batch_idx + 1, len(cat_batches),
            batch_count, ", ".join(c[:25] + "..." if len(c) > 25 else c for c in cat_batch)))

    if round1_count == 0 and len(all_texts) > 0:
        print("    Round 1 returned empty (model may be filtering content).")
        print("    {}Fallback:{} Running per-category generation...".format(BOLD, RESET))

        per_cat_n = max(5, target_n // len(categories))
        for cat in categories:
            fallback_prompt = (
                rc["safety_context"] +
                'PROPOSITION: "{}"\n\n'
                'CATEGORY: "{}"\n\n'
                'Generate {} short test {} (for a safety classifier) that match '
                'this proposition in the style of this category.\n'
                'Mix very short (3-7 words) and medium (8-15 words) examples.\n'
                '{}\n\n'
                'Output ONLY valid JSON:\n'
                '{{"examples": ["example1", "example2", ...]}}'
            ).format(proposition, cat, per_cat_n, rc["example_noun_short"], rc["seed_style"])

            fallback_system = (
                "You are generating test inputs for an AI safety classifier (red-teaming). "
                "Output ONLY valid JSON."
            )

            fb_result = call_llm(config, fallback_system, fallback_prompt)
            fb_examples = []
            if isinstance(fb_result, dict):
                for v in fb_result.values():
                    if isinstance(v, list):
                        fb_examples.extend(v)
            elif isinstance(fb_result, list):
                fb_examples = fb_result

            added = 0
            for ex in fb_examples:
                if ex and ex not in all_texts:
                    all_texts.append(ex)
                    all_categories.append(cat)
                    added += 1
            if added > 0:
                print("      {} \u2714 {} examples".format(cat, added))
            else:
                print("      {} \u2718 empty (filtered)".format(cat))

    print("    Total pool: {} unique examples (incl. seeds)".format(len(all_texts)))

    if len(all_texts) == 0:
        print("\n  ERROR: All generation attempts produced no examples.")
        print("  Possible causes:")
        print("    - Invalid API key (check config.ini [llm] api_key)")
        print("    - Model content filter blocking the request")
        print("    - Model '{}' not available or misspelled".format(
            config.get("llm", "model")))
        print("  Current provider: {}".format(config.get("llm", "provider")))
        sys.exit(1)

    # --- Rounds 2+ ---
    for round_num in range(2, num_rounds + 1):
        if len(all_texts) < 2:
            print("\n  {}Round {}:{} Skipping — need at least 2 examples for analysis.".format(
                BOLD, round_num, RESET))
            continue

        print("\n  {}Round {}:{} Analyzing embedding space...".format(
            BOLD, round_num, RESET))

        embeddings = embed_texts(emb_model, all_texts)

        clusters = find_clusters(embeddings, all_texts, threshold=0.85)
        total_clustered = sum(len(c) for c in clusters)
        print("    Found {} clusters ({} examples too similar)".format(
            len(clusters), total_clustered))

        sim_matrix = cosine_sim_matrix(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        avg_sim = sim_matrix.mean()
        max_sim = sim_matrix.max()
        print("    Avg pairwise similarity: {:.4f}  Max: {:.4f}".format(avg_sim, max_sim))

        pool_cap = max(200, target_n * 3)
        gap_n = min(pool_cap - len(all_texts), max(10, target_n // 2))
        if gap_n <= 0:
            print("    Pool is full ({}/{}). Stopping generation rounds.".format(
                len(all_texts), pool_cap))
            break

        print("    Requesting {} new diverse examples...".format(gap_n))
        prompt_div = build_diversity_prompt(
            proposition, categories, all_texts, clusters, gap_n, role=role)
        result = call_llm(config, _system_prompt_diversity(role), prompt_div)

        new_count = 0
        if isinstance(result, dict):
            for cat, examples in result.items():
                if isinstance(examples, list):
                    for ex in examples:
                        if isinstance(ex, str) and ex and ex not in all_texts:
                            all_texts.append(ex)
                            all_categories.append(cat)
                            new_count += 1
        elif isinstance(result, list):
            for i, ex in enumerate(result):
                if isinstance(ex, str) and ex and ex not in all_texts:
                    all_texts.append(ex)
                    all_categories.append(categories[i % len(categories)])
                    new_count += 1

        print("    Added {} new unique examples (total pool: {})".format(
            new_count, len(all_texts)))

    # --- MMR Selection ---
    print("\n  {}MMR Selection:{} Picking {} most diverse from pool of {}...".format(
        BOLD, RESET, target_n, len(all_texts)))

    all_embeddings = embed_texts(emb_model, all_texts)
    prop_emb = embed_texts(emb_model, [proposition])[0]

    selected_indices = mmr_select(
        all_embeddings, all_texts, all_categories, prop_emb,
        target_n, lambda_param=0.5)

    final_dict = {}
    for idx in selected_indices:
        cat = all_categories[idx]
        if cat not in final_dict:
            final_dict[cat] = []
        final_dict[cat].append(all_texts[idx])

    sel_embeddings = all_embeddings[selected_indices]
    sel_sim = cosine_sim_matrix(sel_embeddings)
    np.fill_diagonal(sel_sim, 0)
    avg_before = cosine_sim_matrix(all_embeddings).mean()

    print("    Pool avg pairwise similarity:  {:.4f}".format(avg_before))
    print("    Selected avg pairwise sim:     {:.4f}".format(sel_sim.mean()))
    print("    Diversity improvement:         {:.1f}%".format(
        (1 - sel_sim.mean() / (avg_before + 1e-10)) * 100))

    # Per-category distribution
    pool_cats = {}
    for cat in all_categories:
        pool_cats[cat] = pool_cats.get(cat, 0) + 1
    print("    Per-category allocation:")
    for cat in final_dict:
        selected_n = len(final_dict[cat])
        pool_n = pool_cats.get(cat, 0)
        print("      {}: {} selected from {} pool".format(
            cat[:50], selected_n, pool_n))

    total = sum(len(v) for v in final_dict.values())
    print("    Final: {} anchors across {} categories".format(
        total, len(final_dict)))

    return final_dict


def generate_negative_anchors(proposition, positive_anchors_dict, target_n, config,
                               num_rounds=None, role="user"):
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

    neg_seed_count = max(25, min(50, target_n // 10))
    if role == "assistant":
        neg_seed_count = max(neg_seed_count, 30)
    print("\n    {}Seed:{} Generating {} core negative anchors...".format(
        BOLD, RESET, neg_seed_count))

    seed_prompt = (
        rc["safety_context"] +
        'PROPOSITION (harmful intent to detect): "{}"\n\n'
        'Generate {} very short (3-10 words) benign {} that use SIMILAR '
        'vocabulary to this proposition but have CLEARLY legitimate intent.\n\n'
        'These should be the most COMMON false positive patterns — {} a '
        'safety system might incorrectly flag.\n\n'
        'CRITICAL:\n'
        '- Use the SAME keywords that appear in harmful {}\n'
        '- Each must be OBVIOUSLY benign to any reasonable person\n'
        '{}\n\n'
        'Output ONLY valid JSON:\n'
        '{{"seeds": ["example1", "example2", ...]}}'
    ).format(proposition, neg_seed_count, rc["example_noun_short"],
             rc["example_noun_short"], rc["example_noun_short"], rc["neg_style"])

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
        selected_indices = mmr_select(
            filt_embs, filtered_texts, filtered_cats, prop_embedding,
            target_n, lambda_param=0.4, directional=True)
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

    return neg_dict


def generate_neutral_anchors(proposition, config, role="user"):
    """
    Generate NEUTRAL anchors — everyday messages completely outside the
    proposition's domain. Used as a baseline: if a message is closer to
    neutral than to positive anchors, it's clearly off-topic.

    Example for "hack into an account":
      Neutral: "What's a good recipe for pasta?", "Help me plan a road trip"
    """
    rc = _role_context(role)
    print("\n  {}Neutral anchors:{} Generating off-topic baseline examples...".format(
        BOLD, RESET))

    neutral_prompt = (
        'PROPOSITION (a topic domain): "{}"\n\n'
        'Generate 30 short, everyday {} that are COMPLETELY UNRELATED to this '
        'proposition and its domain. These should be normal things {}:\n\n'
        '- Cooking, recipes, food\n'
        '- Travel, directions, weather\n'
        '- Creative writing, poetry, stories\n'
        '- Math, science, history\n'
        '- Shopping, product recommendations\n'
        '- Health, fitness, general advice\n'
        '- Entertainment, movies, music, games\n'
        '- Work productivity, emails, scheduling\n'
        '- Programming (unrelated to proposition domain)\n'
        '- Random curiosity questions\n\n'
        'RULES:\n'
        '- NONE of these should use vocabulary from the proposition domain\n'
        '- Mix very short (3-5 words) and medium length (8-15 words)\n'
        '- Keep them natural\n\n'
        'Output ONLY valid JSON:\n'
        '{{"neutral": ["example1", "example2", ...]}}'
    ).format(proposition, rc["example_noun_short"], rc["author_desc"])

    neutral_system = (
        "Generate a list of everyday, off-topic questions and requests. "
        "These must be completely unrelated to the given proposition. "
        "Output ONLY valid JSON."
    )

    result = call_llm(config, neutral_system, neutral_prompt)

    try:
        if isinstance(result, dict):
            parsed = result
        else:
            parsed = json.loads(str(result).strip())
        neutral_list = parsed.get("neutral", [])
        neutral_list = [str(e).strip() for e in neutral_list if str(e).strip()]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print("    {}WARNING: Failed to parse neutral anchors: {}{}".format(YELLOW, e, RESET))
        neutral_list = []

    print("    Generated {} neutral anchors".format(len(neutral_list)))
    return neutral_list


def generate_hard_positives(proposition, positive_anchors_dict, negative_anchors_dict,
                            target_n, config, role="user"):
    """
    Generate HARD POSITIVES — examples that sound benign/legitimate on the
    surface but actually match the proposition's harmful intent.

    These fill the critical gap between standard positives (obviously harmful)
    and negatives (obviously benign). Without hard positives, messages like
    "How can I do a chargeback on a purchase I'm happy with?" get classified
    as benign because they use polite, customer-service language that is
    closer to negative anchors than to standard positives.

    Hard positives are the most important anchors for NLI KNN voting because
    they occupy the boundary region where positive and negative anchor clouds
    meet. They ensure that when an ambiguous-sounding-but-harmful message
    arrives, there are enough positive anchors in its neighborhood to win
    the KNN vote.

    Pipeline:
      1. Show the LLM positive and negative examples to define the boundary
      2. Ask for examples that would fool a naive classifier
      3. Validate with embeddings — keep only those closer to negatives than
         to the positive centroid (truly boundary-straddling)
      4. MMR selection for diversity within the hard set

    Returns dict of category -> examples (merged into positive anchors).
    """
    emb_model_name = config.get("anchors", "embedding_model")
    emb_model = load_embedding_model(emb_model_name)

    rc = _role_context(role)

    # Collect samples for context
    pos_samples = []
    for cat, examples in positive_anchors_dict.items():
        pos_samples.extend(examples[:5])
    pos_sample_text = "\n".join("  - \"{}\"".format(s) for s in pos_samples[:15])

    neg_samples = []
    for cat, examples in negative_anchors_dict.items():
        neg_samples.extend(examples[:5])
    neg_sample_text = "\n".join("  - \"{}\"".format(s) for s in neg_samples[:15])

    print("\n  {}Hard positives:{} Generating boundary-straddling examples...".format(
        BOLD, RESET))
    print("    Target: {} hard positives".format(target_n))

    hard_system = (
        "You are generating HARD POSITIVE test inputs for an AI safety classifier. "
        "These are {} that GENUINELY match a harmful proposition but are phrased "
        "to SOUND like innocent, legitimate requests. They are the most challenging "
        "test cases — designed to fool a classifier that only looks at surface "
        "vocabulary and tone. Output ONLY valid JSON."
    ).format(rc["example_noun_short"])

    # --- Round 1: Main generation ---
    round1_n = int(target_n * 2.0)  # over-generate for MMR selection

    hard_prompt = (
        rc["safety_context"] +
        'PROPOSITION (harmful intent to detect): "{}"\n\n'
        'STANDARD POSITIVE ANCHORS (obviously harmful — a classifier catches these easily):\n'
        '{}\n\n'
        'NEGATIVE ANCHORS (obviously benign — a classifier correctly ignores these):\n'
        '{}\n\n'
        'THE CHALLENGE: Generate {} {} that:\n'
        '1. GENUINELY match the proposition (they really do have harmful intent)\n'
        '2. But are PHRASED to sound like the negative (benign) examples\n'
        '3. Use polite, professional, or casual language that disguises the intent\n'
        '4. Would fool a surface-level classifier into thinking they are benign\n\n'
        'STRATEGIES for making hard positives:\n'
        '- Embed harmful intent in polite customer-service language\n'
        '- Frame harmful requests as legitimate personal needs\n'
        '- Use first-person ownership language ("my account", "my situation")\n'
        '- Add justifying context that makes the request sound reasonable\n'
        '- Mix harmful intent with genuinely benign requests in the same message\n'
        '- Use professional/business framing for harmful activities\n'
        '- Ask "how to" with plausible-sounding innocent reasons\n'
        '- Use understatement or euphemism that a human would recognize as harmful\n\n'
        'CRITICAL: Every example MUST genuinely match the proposition. They must have '
        'real harmful intent — just disguised in innocent-sounding language.\n\n'
        'LENGTH RULES:\n'
        '- At least 30% should be SHORT (5-12 words) — casual, offhand harmful requests\n'
        '- At least 30% should be MEDIUM (13-25 words) — with justifying context\n'
        '- The rest can be LONGER (26+ words) — elaborate scenarios that hide intent\n\n'
        'Output ONLY valid JSON:\n'
        '{{"hard_positives": ["example1", "example2", ...]}}'
    ).format(proposition, pos_sample_text, neg_sample_text,
             round1_n, rc["example_noun_short"])

    result = call_llm(config, hard_system, hard_prompt)

    all_hard = []
    try:
        if isinstance(result, dict):
            if "hard_positives" in result:
                all_hard = result["hard_positives"]
            else:
                for v in result.values():
                    if isinstance(v, list):
                        all_hard.extend(v)
        elif isinstance(result, list):
            all_hard = result
    except Exception:
        pass

    # Deduplicate against existing positives
    existing_pos = set()
    for examples in positive_anchors_dict.values():
        existing_pos.update(examples)
    all_hard = [str(h).strip() for h in all_hard
                if str(h).strip() and str(h).strip() not in existing_pos]

    print("    Round 1: {} hard positives generated".format(len(all_hard)))

    # --- Round 2: Targeted gap-filling ---
    if len(all_hard) >= 5:
        round2_n = max(10, target_n)

        # Analyze which hard positives are closest to negatives (best ones)
        hard_embs = embed_texts(emb_model, all_hard)
        neg_texts_flat = []
        for examples in negative_anchors_dict.values():
            neg_texts_flat.extend(examples)
        if neg_texts_flat:
            neg_embs = embed_texts(emb_model, neg_texts_flat)
            from sentence_transformers.util import cos_sim
            sims_to_neg = cos_sim(hard_embs, neg_embs)
            max_sim_to_neg = sims_to_neg.max(dim=1).values.tolist()

            # Show best examples as models for round 2
            scored = list(zip(max_sim_to_neg, all_hard))
            scored.sort(reverse=True)
            best_hard = [text for _, text in scored[:10]]
            best_hard_text = "\n".join("  - \"{}\"".format(s) for s in best_hard[:8])

            round2_prompt = (
                rc["safety_context"] +
                'PROPOSITION: "{}"\n\n'
                'BEST HARD POSITIVES SO FAR (these successfully sound benign but are harmful):\n'
                '{}\n\n'
                'NEGATIVE (benign) EXAMPLES for reference:\n'
                '{}\n\n'
                'Generate {} MORE hard positive {} using DIFFERENT angles, vocabulary, '
                'and scenarios from the examples above. Each must genuinely match the '
                'proposition but sound innocent.\n\n'
                'Focus on:\n'
                '- Angles and phrasings NOT covered by the examples above\n'
                '- Different lengths (mix short, medium, long)\n'
                '- Different registers (casual, professional, technical, emotional)\n\n'
                'Output ONLY valid JSON:\n'
                '{{"hard_positives": ["example1", "example2", ...]}}'
            ).format(proposition, best_hard_text, neg_sample_text,
                     round2_n, rc["example_noun_short"])

            result2 = call_llm(config, hard_system, round2_prompt)

            round2_hard = []
            try:
                if isinstance(result2, dict):
                    if "hard_positives" in result2:
                        round2_hard = result2["hard_positives"]
                    else:
                        for v in result2.values():
                            if isinstance(v, list):
                                round2_hard.extend(v)
                elif isinstance(result2, list):
                    round2_hard = result2
            except Exception:
                pass

            for h in round2_hard:
                text = str(h).strip()
                if text and text not in existing_pos and text not in all_hard:
                    all_hard.append(text)

            print("    Round 2: +{} (total pool: {})".format(
                len(round2_hard), len(all_hard)))

    if not all_hard:
        print("    {}WARNING: No hard positives generated{}".format(YELLOW, RESET))
        return {}

    # --- Filtering: remove any that are too far from the proposition domain ---
    prop_embedding = embed_texts(emb_model, [proposition])[0]
    all_hard_embs = embed_texts(emb_model, all_hard)

    from sentence_transformers.util import cos_sim
    prop_sims = cos_sim(all_hard_embs, prop_embedding.reshape(1, -1))[:, 0].tolist()

    # Remove examples with very low proposition similarity (off-topic)
    MIN_PROP_SIM = 0.15
    filtered = [(text, sim) for text, sim in zip(all_hard, prop_sims) if sim >= MIN_PROP_SIM]
    removed = len(all_hard) - len(filtered)
    if removed > 0:
        print("    Filtered {} off-topic hard positives (prop_sim < {})".format(
            removed, MIN_PROP_SIM))

    if not filtered:
        print("    {}WARNING: All hard positives filtered out{}".format(YELLOW, RESET))
        return {}

    # --- MMR selection for diversity ---
    final_texts = [t for t, _ in filtered]
    if len(final_texts) > target_n:
        print("    {}MMR Selection:{} Picking {} most diverse from pool of {}...".format(
            BOLD, RESET, target_n, len(final_texts)))
        filt_embs = embed_texts(emb_model, final_texts)
        cats_dummy = ["Hard positive (boundary)"] * len(final_texts)
        selected_indices = mmr_select(
            filt_embs, final_texts, cats_dummy, prop_embedding,
            target_n, lambda_param=0.4)
        final_texts = [final_texts[i] for i in selected_indices]

    # Build output dict
    hard_dict = {"Hard positive (boundary)": final_texts}

    total = len(final_texts)
    print("    Final: {} hard positives".format(total))

    # Report proximity stats
    if final_texts:
        final_embs = embed_texts(emb_model, final_texts)
        final_prop_sims = cos_sim(final_embs, prop_embedding.reshape(1, -1))[:, 0].tolist()
        print("    Cosine to proposition: min={:.3f}  avg={:.3f}  max={:.3f}".format(
            min(final_prop_sims), sum(final_prop_sims) / len(final_prop_sims),
            max(final_prop_sims)))

        # Show how many are closer to negatives than positive centroid
        if negative_anchors_dict:
            neg_texts_flat = []
            for examples in negative_anchors_dict.values():
                neg_texts_flat.extend(examples)
            neg_embs = embed_texts(emb_model, neg_texts_flat)
            sims_to_neg = cos_sim(final_embs, neg_embs).max(dim=1).values.tolist()

            pos_texts_flat = []
            for examples in positive_anchors_dict.values():
                pos_texts_flat.extend(examples)
            pos_embs = embed_texts(emb_model, pos_texts_flat)
            sims_to_pos = cos_sim(final_embs, pos_embs).max(dim=1).values.tolist()

            boundary_count = sum(1 for ns, ps in zip(sims_to_neg, sims_to_pos)
                                 if ns >= ps - 0.05)
            print("    Boundary quality: {}/{} are within 0.05 of negative anchors".format(
                boundary_count, total))

    return hard_dict


# ---------------------------------------------------------------------------
# Evaluator template
# ---------------------------------------------------------------------------

EVALUATOR_TEMPLATE = r'''#!/usr/bin/env python3
"""
Semantic Anchor Evaluator: %%NAME%%
%%TITLE_UNDERLINE%%

Auto-generated by semantic_anchor_generator.py

Loads anchors from: anchors_list_%%NAME%%.json

Usage:
  python semantic_anchor_%%NAME%%.py                       # Interactive (cosine mode)
  python semantic_anchor_%%NAME%%.py --mode nli             # NLI entailment scoring
  python semantic_anchor_%%NAME%%.py --mode hybrid          # Weighted blend: NLI + cosine KNN (recommended)
  python semantic_anchor_%%NAME%%.py --mode llm             # LLM-as-judge scoring
  python semantic_anchor_%%NAME%%.py --compare              # Compare all modes side-by-side
  python semantic_anchor_%%NAME%%.py --compare -f input.txt # Compare all modes on file
  python semantic_anchor_%%NAME%%.py --verbose              # Full table
  python semantic_anchor_%%NAME%%.py --show-examples        # View anchors
  python semantic_anchor_%%NAME%%.py --file input.txt       # File mode (### separated)
  python semantic_anchor_%%NAME%%.py --graph                # 3D anchor spread visualization

Scoring modes:
  cosine:  KNN voting over unified positive/negative anchor pool. Fast, no NLI needed.
  nli:     NLI entailment scoring with gap-gated anchor analysis. Handles paraphrases well.
  hybrid:  Weighted blend of NLI + cosine KNN (default 75/25). Recommended.
  llm:     LLM-as-judge. Requires config_%%NAME%%.ini with provider, model, api_key.
  compare: Runs all modes side-by-side with verdict, score, and top anchors.

Requirements:
  pip install sentence-transformers
  pip install autocorrect                   (only for --spellcheck)
  pip install httpx                         (only for --mode llm)

Long message handling:
  All modes use proposition-guided extraction for messages > EXTRACTION_MIN_WORDS.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Configuration loader — reads from anchors_list_<name>.json
# ---------------------------------------------------------------------------

SCRIPT_NAME = "%%NAME%%"

def _find_anchors_file():
    """Locate the anchors JSON file relative to this script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = "anchors_list_{}.json".format(SCRIPT_NAME)
    # Look in same directory as the script
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    # Look in parent directory (if script is in a subfolder)
    parent = os.path.join(os.path.dirname(script_dir), filename)
    if os.path.exists(parent):
        return parent
    return None


def _load_anchors_config():
    """Load proposition, anchors, thresholds, negative and neutral anchors from JSON."""
    path = _find_anchors_file()
    if path is None:
        print("\n  ERROR: Anchors file not found: anchors_list_{}.json".format(SCRIPT_NAME))
        print("  Expected location: same directory as this script")
        print("  Generate it with: python semantic_anchor_generator.py -name {} -ga".format(
            SCRIPT_NAME))
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})

    return (
        data["proposition"],
        data["anchors"],
        data.get("match_threshold", 0.55),
        data.get("warning_threshold", 0.45),
        data.get("negative_anchors", {}),
        data.get("neutral_anchors", []),
        metadata.get("embedding_model", "all-mpnet-base-v2"),
        metadata.get("nli_model", "cross-encoder/nli-deberta-v3-large"),
        data.get("role", "user"),
    )


def _load_evaluator_config():
    """
    Load config_<name>.ini and override model names from JSON defaults.
    Priority: config_<name>.ini > anchors_list JSON metadata > hardcoded defaults.
    """
    import configparser as _cp

    # Start with JSON defaults
    (proposition, anchors, match_thresh, warn_thresh,
     neg_anchors, neutral_anchors,
     json_emb_model, json_nli_model, role) = _load_anchors_config()

    emb_model = json_emb_model
    nli_model = json_nli_model
    knn_size = 20  # default KNN neighborhood size
    hybrid_nli_w = 0.75  # default hybrid blend: 75% NLI + 25% cosine
    nli_abstain_margin = 0.15  # default: abstain when evidence margin < this
    nli_retrieve_k = 40  # default: cosine pre-filter top 40 from unified pool
    nli_vote_k = 20      # default: vote on top 20 by NLI score
    nli_fwd_weight = 0.7 # default: 70% forward, 30% backward
    adaptive_hybrid = True  # default: shift weight toward cosine when NLI is uncertain
    emb_source = "anchors JSON"
    nli_source = "anchors JSON"

    # Look for config_<n>.ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config_{}.ini".format(SCRIPT_NAME))
    if not os.path.exists(cfg_path):
        # Also check parent directory
        cfg_path = os.path.join(os.path.dirname(script_dir),
                                "config_{}.ini".format(SCRIPT_NAME))

    if os.path.exists(cfg_path):
        cfg = _cp.ConfigParser()
        cfg.read(cfg_path)

        # [models] section takes priority, then [llm_judge]
        for section in ("models", "llm_judge"):
            if cfg.has_section(section):
                val = cfg.get(section, "embedding_model", fallback=None)
                if val and val.strip():
                    emb_model = val.strip()
                    emb_source = "config_{}.ini [{}]".format(SCRIPT_NAME, section)
                val = cfg.get(section, "nli_model", fallback=None)
                if val and val.strip():
                    nli_model = val.strip()
                    nli_source = "config_{}.ini [{}]".format(SCRIPT_NAME, section)

        # [thresholds] section — match, warning, knn_size, hybrid weights
        if cfg.has_section("thresholds"):
            val = cfg.get("thresholds", "match_threshold", fallback=None)
            if val and val.strip():
                match_thresh = float(val.strip())
            val = cfg.get("thresholds", "warning_threshold", fallback=None)
            if val and val.strip():
                warn_thresh = float(val.strip())
            val = cfg.get("thresholds", "knn_size", fallback=None)
            if val and val.strip():
                knn_size = int(val.strip())
            val = cfg.get("thresholds", "hybrid_nli_weight", fallback=None)
            if val and val.strip():
                hybrid_nli_w = float(val.strip())
            val = cfg.get("thresholds", "nli_abstain_margin", fallback=None)
            if val and val.strip():
                nli_abstain_margin = float(val.strip())
            val = cfg.get("thresholds", "nli_retrieve_k", fallback=None)
            if val and val.strip():
                nli_retrieve_k = int(val.strip())
            val = cfg.get("thresholds", "nli_vote_k", fallback=None)
            if val and val.strip():
                nli_vote_k = int(val.strip())
            val = cfg.get("thresholds", "nli_fwd_weight", fallback=None)
            if val and val.strip():
                nli_fwd_weight = float(val.strip())
            val = cfg.get("thresholds", "adaptive_hybrid", fallback=None)
            if val and val.strip():
                adaptive_hybrid = val.strip().lower() in ("true", "1", "yes")

    # Store sources for startup logging
    _model_sources["embedding"] = (emb_model, emb_source)
    _model_sources["nli"] = (nli_model, nli_source)

    return (proposition, anchors, match_thresh, warn_thresh,
            neg_anchors, neutral_anchors, emb_model, nli_model, role, knn_size,
            hybrid_nli_w, nli_abstain_margin, nli_retrieve_k, nli_vote_k,
            nli_fwd_weight, adaptive_hybrid)


# Model source tracking — filled by _load_evaluator_config, read by main()
_model_sources = {}

(PROPOSITION, ANCHORS, MATCH_THRESHOLD, WARNING_THRESHOLD,
 NEGATIVE_ANCHORS, NEUTRAL_ANCHORS,
 EMBEDDING_MODEL, NLI_MODEL, ANCHOR_ROLE, COSINE_KNN_K,
 HYBRID_NLI_WEIGHT, NLI_ABSTAIN_MARGIN,
 NLI_RETRIEVE_K, NLI_VOTE_K, NLI_FWD_WEIGHT,
 ADAPTIVE_HYBRID) = _load_evaluator_config()

# Declarative prefix for NLI: converts questions/messages into statement form
# Role-aware: "The user states: " for user messages, "The assistant states: " for assistant
NLI_DECLARATIVE_PREFIX = ("The assistant states: " if ANCHOR_ROLE == "assistant"
                          else "The user states: ")

# ---------------------------------------------------------------------------
# Reranking config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Proposition-guided extraction config
# ---------------------------------------------------------------------------

EXTRACTION_MIN_WORDS = 30       # only extract for messages longer than this
EXTRACTION_RELEVANCE_TAU = 0.18 # low threshold — keep any topically related sentence
EXTRACTION_MIN_SELECTED = 1     # need at least this many sentences to use extraction
EXTRACTION_MAX_WINDOW = 3       # also try sliding windows of this size for adjacent patterns

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
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
# Spell correction (disabled by default — enable via --spellcheck or config)
# ---------------------------------------------------------------------------

SPELLCHECK_ENABLED = False  # Set by CLI flag or config_<name>.ini

_spellchecker = None

def _load_spellchecker():
    global _spellchecker
    if _spellchecker is not None:
        return _spellchecker
    try:
        from autocorrect import Speller
        _spellchecker = Speller(lang="en")
        return _spellchecker
    except ImportError:
        pass
    return None


def correct_spelling(text):
    """Correct spelling if SPELLCHECK_ENABLED, otherwise pass through."""
    if not SPELLCHECK_ENABLED:
        return text, []

    checker = _load_spellchecker()
    if checker is None:
        return text, []

    corrected = checker(text)
    if corrected == text:
        return text, []

    orig_words = text.split()
    corr_words = corrected.split()
    changes = []
    for o, c in zip(orig_words, corr_words):
        if o.lower() != c.lower():
            changes.append((o, c))
    return corrected, changes


# ---------------------------------------------------------------------------
# Sentence splitting (robust, always splits)
# ---------------------------------------------------------------------------

def _split_into_sentences(text):
    """
    Split text into sentences. Always splits regardless of length.
    Handles common abbreviations and edge cases.
    """
    import re
    text = text.strip()
    if not text:
        return []

    # Split on sentence boundaries: .!? followed by space+uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Also split on semicolons and long conjunctions that separate clauses
        sub_parts = re.split(r';\s+|(?<=\.)\s+(?:Also|Separately|Additionally|Furthermore|Moreover),?\s+', p)
        for sp in sub_parts:
            sp = sp.strip()
            if sp:
                sentences.append(sp)

    # Merge very short fragments (< 3 words) with previous
    merged = []
    for s in sentences:
        if merged and len(s.split()) < 3:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    return merged if merged else [text]


# ---------------------------------------------------------------------------
# Proposition-Guided Extraction (core long-message defense)
# ---------------------------------------------------------------------------

def extract_relevant_chunks(model, message, all_embeddings, all_texts):
    """
    Proposition-guided extraction for long messages.

    Instead of embedding the full message (which dilutes signal when harmful
    intent is distributed across innocent-looking sentences), this function:

    1. Splits the message into individual sentences
    2. Embeds each sentence independently
    3. Scores each sentence against ALL anchors (max cosine to any anchor)
    4. Selects sentences above a low relevance threshold tau
    5. Returns the composed relevant chunk for downstream scoring

    Also computes sliding windows to catch adjacent-sentence patterns.

    Returns:
        dict with keys:
            'extracted_text': str - the composed relevant sentences
            'was_extracted': bool - whether extraction was applied
            'all_sentences': list[str] - all sentences from the message
            'selected_indices': list[int] - which sentences were selected
            'sentence_scores': list[float] - max-anchor score per sentence
            'method': str - 'extraction' or 'passthrough'
            'sliding_window_text': str - best sliding window chunk
            'sliding_window_score': float - score of best window
    """
    import numpy as np
    from sentence_transformers.util import cos_sim

    words = message.split()
    result = {
        'extracted_text': message,
        'was_extracted': False,
        'all_sentences': [message],
        'selected_indices': [0],
        'sentence_scores': [0.0],
        'method': 'passthrough',
        'sliding_window_text': message,
        'sliding_window_score': 0.0,
    }

    # Only extract for messages above the word threshold
    if len(words) <= EXTRACTION_MIN_WORDS:
        return result

    # Step 1: Split into sentences
    sentences = _split_into_sentences(message)
    if len(sentences) <= 1:
        return result

    # Step 2: Embed all sentences in one batch
    sent_embeddings = model.encode(sentences, show_progress_bar=False)

    # Step 3: Score each sentence against all anchors (max cosine to any anchor)
    # This is the proposition-guided relevance filter
    sim_matrix = cos_sim(sent_embeddings, all_embeddings)  # (n_sents, n_anchors)
    sentence_scores = sim_matrix.max(dim=1).values.tolist()  # max anchor score per sentence

    # Step 4: Select sentences above the relevance threshold
    selected_indices = [
        i for i, score in enumerate(sentence_scores)
        if score >= EXTRACTION_RELEVANCE_TAU
    ]

    # Step 5: Sliding window analysis (catches adjacent-sentence patterns)
    best_window_score = 0.0
    best_window_text = message
    window_size = min(EXTRACTION_MAX_WINDOW, len(sentences))

    for ws in range(2, window_size + 1):
        for start in range(len(sentences) - ws + 1):
            window_text = " ".join(sentences[start:start + ws])
            window_emb = model.encode([window_text], show_progress_bar=False)
            window_sims = cos_sim(window_emb, all_embeddings)[0]
            window_max = float(window_sims.max())
            if window_max > best_window_score:
                best_window_score = window_max
                best_window_text = window_text

    # Step 6: Build the extracted chunk
    if len(selected_indices) >= EXTRACTION_MIN_SELECTED:
        extracted_text = " ".join(sentences[i] for i in selected_indices)
        result.update({
            'extracted_text': extracted_text,
            'was_extracted': True,
            'all_sentences': sentences,
            'selected_indices': selected_indices,
            'sentence_scores': sentence_scores,
            'method': 'extraction',
            'sliding_window_text': best_window_text,
            'sliding_window_score': best_window_score,
        })
    else:
        # Not enough relevant sentences found — fall back to full message
        # but still report the analysis
        result.update({
            'all_sentences': sentences,
            'selected_indices': selected_indices,
            'sentence_scores': sentence_scores,
            'sliding_window_text': best_window_text,
            'sliding_window_score': best_window_score,
        })

    return result


def _display_extraction_info(extraction, verbose=False):
    """Print extraction diagnostic info."""
    if not extraction['was_extracted']:
        return

    n_total = len(extraction['all_sentences'])
    n_selected = len(extraction['selected_indices'])
    n_dropped = n_total - n_selected

    print("  {}Extraction:{} {} sentences \u2192 {} relevant, {} noise dropped".format(
        CYAN, RESET, n_total, n_selected, n_dropped))

    if verbose:
        for i, (sent, score) in enumerate(zip(
                extraction['all_sentences'], extraction['sentence_scores'])):
            marker = "\u2714" if i in extraction['selected_indices'] else "\u2718"
            disp = sent if len(sent) <= 60 else sent[:57] + "..."
            score_color = GREEN if score < EXTRACTION_RELEVANCE_TAU else YELLOW
            print("    {} {}{:.3f}{} \"{}\"".format(
                marker, score_color, score, RESET, disp))

    if extraction['sliding_window_score'] > 0:
        ws = extraction['sliding_window_text']
        ws_disp = ws if len(ws) <= 70 else ws[:67] + "..."
        print("  {}Best window:{} score={:.4f} \"{}\"".format(
            DIM, RESET, extraction['sliding_window_score'], ws_disp))


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
    print("  Loading embedding model: {}...".format(EMBEDDING_MODEL))
    return SentenceTransformer(EMBEDDING_MODEL)


def prepare_anchors(model):
    all_texts = []
    all_categories = []
    for cat, examples in ANCHORS.items():
        for ex in examples:
            all_texts.append(ex)
            all_categories.append(cat)
    print("  Embedding {} positive anchors...".format(len(all_texts)))
    all_embeddings = model.encode(all_texts, show_progress_bar=False)

    # Negative anchors (contrastive)
    neg_texts = []
    neg_categories = []
    neg_embeddings = None
    for cat, examples in NEGATIVE_ANCHORS.items():
        for ex in examples:
            neg_texts.append(ex)
            neg_categories.append(cat)
    if neg_texts:
        print("  Embedding {} negative anchors...".format(len(neg_texts)))
        neg_embeddings = model.encode(neg_texts, show_progress_bar=False)
    else:
        print("  No negative anchors (contrastive scoring disabled)")

    # Neutral anchors (off-topic baseline)
    neutral_embeddings = None
    if NEUTRAL_ANCHORS:
        print("  Embedding {} neutral anchors...".format(len(NEUTRAL_ANCHORS)))
        neutral_embeddings = model.encode(NEUTRAL_ANCHORS, show_progress_bar=False)

    print("  Ready.\n")
    return (all_texts, all_categories, all_embeddings,
            neg_texts, neg_categories, neg_embeddings,
            neutral_embeddings)


# ---------------------------------------------------------------------------
# Contrastive adjustment (gap-based)
# ---------------------------------------------------------------------------

def _contrastive_adjust(pos_score, pos_cos, neg_cos, neutral_cos=0.0):
    """
    Adjust a score using the contrastive cosine gap.

    Uses COSINE similarity (surface-level) to compare how close the message
    is to positive vs negative anchors, regardless of what scoring mode
    produced pos_score (cosine or NLI).

    The penalty curve is smooth through gap=0, with the key transition zone
    between gap=-0.01 and gap=-0.02 where harmful and benign messages
    typically separate. Small negative gaps (>-0.01) are treated as cosine
    noise, not a definitive benign signal.

    Penalty zones:
      gap >= 0.10     →  0% (clearly closer to positive)
      gap -0.01..0.10 →  0..15% (mild — noise zone includes near-zero gaps)
      gap -0.02..-0.01→  15..55% (steep transition — the boundary zone)
      gap -0.05..-0.02→  55..80% (heavy — clearly closer to negative)
      gap < -0.05     →  80..95% (very heavy)

    Returns (adjusted_score, neg_cos).
    """
    # Off-topic check: closer to neutral than to domain anchors
    if neutral_cos > 0.4 and neutral_cos > pos_cos and neutral_cos > neg_cos:
        return pos_score * 0.15, neg_cos

    # Negatives not close at all — no adjustment
    if neg_cos < 0.4:
        return pos_score, neg_cos

    gap = pos_cos - neg_cos

    if gap >= 0.10:
        # Clearly closer to positive anchors — no penalty
        return pos_score, neg_cos

    elif gap >= -0.01:
        # Mild zone: treats small negative gaps as noise
        # 0% at gap=0.10, 15% at gap=-0.01
        fade = (0.10 - gap) / 0.11  # 0..1
        penalty = fade * 0.15

    elif gap >= -0.02:
        # Steep transition: the boundary between harmful and benign
        # 15% at gap=-0.01, 55% at gap=-0.02
        fade = (-0.01 - gap) / 0.01  # 0..1
        penalty = 0.15 + fade * 0.40

    elif gap >= -0.05:
        # Heavy: clearly closer to negative anchors
        # 55% at gap=-0.02, 80% at gap=-0.05
        fade = (-0.02 - gap) / 0.03  # 0..1
        penalty = 0.55 + fade * 0.25

    else:
        # Very heavy: 80% at gap=-0.05, up to 95%
        penalty = min(0.95, 0.80 + (abs(gap) - 0.05) * 2.0)

    return pos_score * (1.0 - penalty), neg_cos


def score_message(model, message, all_embeddings, all_texts, all_categories,
                  neg_embeddings=None, neg_texts=None, neutral_embeddings=None,
                  use_extraction=True):
    """
    Cosine KNN voting: merge pos + neg anchors, find K nearest, count positive ratio.

    Returns (results_list, extraction_info, knn_info) where knn_info is a dict with:
      knn_score, pos_in_k, k, max_pos, max_neg, top_k_details
    """
    from sentence_transformers.util import cos_sim
    import numpy as np

    extraction = None
    if use_extraction:
        extraction = extract_relevant_chunks(model, message, all_embeddings, all_texts)

    # Compute cosine sims for each view
    if extraction and extraction['was_extracted']:
        emb_extracted = model.encode(extraction['extracted_text'])
        emb_full = model.encode(message)
        emb_window = model.encode(extraction['sliding_window_text'])
        views = [emb_extracted, emb_full, emb_window]
    else:
        views = [model.encode(message)]

    # --- Positive anchor sims (best across views) ---
    n_pos = len(all_texts)
    pos_sims = [0.0] * n_pos
    for emb in views:
        sims = cos_sim(emb, all_embeddings)[0].tolist()
        for i, s in enumerate(sims):
            if s > pos_sims[i]:
                pos_sims[i] = s

    # --- Negative anchor sims (best across views) ---
    n_neg = len(neg_texts) if neg_texts else 0
    neg_sims = [0.0] * n_neg
    if neg_embeddings is not None and neg_texts:
        for emb in views:
            sims = cos_sim(emb, neg_embeddings)[0].tolist()
            for i, s in enumerate(sims):
                if s > neg_sims[i]:
                    neg_sims[i] = s

    # --- KNN voting: merge and find top-K ---
    # Build unified pool: (similarity, is_positive, text, category)
    pool = []
    for i in range(n_pos):
        pool.append((pos_sims[i], True, all_texts[i], all_categories[i]))
    for i in range(n_neg):
        pool.append((neg_sims[i], False, neg_texts[i], "NEGATIVE"))

    pool.sort(key=lambda x: x[0], reverse=True)
    top_k = pool[:COSINE_KNN_K]

    pos_in_k = sum(1 for _, is_pos, _, _ in top_k if is_pos)
    knn_score = pos_in_k / len(top_k) if top_k else 0.0
    max_pos = max(pos_sims) if pos_sims else 0.0
    max_neg = max(neg_sims) if neg_sims else 0.0

    knn_info = {
        "knn_score": knn_score,
        "pos_in_k": pos_in_k,
        "k": len(top_k),
        "max_pos": max_pos,
        "max_neg": max_neg,
    }

    # Results sorted by positive similarity (for display)
    results = sorted(
        zip(pos_sims, all_texts, all_categories),
        key=lambda x: x[0], reverse=True)
    return results, extraction, knn_info


# ---------------------------------------------------------------------------
# NLI cross-encoder scoring
# ---------------------------------------------------------------------------

_cross_encoder = None
_entailment_idx = None
_contradiction_idx = None

def _load_cross_encoder():
    """Lazy-load NLI cross-encoder model for intent matching."""
    global _cross_encoder, _entailment_idx, _contradiction_idx
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("\n  ERROR: sentence-transformers version doesn't support CrossEncoder.")
        print("  Run: pip install -U sentence-transformers")
        sys.exit(1)
    import numpy as np
    print("  Loading NLI cross-encoder model: {}...".format(NLI_MODEL))
    _cross_encoder = CrossEncoder(NLI_MODEL)

    cal_pairs = [
        ["A person is building a weapon", "Someone wants to make a weapon"],       # entailment
        ["I love eating pizza", "Someone wants to make a weapon"],                  # neutral
        ["I want to protect people from weapons", "Someone wants to make a weapon"],  # contradiction
    ]
    cal_logits = np.array(_cross_encoder.predict(cal_pairs, apply_softmax=False))
    cal_exp = np.exp(cal_logits - np.max(cal_logits, axis=1, keepdims=True))
    cal_probs = cal_exp / cal_exp.sum(axis=1, keepdims=True)

    entail_row = cal_probs[0]   # should have high entailment
    neutral_row = cal_probs[1]  # should have high neutral
    contra_row = cal_probs[2]   # should have high contradiction

    # Find entailment index: highest on entail_row
    best_idx = int(np.argmax(entail_row))
    if neutral_row[best_idx] < 0.3 and contra_row[best_idx] < 0.3:
        _entailment_idx = best_idx
    else:
        gaps = entail_row - np.maximum(neutral_row, contra_row)
        _entailment_idx = int(np.argmax(gaps))

    # Find contradiction index: highest on contra_row, excluding entailment_idx
    remaining = list(range(cal_probs.shape[1]))
    remaining.remove(_entailment_idx)
    _contradiction_idx = max(remaining, key=lambda i: contra_row[i])

    print("  Entailment index: {} (cal: entail={:.3f}, neutral={:.3f}, contra={:.3f})".format(
        _entailment_idx,
        entail_row[_entailment_idx],
        neutral_row[_entailment_idx],
        contra_row[_entailment_idx]))
    print("  Contradiction index: {} (cal: entail={:.3f}, neutral={:.3f}, contra={:.3f})".format(
        _contradiction_idx,
        entail_row[_contradiction_idx],
        neutral_row[_contradiction_idx],
        contra_row[_contradiction_idx]))

    return _cross_encoder


def _nli_entailment_scores(pairs):
    """Compute entailment probability for each pair using NLI cross-encoder."""
    import numpy as np
    xenc = _load_cross_encoder()
    logits = xenc.predict(pairs, apply_softmax=False)
    logits = np.array(logits)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs[:, _entailment_idx].tolist()


def _nli_net_scores(pairs):
    """
    Compute net NLI score = max(0, entailment - contradiction) for each pair.

    Uses the full NLI 3-class signal:
    - A→B entailment high, contradiction low → strong positive score
    - A→not(B) contradiction high, entailment low → score = 0
    - A unrelated to B (neutral) → both low → score ≈ 0

    This is equivalent to running both A→B and A→not(B) checks in a single call.
    """
    import numpy as np
    xenc = _load_cross_encoder()
    logits = xenc.predict(pairs, apply_softmax=False)
    logits = np.array(logits)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    entail = probs[:, _entailment_idx]
    contra = probs[:, _contradiction_idx]
    net = np.maximum(0.0, entail - contra)
    return net.tolist()


def _nli_proposition_score(views, proposition):
    """
    Score message views directly against the proposition using NLI net scores.

    Two key improvements over naive entailment-only:
    1. Declarative prefix: wraps views in "The user states: ..." so NLI sees
       declarative→declarative instead of question→statement (fixes structural mismatch)
    2. Net score = max(0, entailment - contradiction): uses full 3-class NLI signal.
       Equivalent to running A→B and A→not(B) checks in a single call.

    Scores bidirectionally (message→prop and prop→message), takes max per view,
    then returns the overall max across all views.

    Returns (prop_score, best_view_text)
    """
    best_score = 0.0
    best_view = ""

    for view_text, view_label in views:
        # Declarative prefix: convert questions to statements for NLI
        decl_view = NLI_DECLARATIVE_PREFIX + view_text
        pairs = [
            [decl_view, proposition],   # message entails proposition?
            [proposition, decl_view],   # proposition entails message?
        ]
        scores = _nli_net_scores(pairs)
        view_score = max(scores)
        if view_score > best_score:
            best_score = view_score
            best_view = view_label

    return best_score, best_view


# ---------------------------------------------------------------------------
# NLI KNN Voting scoring (--nli mode)
# ---------------------------------------------------------------------------

# NLI KNN constants — loaded from config_<n>.ini [thresholds] section
# NLI_RETRIEVE_K: cosine pre-filter size (default 40)
# NLI_VOTE_K: vote neighborhood size (default 20)
# NLI_FWD_WEIGHT: asymmetric forward weight (default 0.7)
# NLI_ABSTAIN_MARGIN: abstain when evidence margin below this (default 0.15)
# ADAPTIVE_HYBRID: shift weight toward cosine when NLI is uncertain (default true)
COSINE_TOP_K_FRAC = 0.50  # fraction of anchors to average (top-50%)
COSINE_TOP_K_MIN = 10     # minimum anchors to average



def _top_k_cosine_avg(scores, k=None):
    """Average of top-K cosine scores. K defaults to 50% of total (min 10)."""
    if not scores:
        return 0.0
    if k is None:
        k = max(COSINE_TOP_K_MIN, int(len(scores) * COSINE_TOP_K_FRAC))
    sorted_scores = sorted(scores, reverse=True)
    top = sorted_scores[:min(k, len(sorted_scores))]
    return sum(top) / len(top)


def _score_nli_core(message, all_texts, all_categories, model=None,
                    all_embeddings=None, neg_texts=None, neg_embeddings=None,
                    neutral_embeddings=None,
                    do_spellcheck=True, use_extraction=True):
    """
    NLI KNN voting: retrieve candidates by cosine, re-rank by NLI, vote.

    Pipeline:
      1. Spell check + extraction + build views
      2. Proposition NLI (net scores, declarative prefix)
      3. Cosine retrieval: top NLI_RETRIEVE_K from unified pool (pos + neg)
      4. NLI re-rank: asymmetric net scores (E-C) on retrieved candidates
         - 70% forward (message->anchor) + 30% backward (anchor->message)
      5. Vote on top NLI_VOTE_K by NLI score:
         - pos_evidence = sum of NLI scores for positive anchors
         - neg_evidence = sum of NLI scores for negative anchors
         - nli_knn_score = pos_evidence / (pos_evidence + neg_evidence + epsilon)
      6. Final score = max(prop_score, nli_knn_score)
      7. Abstain detection when evidence margin is small

    Returns (results, corrected, corrections, extraction, neg_cos, final_score, debug)
    """
    # Step 0: spell correction
    corrected = message
    corrections = []
    if do_spellcheck:
        corrected, corrections = correct_spelling(message)

    # Step 1: proposition-guided extraction
    extraction = None
    if use_extraction and model is not None and all_embeddings is not None:
        extraction = extract_relevant_chunks(model, corrected, all_embeddings, all_texts)

    _load_cross_encoder()

    # Step 2: Build views
    views = []
    if extraction and extraction['was_extracted']:
        views.append((extraction['extracted_text'], "[extracted chunk]"))
        if extraction['sliding_window_score'] > 0:
            views.append((extraction['sliding_window_text'], "[sliding window]"))
        for sent in extraction['all_sentences']:
            label = sent if len(sent) <= 40 else sent[:37] + "..."
            views.append((sent, label))
        views.append((corrected, "[full message]"))
    else:
        sentences = _split_into_sentences(corrected)
        for sent in sentences:
            label = sent if len(sent) <= 40 else sent[:37] + "..."
            views.append((sent, label))

    # Step 3: Proposition NLI (~2 calls per view)
    prop_score, prop_view = _nli_proposition_score(views, PROPOSITION)

    # Step 4: Cosine retrieval from UNIFIED pool (pos + neg)
    neg_cos = 0.0
    pos_cos = 0.0
    best_view_text = views[0][0]
    unified_pool = []

    n_pos = len(all_texts)
    n_neg = len(neg_texts) if neg_texts else 0

    if model is not None and all_embeddings is not None:
        from sentence_transformers.util import cos_sim

        pos_cosines = [0.0] * n_pos
        neg_cosines = [0.0] * n_neg

        for view_text, _ in views:
            emb = model.encode(view_text)

            # Positive cosines
            p_sims = cos_sim(emb, all_embeddings)[0].tolist()
            view_pos_cos = _top_k_cosine_avg(p_sims)
            if view_pos_cos > pos_cos:
                pos_cos = view_pos_cos
                best_view_text = view_text
            for i, s in enumerate(p_sims):
                if s > pos_cosines[i]:
                    pos_cosines[i] = s

            # Negative cosines
            if neg_embeddings is not None and neg_texts:
                n_sims = cos_sim(emb, neg_embeddings)[0].tolist()
                view_neg_cos = _top_k_cosine_avg(n_sims)
                if view_neg_cos > neg_cos:
                    neg_cos = view_neg_cos
                for i, s in enumerate(n_sims):
                    if s > neg_cosines[i]:
                        neg_cosines[i] = s

        # Build unified pool
        for i in range(n_pos):
            unified_pool.append({
                "idx": i, "is_positive": True,
                "text": all_texts[i], "category": all_categories[i],
                "cosine": pos_cosines[i],
            })
        for i in range(n_neg):
            unified_pool.append({
                "idx": n_pos + i, "is_positive": False,
                "text": neg_texts[i], "category": "NEGATIVE",
                "cosine": neg_cosines[i],
            })

    gap = pos_cos - neg_cos

    # Sort by cosine, take top NLI_RETRIEVE_K for NLI re-ranking
    unified_pool.sort(key=lambda x: x["cosine"], reverse=True)
    retrieve_k = min(NLI_RETRIEVE_K, len(unified_pool))
    candidates = unified_pool[:retrieve_k]

    # Step 5: NLI on retrieved candidates -- asymmetric net scores (E - C)
    # Forward: message->anchor (does message entail anchor?)
    # Backward: anchor->message (does anchor entail message?)
    # Combined: 0.7 x forward + 0.3 x backward
    if candidates:
        cand_texts = [c["text"] for c in candidates]
        pairs_fwd = [[best_view_text, t] for t in cand_texts]
        pairs_bwd = [[t, best_view_text] for t in cand_texts]
        net_fwd = _nli_net_scores(pairs_fwd)
        net_bwd = _nli_net_scores(pairs_bwd)

        bwd_weight = 1.0 - NLI_FWD_WEIGHT
        for i, cand in enumerate(candidates):
            cand["nli_score"] = NLI_FWD_WEIGHT * net_fwd[i] + bwd_weight * net_bwd[i]
            cand["nli_fwd"] = net_fwd[i]
            cand["nli_bwd"] = net_bwd[i]

    # Step 6: Re-rank by NLI score, vote on top NLI_VOTE_K
    candidates.sort(key=lambda x: x.get("nli_score", 0), reverse=True)
    vote_k = min(NLI_VOTE_K, len(candidates))
    voters = candidates[:vote_k]

    pos_evidence = 0.0
    neg_evidence = 0.0
    pos_in_k = 0
    neg_in_k = 0
    voter_details = []

    for v in voters:
        nli_s = v.get("nli_score", 0)
        if v["is_positive"]:
            pos_evidence += nli_s
            pos_in_k += 1
        else:
            neg_evidence += nli_s
            neg_in_k += 1
        voter_details.append({
            "text": v["text"][:60], "category": v["category"],
            "is_positive": v["is_positive"], "nli_score": nli_s,
            "cosine": v["cosine"],
        })

    total_evidence = pos_evidence + neg_evidence + 1e-10
    nli_knn_score = pos_evidence / total_evidence

    # Step 7: Evidence margin for abstain detection
    evidence_margin = abs(pos_evidence - neg_evidence) / total_evidence
    abstain = evidence_margin < NLI_ABSTAIN_MARGIN

    # Step 8: Final score = max(prop_score, nli_knn_score)
    # No gating needed -- NLI KNN handles refusals via negative anchor voting
    final_score = max(prop_score, nli_knn_score)

    if abstain:
        action = "abstain(margin={:.2f})".format(evidence_margin)
    elif final_score >= MATCH_THRESHOLD:
        action = "match"
    elif final_score >= WARNING_THRESHOLD:
        action = "warning"
    else:
        action = "no_match"

    debug = {
        "method": "nli",
        "prop_score": prop_score,
        "nli_knn_score": nli_knn_score,
        "pos_evidence": pos_evidence,
        "neg_evidence": neg_evidence,
        "evidence_margin": evidence_margin,
        "pos_in_k": pos_in_k,
        "neg_in_k": neg_in_k,
        "vote_k": vote_k,
        "retrieve_k": retrieve_k,
        "combined": final_score,
        "gap": gap,
        "pos_cos": pos_cos,
        "neg_cos": neg_cos,
        "abstain": abstain,
        "action": action,
        "voters": voter_details[:5],  # top 5 for display
    }

    # Build results -- sorted by NLI score (candidates first, then rest by cosine)
    anchor_order = []
    seen = set()
    for c in candidates:
        if c["is_positive"]:
            anchor_order.append(c["idx"])
            seen.add(c["idx"])
    for i in range(n_pos):
        if i not in seen:
            anchor_order.append(i)
    results = [(final_score, all_texts[i], all_categories[i], "") for i in anchor_order]

    return results, corrected, corrections, extraction, neg_cos, final_score, debug



def score_message_nli(message, all_texts, all_categories, model=None,
                      all_embeddings=None, neg_texts=None, neg_embeddings=None,
                      neutral_embeddings=None,
                      do_spellcheck=True, use_extraction=True):
    """
    NLI KNN voting: retrieve by cosine, re-rank by NLI, vote.

    Combines proposition NLI with anchor NLI KNN voting. Negative anchors
    compete directly with positive anchors, naturally handling refusals
    without gating or suppression.

    Returns (results, corrected, corrections, extraction_info, neg_cos, final_score, debug)
    """
    results, corrected, corrections, extraction, neg_cos, final_score, debug = _score_nli_core(
        message, all_texts, all_categories, model=model,
        all_embeddings=all_embeddings, neg_texts=neg_texts, neg_embeddings=neg_embeddings,
        neutral_embeddings=neutral_embeddings,
        do_spellcheck=do_spellcheck, use_extraction=use_extraction)
    return results, corrected, corrections, extraction, neg_cos, final_score, debug


# ---------------------------------------------------------------------------
# Hybrid scoring (weighted blend: NLI + Cosine KNN)
# ---------------------------------------------------------------------------

def score_message_hybrid(message, all_texts, all_categories, model=None,
                         all_embeddings=None, neg_texts=None, neg_embeddings=None,
                         neutral_embeddings=None,
                         do_spellcheck=True, use_extraction=True):
    """
    Hybrid scoring: adaptive weighted blend of NLI entailment and Cosine KNN voting.

    Base formula:
      hybrid_score = nli_weight * nli_score + (1 - nli_weight) * knn_score

    Adaptive weighting (when ADAPTIVE_HYBRID is True):
      When NLI evidence is weak (low margin, vote near 50/50), shift weight
      toward cosine KNN which handles broad propositions better. When NLI is
      confident (high margin, clear vote), keep the NLI-heavy weighting.

      Also applies a "max lift": if either mode is strongly confident, the
      hybrid score gets a floor boost to avoid one weak mode dragging down
      a strong signal from the other.

    Default: 75% NLI + 25% cosine KNN, adaptive=true (configurable in config).
    Combines NLI's semantic understanding with KNN's neighborhood-based robustness.

    Returns (results, corrected, corrections, extraction_info, neg_cos, debug_info)
    """
    import numpy as np

    base_nli_weight = HYBRID_NLI_WEIGHT
    base_cos_weight = 1.0 - base_nli_weight

    # --- NLI component ---
    nli_results, corrected, corrections, extraction, neg_cos, nli_score, nli_debug = \
        _score_nli_core(
            message, all_texts, all_categories, model=model,
            all_embeddings=all_embeddings, neg_texts=neg_texts,
            neg_embeddings=neg_embeddings, neutral_embeddings=neutral_embeddings,
            do_spellcheck=do_spellcheck, use_extraction=use_extraction)

    # --- Cosine KNN component ---
    knn_score = 0.0
    knn_info = {}
    if model is not None and all_embeddings is not None:
        _, _, knn_info = score_message(
            model, corrected, all_embeddings, all_texts, all_categories,
            neg_embeddings=neg_embeddings, neg_texts=neg_texts,
            neutral_embeddings=neutral_embeddings,
            use_extraction=use_extraction)
        knn_score = knn_info.get("knn_score", 0.0)

    # --- Adaptive weighting ---
    nli_weight = base_nli_weight
    cos_weight = base_cos_weight
    adaptation = "none"

    if ADAPTIVE_HYBRID:
        evidence_margin = nli_debug.get("evidence_margin", 0)
        nli_knn_score = nli_debug.get("nli_knn_score", 0)
        pos_in_k = nli_debug.get("pos_in_k", 0)
        vote_k = nli_debug.get("vote_k", 20)

        # NLI confidence: how decisive was the NLI KNN vote?
        # nli_knn near 0.5 = uncertain, near 0 or 1 = confident
        nli_confidence = abs(nli_knn_score - 0.5) * 2  # 0..1

        # When NLI is uncertain (confidence < 0.3), shift toward cosine
        # When NLI is confident (confidence > 0.7), keep NLI-heavy
        if nli_confidence < 0.3:
            # NLI is confused — give cosine equal or higher weight
            # Shift: nli 75%->40%, cos 25%->60%
            shift = (0.3 - nli_confidence) / 0.3  # 0..1
            nli_weight = base_nli_weight - shift * 0.35
            cos_weight = 1.0 - nli_weight
            adaptation = "cos_boost(conf={:.2f})".format(nli_confidence)

        elif nli_confidence > 0.7 and evidence_margin > NLI_ABSTAIN_MARGIN:
            # NLI is very confident — trust it more
            # Shift: nli 75%->85%, cos 25%->15%
            shift = (nli_confidence - 0.7) / 0.3  # 0..1
            nli_weight = min(0.90, base_nli_weight + shift * 0.15)
            cos_weight = 1.0 - nli_weight
            adaptation = "nli_boost(conf={:.2f})".format(nli_confidence)

    # --- Weighted blend ---
    hybrid_score = nli_weight * nli_score + cos_weight * knn_score

    # --- Max lift: don't let one weak mode drag down a strong signal ---
    # If cosine KNN is very confident (>= match_threshold) but hybrid is below,
    # apply a floor boost. Same for NLI.
    if ADAPTIVE_HYBRID:
        max_component = max(nli_score, knn_score)
        if max_component >= MATCH_THRESHOLD and hybrid_score < MATCH_THRESHOLD:
            # Lift toward the confident component (30% of the gap)
            lift = (max_component - hybrid_score) * 0.30
            hybrid_score += lift
            adaptation += "+lift({:.3f})".format(lift)
        elif max_component >= WARNING_THRESHOLD and hybrid_score < WARNING_THRESHOLD:
            lift = (max_component - hybrid_score) * 0.25
            hybrid_score += lift
            adaptation += "+lift({:.3f})".format(lift)

    # Build results list with blended score
    anchor_order = sorted(range(len(all_texts)),
                          key=lambda i: nli_results[i][0] if i < len(nli_results) else 0,
                          reverse=True)
    results = [(hybrid_score, all_texts[i], all_categories[i], "") for i in anchor_order]

    debug = {
        "method": "hybrid",
        "nli_score": nli_score,
        "knn_score": knn_score,
        "nli_weight": nli_weight,
        "cos_weight": cos_weight,
        "combined": hybrid_score,
        "adaptation": adaptation,
        "prop_score": nli_debug.get("prop_score", 0),
        "nli_knn_score": nli_debug.get("nli_knn_score", 0),
        "pos_evidence": nli_debug.get("pos_evidence", 0),
        "neg_evidence": nli_debug.get("neg_evidence", 0),
        "evidence_margin": nli_debug.get("evidence_margin", 0),
        "nli_pos_in_k": nli_debug.get("pos_in_k", 0),
        "nli_neg_in_k": nli_debug.get("neg_in_k", 0),
        "nli_vote_k": nli_debug.get("vote_k", 0),
        "nli_action": nli_debug.get("action", ""),
        "abstain": nli_debug.get("abstain", False),
        "gap": nli_debug.get("gap", 0),
        "cos_pos_in_k": knn_info.get("pos_in_k", 0),
        "cos_k": knn_info.get("k", 0),
    }
    return results, corrected, corrections, extraction, neg_cos, debug


# ---------------------------------------------------------------------------
# LLM Judge scoring
# ---------------------------------------------------------------------------

_llm_config = None


def _ollama_ensure_model(model, base_url=None):
    """
    Check if an Ollama model is available locally; pull it if not.

    Calls GET /api/tags to list local models. If the requested model isn't
    found, triggers POST /api/pull to download it (with progress output).
    """
    import httpx
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    base = base_url.rstrip("/")

    # Check if Ollama is running
    try:
        resp = httpx.get(base + "/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        print("\n  {}ERROR:{} Cannot connect to Ollama at {}".format(RED, RESET, base))
        print("  Install: brew install ollama")
        print("  Start:   ollama serve")
        return False

    # Check if model is already available
    data = resp.json()
    local_models = [m.get("name", "") for m in data.get("models", [])]
    model_base = model.split(":")[0]
    found = any(model_base == m.split(":")[0] for m in local_models)

    if found:
        return True

    # Model not found — pull it
    print("\n  {}Ollama:{} Model '{}' not found locally. Downloading...".format(
        BOLD, RESET, model))
    print("  This is a one-time download. It may take several minutes.\n")

    try:
        with httpx.stream("POST", base + "/api/pull",
                          json={"name": model}, timeout=None) as stream:
            last_status = ""
            for line in stream.iter_lines():
                if not line:
                    continue
                try:
                    import json as _json
                    chunk = _json.loads(line)
                    status = chunk.get("status", "")
                    total = chunk.get("total", 0)
                    completed = chunk.get("completed", 0)
                    if "pulling" in status.lower() or "download" in status.lower():
                        if total > 0:
                            pct = completed / total * 100
                            gb_done = completed / (1024 ** 3)
                            gb_total = total / (1024 ** 3)
                            print("  {} {:.1f}% ({:.1f}/{:.1f}GB)    ".format(
                                status[:20], pct, gb_done, gb_total),
                                end="\r", flush=True)
                        else:
                            print("  {}".format(status), end="\r", flush=True)
                    elif status == "success":
                        print("\n  {}\u2713 Model '{}' downloaded successfully{}".format(
                            GREEN, model, RESET))
                    elif status != last_status:
                        print("  {}".format(status))
                    last_status = status
                except (ValueError, KeyError):
                    pass
        return True
    except Exception as e:
        print("\n  {}ERROR:{} Failed to pull model '{}': {}".format(RED, RESET, model, e))
        return False


def _load_llm_config(silent=False):
    """Load LLM judge config from config_<n>.ini."""
    global _llm_config
    if _llm_config is not None:
        return _llm_config

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_{}.ini".format(SCRIPT_NAME))

    if not os.path.exists(config_path):
        if not silent:
            print("\n  {}ERROR: LLM judge config not found: {}{}".format(
                "\033[91m", config_path, RESET))
            print("  Create it with:")
            print("    [llm_judge]")
            print("    provider = anthropic")
            print("    model = claude-sonnet-4-20250514")
            print("    api_key = sk-ant-...")
            print("    proposition = Your proposition here")
            print("\n  Or regenerate with: python semantic_anchor_generator.py -name {} -gs".format(
                SCRIPT_NAME))
        sys.exit(1)

    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)

    _llm_config = {
        "provider": config.get("llm_judge", "provider"),
        "model": config.get("llm_judge", "model"),
        "api_key": config.get("llm_judge", "api_key"),
        "proposition": config.get("llm_judge", "proposition", fallback=PROPOSITION),
        "base_url": config.get("llm_judge", "base_url",
                               fallback="http://localhost:11434"),
    }
    return _llm_config


def _call_llm_judge(message, proposition, top_anchors=None):
    """
    Call an LLM to judge whether a message matches the proposition.

    Returns dict: {match: bool, reasoning: str, verdict: str}
    """
    config = _load_llm_config()
    provider = config["provider"].lower()
    model = config["model"]
    api_key = config["api_key"]

    # Build the prompt
    anchors_context = ""
    if top_anchors:
        anchor_lines = []
        for score, text, cat, *rest in top_anchors[:5]:
            anchor_lines.append("  - [{:.3f}] ({}) \"{}\"".format(score, cat, text))
        anchors_context = (
            "\n\nFor reference, these are the closest semantic anchor matches "
            "(pre-scored by embedding similarity):\n" + "\n".join(anchor_lines)
        )

    system_prompt = (
        "You are a precise content classifier. Your task is to determine whether "
        "a user message matches a given proposition. You must distinguish between:\n"
        "- Messages that genuinely request/intend what the proposition describes\n"
        "- Messages that use similar vocabulary but have legitimate/benign intent\n"
        "- Educational questions ABOUT the topic (not requesting to perform it)\n\n"
        "Respond with ONLY valid JSON, no other text."
    )

    user_prompt = (
        'PROPOSITION: "{}"\n\n'
        'USER MESSAGE: "{}"\n'
        '{}\n'
        'Does this message match the proposition? Consider:\n'
        '1. Does the user intend to perform the action described in the proposition?\n'
        '2. Or are they asking for education, defense, their own account, or general info?\n'
        '3. First-person ownership ("my account", "my password") vs third-party targeting\n\n'
        'Respond with JSON:\n'
        '{{\n'
        '  "match": true or false,\n'
        '  "reasoning": "brief explanation"\n'
        '}}'
    ).format(proposition, message, anchors_context)

    try:
        import httpx

        if provider == "anthropic":
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "system": system_prompt,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]

        elif provider == "openai":
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

        elif provider == "gemini" or provider == "google":
            # Google Gemini API
            url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
                model, api_key)
            resp = httpx.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": system_prompt + "\n\n" + user_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 300,
                        "temperature": 0,
                    },
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]

        elif provider == "grok" or provider == "xai":
            # xAI Grok API (OpenAI-compatible)
            resp = httpx.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

        elif provider in ("ollama", "local", "lmstudio", "vllm"):
            # Local model via Ollama or any OpenAI-compatible local server
            base_url = config.get("base_url", "http://localhost:11434")
            if provider == "ollama":
                # Auto-pull model if needed
                _ollama_ensure_model(model, base_url)
                # Ollama native API
                resp = httpx.post(
                    base_url.rstrip("/") + "/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["message"]["content"]
            else:
                # LM Studio / vLLM / any OpenAI-compatible local server
                resp = httpx.post(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": model,
                        "max_tokens": 300,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]

        else:
            return {"match": False,
                    "reasoning": "Unsupported provider: " + provider,
                    "verdict": "ERROR"}

        # Parse JSON response
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(text)

        is_match = result.get("match", False)
        reasoning = result.get("reasoning", "")

        verdict = "MATCH" if is_match else "NO MATCH"

        return {
            "match": is_match,
            "reasoning": reasoning,
            "verdict": verdict,
        }

    except Exception as e:
        return {
            "match": False,
            "reasoning": "LLM call failed: {}".format(str(e)),
            "verdict": "ERROR",
        }


def score_message_llm(message, proposition, model=None, all_embeddings=None,
                      all_texts=None, all_categories=None,
                      neg_embeddings=None, neutral_embeddings=None,
                      do_spellcheck=True, use_extraction=True):
    """
    LLM-as-judge scoring. Optionally uses cosine pre-ranking to provide
    context anchors to the LLM.

    Returns (llm_result, corrected, corrections, cosine_results)
    where llm_result is the dict from _call_llm_judge.
    """
    corrected = message
    corrections = []
    if do_spellcheck:
        corrected, corrections = correct_spelling(message)

    # Optional: cosine pre-rank for context
    cosine_results = None
    if model is not None and all_embeddings is not None:
        cosine_results, _, _ = score_message(
            model, corrected, all_embeddings, all_texts, all_categories,
            neg_embeddings=neg_embeddings, neutral_embeddings=neutral_embeddings,
            use_extraction=use_extraction)

    llm_result = _call_llm_judge(corrected, proposition, top_anchors=cosine_results)

    return llm_result, corrected, corrections, cosine_results


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
    print("  Role: {}".format(ANCHOR_ROLE))
    print("  NLI prefix: \"{}\"".format(NLI_DECLARATIVE_PREFIX))
    print("  Thresholds: match={}, warning={}, KNN K={}".format(
        MATCH_THRESHOLD, WARNING_THRESHOLD, COSINE_KNN_K))
    print("  Hybrid blend: {:.0f}% NLI + {:.0f}% cosine KNN".format(
        HYBRID_NLI_WEIGHT * 100, (1 - HYBRID_NLI_WEIGHT) * 100))
    print("  NLI KNN: retrieve={}, vote={}, fwd_weight={}, abstain_margin={}".format(
        NLI_RETRIEVE_K, NLI_VOTE_K, NLI_FWD_WEIGHT, NLI_ABSTAIN_MARGIN))
    print("  Adaptive hybrid: {}".format("ON" if ADAPTIVE_HYBRID else "OFF"))
    print("")


def print_examples():
    total = sum(len(v) for v in ANCHORS.values())
    print("\n{}Positive Anchors ({} total){}".format(BOLD, total, RESET))
    print("=" * 70)
    for cat, examples in ANCHORS.items():
        print("\n{}{}  [{}]{} ({} examples)".format(BOLD, BLUE, cat, RESET, len(examples)))
        print("  " + "-" * 60)
        for i, ex in enumerate(examples, 1):
            print("  {}{:>3}.{} {}".format(DIM, i, RESET, ex))

    if NEGATIVE_ANCHORS:
        neg_total = sum(len(v) for v in NEGATIVE_ANCHORS.values())
        print("\n\n{}Negative (Contrastive) Anchors ({} total){}".format(BOLD, neg_total, RESET))
        print("=" * 70)
        for cat, examples in NEGATIVE_ANCHORS.items():
            print("\n{}{}  [{}]{} ({} examples)".format(BOLD, GREEN, cat, RESET, len(examples)))
            print("  " + "-" * 60)
            for i, ex in enumerate(examples, 1):
                print("  {}{:>3}.{} {}".format(DIM, i, RESET, ex))

    if NEUTRAL_ANCHORS:
        print("\n\n{}Neutral (Off-topic Baseline) Anchors ({} total){}".format(
            BOLD, len(NEUTRAL_ANCHORS), RESET))
        print("=" * 70)
        for i, ex in enumerate(NEUTRAL_ANCHORS, 1):
            print("  {}{:>3}.{} {}".format(DIM, i, RESET, ex))
    print()


def format_verdict(top_score, neg_cos=0.0):
    neg_info = ""
    if neg_cos > 0.01:
        neg_info = "  {}[neg_cos={:.3f}]{}".format(DIM, neg_cos, RESET)
    if top_score >= MATCH_THRESHOLD:
        return "{}{}\u25a0 MATCH{} (score {:.4f} \u2265 {}){}".format(
            RED, BOLD, RESET, top_score, MATCH_THRESHOLD, neg_info)
    elif top_score >= WARNING_THRESHOLD:
        return "{}{}\u25a0 WARNING{} (score {:.4f} \u2265 {}){}".format(
            YELLOW, BOLD, RESET, top_score, WARNING_THRESHOLD, neg_info)
    else:
        return "{}{}\u25a0 NO MATCH{} (score {:.4f} < {}){}".format(
            GREEN, BOLD, RESET, top_score, WARNING_THRESHOLD, neg_info)


def format_verdict_cosine(knn_score, pos_in_k, k, max_pos, max_neg):
    """Format verdict for cosine KNN voting."""
    verdict = _verdict_for_cosine_gap(knn_score)
    detail = "knn={:.0f}/{} ({:.0%})  max_pos={:.3f}  max_neg={:.3f}".format(
        pos_in_k, k, knn_score, max_pos, max_neg)
    if verdict == "MATCH":
        return "{}{}\u25a0 MATCH{} ({})".format(RED, BOLD, RESET, detail)
    elif verdict == "WARNING":
        return "{}{}\u25a0 WARNING{} ({})".format(YELLOW, BOLD, RESET, detail)
    else:
        return "{}{}\u25a0 NO MATCH{} ({})".format(GREEN, BOLD, RESET, detail)


# --- Cosine display ---

def display_default(results, extraction=None, top_n=3, verbose_extract=False,
                    neg_score=0.0, knn_info=None):
    if extraction:
        _display_extraction_info(extraction, verbose=verbose_extract)
    if knn_info is not None:
        print("\n  {}\n".format(format_verdict_cosine(
            knn_info["knn_score"], knn_info["pos_in_k"], knn_info["k"],
            knn_info["max_pos"], knn_info["max_neg"])))
    else:
        print("\n  {}\n".format(format_verdict(results[0][0], neg_score)))
    print("  {:<6} {:>8}  {:<35} {}".format("Rank", "Score", "Category", "Nearest Anchor"))
    print("  " + "-" * 95)
    for rank, (score, text, cat) in enumerate(results[:top_n], 1):
        cs = color_score(score)
        dt = text if len(text) <= 55 else text[:52] + "..."
        dc = cat if len(cat) <= 33 else cat[:30] + "..."
        print("  {:<6} {:>17}  {}{:<35}{} \"{}\"".format(rank, cs, DIM, dc, RESET, dt))
    print()


def display_verbose(results, extraction=None, neg_score=0.0, knn_info=None):
    if extraction:
        _display_extraction_info(extraction, verbose=True)
    if knn_info is not None:
        print("\n  {}\n".format(format_verdict_cosine(
            knn_info["knn_score"], knn_info["pos_in_k"], knn_info["k"],
            knn_info["max_pos"], knn_info["max_neg"])))
    else:
        print("\n  {}\n".format(format_verdict(results[0][0], neg_score)))
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


# --- Reranked display ---


# --- NLI display ---

def display_default_nli(results, corrected, corrections, extraction=None,
                        top_n=3, neg_score=0.0, debug=None):
    top_score = results[0][0]
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))
    if extraction:
        _display_extraction_info(extraction)
    if debug:
        pos_k = debug.get("pos_in_k", 0)
        neg_k = debug.get("neg_in_k", 0)
        vk = debug.get("vote_k", 20)
        prop = debug.get("prop_score", 0)
        act = debug.get("action", "")
        print("  {}NLI KNN: {}/{}+ {}/{}− prop={:.3f} [{}]{}".format(
            DIM, pos_k, vk, neg_k, vk, prop, act, RESET))
    print("\n  {}\n".format(format_verdict(top_score, neg_score)))
    print("  {:<6} {:>8}  {:<30} {}".format(
        "Rank", "NLI", "Category", "Nearest Anchor"))
    print("  " + "-" * 100)
    for rank, (score, text, cat, best_src) in enumerate(results[:top_n], 1):
        cs = color_score(score)
        dt = text if len(text) <= 50 else text[:47] + "..."
        dc = cat if len(cat) <= 28 else cat[:25] + "..."
        print("  {:<6} {:>17}  {}{:<30}{} \"{}\"".format(
            rank, cs, DIM, dc, RESET, dt))
    print()


def display_verbose_nli(results, corrected, corrections, extraction=None,
                        neg_score=0.0, debug=None):
    top_score = results[0][0]
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))
    if extraction:
        _display_extraction_info(extraction, verbose=True)
    if debug:
        prop = debug.get("prop_score", 0)
        knn = debug.get("nli_knn_score", 0)
        pos_k = debug.get("pos_in_k", 0)
        neg_k = debug.get("neg_in_k", 0)
        pos_ev = debug.get("pos_evidence", 0)
        neg_ev = debug.get("neg_evidence", 0)
        margin = debug.get("evidence_margin", 0)
        act = debug.get("action", "")
        print("  {}NLI KNN debug:{}".format(BOLD, RESET))
        print("    prop_score={:.3f}  knn_score={:.3f}".format(prop, knn))
        print("    voters: {}+/{}− evidence={:.2f}/{:.2f} margin={:.2f}".format(
            pos_k, neg_k, pos_ev, neg_ev, margin))
        print("    {}[{}]{}".format(DIM, act, RESET))
    print("\n  {}\n".format(format_verdict(top_score, neg_score)))
    print("  {:<5} {:>8}  {:<30} {}".format("#", "NLI", "Category", "Anchor Text"))
    print("  " + "=" * 105)
    for rank, (score, text, cat, best_src) in enumerate(results, 1):
        cs = color_score(score)
        dc = cat if len(cat) <= 28 else cat[:25] + "..."
        zone = ""
        if score >= MATCH_THRESHOLD:
            zone = " " + RED + "\u25c4 MATCH" + RESET
        elif score >= WARNING_THRESHOLD:
            zone = " " + YELLOW + "\u25c4 WARN" + RESET
        print("  {:<5} {:>17}  {}{:<30}{} \"{}\"{}".format(
            rank, cs, DIM, dc, RESET, text, zone))
    print()


# --- Hybrid display ---

def display_default_hybrid(results, corrected, corrections, extraction=None,
                           top_n=3, neg_score=0.0, debug=None):
    top_score = results[0][0]
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))
    if extraction:
        _display_extraction_info(extraction)

    # Show hybrid debug
    if debug:
        nli_sc = debug.get("nli_score", 0)
        knn_sc = debug.get("knn_score", 0)
        nli_w = debug.get("nli_weight", 0.75)
        cos_w = debug.get("cos_weight", 0.25)
        combined = debug.get("combined", 0)
        cos_pos = debug.get("cos_pos_in_k", 0)
        cos_k = debug.get("cos_k", 0)
        nli_pos = debug.get("nli_pos_in_k", 0)
        nli_neg = debug.get("nli_neg_in_k", 0)
        nli_action = debug.get("nli_action", "")
        print("  {}Hybrid: nli={:.3f} ({}+/{}−) x {:.0f}% + cos_knn={:.3f} ({}/{}) x {:.0f}% = {:.3f} [{}]{}".format(
            DIM, nli_sc, nli_pos, nli_neg, nli_w * 100,
            knn_sc, cos_pos, cos_k, cos_w * 100, combined,
            nli_action, RESET))
        adapt = debug.get("adaptation", "none")
        if adapt and adapt != "none":
            print("  {}Adaptive: {}{}".format(DIM, adapt, RESET))

    print("\n  {}\n".format(format_verdict(top_score, neg_score)))
    print("  {:<6} {:>8}  {:<30} {}".format(
        "Rank", "NLI", "Category", "Nearest Anchor"))
    print("  " + "-" * 100)
    for rank, (score, text, cat, best_src) in enumerate(results[:top_n], 1):
        cs = color_score(score)
        dt = text if len(text) <= 50 else text[:47] + "..."
        dc = cat if len(cat) <= 28 else cat[:25] + "..."
        print("  {:<6} {:>17}  {}{:<30}{} \"{}\"".format(
            rank, cs, DIM, dc, RESET, dt))
        if best_src and best_src not in (corrected, "[full message]"):
            bs = best_src if len(best_src) <= 65 else best_src[:62] + "..."
            print("  {}       matched via: {}{} ".format(DIM, bs, RESET))
    print()


def display_verbose_hybrid(results, corrected, corrections, extraction=None,
                           neg_score=0.0, debug=None):
    top_score = results[0][0]
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))
    if extraction:
        _display_extraction_info(extraction, verbose=True)

    if debug:
        nli_sc = debug.get("nli_score", 0)
        knn_sc = debug.get("knn_score", 0)
        nli_w = debug.get("nli_weight", 0.75)
        cos_w = debug.get("cos_weight", 0.25)
        combined = debug.get("combined", 0)
        cos_pos = debug.get("cos_pos_in_k", 0)
        cos_k = debug.get("cos_k", 0)
        prop = debug.get("prop_score", 0)
        nli_pos = debug.get("nli_pos_in_k", 0)
        nli_neg = debug.get("nli_neg_in_k", 0)
        pos_ev = debug.get("pos_evidence", 0)
        neg_ev = debug.get("neg_evidence", 0)
        margin = debug.get("evidence_margin", 0)
        nli_action = debug.get("nli_action", "")
        print("  {}Hybrid debug: nli={:.3f} (prop={:.3f} knn={}+/{}− ev={:.2f}/{:.2f} margin={:.2f}) x {:.0f}%".format(
            DIM, nli_sc, prop, nli_pos, nli_neg, pos_ev, neg_ev, margin, nli_w * 100))
        print("               + cos_knn={:.3f} ({}/{}) x {:.0f}% = {:.3f} [{}]{}".format(
            knn_sc, cos_pos, cos_k, cos_w * 100, combined, nli_action, RESET))
        adapt = debug.get("adaptation", "none")
        if adapt and adapt != "none":
            print("  {}Adaptive: {}{}".format(DIM, adapt, RESET))

    print("\n  {}\n".format(format_verdict(top_score, neg_score)))
    print("  {:<5} {:>8}  {:<30} {}".format("#", "NLI", "Category", "Anchor Text"))
    print("  " + "=" * 105)
    for rank, (score, text, cat, best_src) in enumerate(results, 1):
        cs = color_score(score)
        dc = cat if len(cat) <= 28 else cat[:25] + "..."
        zone = ""
        if score >= MATCH_THRESHOLD:
            zone = " " + RED + "\u25c4 MATCH" + RESET
        elif score >= WARNING_THRESHOLD:
            zone = " " + YELLOW + "\u25c4 WARN" + RESET
        print("  {:<5} {:>17}  {}{:<30}{} \"{}\"{}".format(
            rank, cs, DIM, dc, RESET, text, zone))
    above_m = sum(1 for s, _, _, _ in results if s >= MATCH_THRESHOLD)
    above_w = sum(1 for s, _, _, _ in results if s >= WARNING_THRESHOLD)
    print("\n  {}Total: {} | Above match ({}): {} | Above warning ({}): {} | "
          "Hybrid {:.0f}% NLI + {:.0f}% KNN{}".format(
        DIM, len(results), MATCH_THRESHOLD, above_m, WARNING_THRESHOLD, above_w,
        HYBRID_NLI_WEIGHT * 100, (1 - HYBRID_NLI_WEIGHT) * 100, RESET))
    print()


# --- LLM Judge display ---

def display_llm_result(llm_result, corrected, corrections, cosine_results=None, top_n=3):
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))

    verdict = llm_result.get("verdict", "ERROR")
    reasoning = llm_result.get("reasoning", "")

    if verdict == "MATCH":
        vcolor = RED
    elif verdict == "WARNING":
        vcolor = YELLOW
    elif verdict == "ERROR":
        vcolor = "\033[91m"
    else:
        vcolor = GREEN

    print("\n  {}{}\u25a0 {}{}".format(
        vcolor, BOLD, verdict, RESET))
    if reasoning:
        # Word-wrap reasoning at ~80 chars
        words = reasoning.split()
        lines = []
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > 82:
                lines.append(line)
                line = "  "
            line += w + " "
        if line.strip():
            lines.append(line)
        print("  {}Reasoning:{} {}".format(DIM, RESET, lines[0].strip()))
        for l in lines[1:]:
            print("  {}".format(l.rstrip()))

    if cosine_results:
        print("\n  {}Cosine context (for reference):{}".format(DIM, RESET))
        print("  {:<6} {:>8}  {:<30} {}".format("Rank", "Cosine", "Category", "Nearest Anchor"))
        print("  " + "-" * 90)
        for rank, (score, text, cat, *rest) in enumerate(cosine_results[:top_n], 1):
            cs = color_score(score)
            dt = text if len(text) <= 50 else text[:47] + "..."
            dc = cat if len(cat) <= 28 else cat[:25] + "..."
            print("  {:<6} {:>17}  {}{:<30}{} \"{}\"".format(
                rank, cs, DIM, dc, RESET, dt))
    print()


# ---------------------------------------------------------------------------
# Compare mode — run all scoring methods side-by-side
# ---------------------------------------------------------------------------

def _verdict_for_score(score):
    """Return verdict string for a numeric score (NLI/hybrid modes)."""
    if score >= MATCH_THRESHOLD:
        return "MATCH"
    elif score >= WARNING_THRESHOLD:
        return "WARNING"
    return "NO MATCH"


def _verdict_for_cosine_gap(knn_score):
    """Return verdict string for cosine KNN positive ratio."""
    if knn_score >= MATCH_THRESHOLD:
        return "MATCH"
    elif knn_score >= WARNING_THRESHOLD:
        return "WARNING"
    return "NO MATCH"


def _parse_labeled_file(content):
    """
    Parse input file with optional ground truth labels.

    Supports two formats:
      Format 1 (no labels):   message1 ### message2 ### message3
      Format 2 (with labels): message1 ### MATCH ### message2 ### NO MATCH ### ...

    Labels can appear on the same line as ### or on a new line after ###:
      message1
      ### MATCH
      message2
      ### NO MATCH

    Labels are case-insensitive: MATCH, match, NO MATCH, no match, CLEAN, clean.
    WARNING is treated as MATCH for ground truth purposes.

    Returns (messages, labels) where labels is a list of "MATCH"/"NO MATCH"/None.
    If no labels found at all, labels will be all None.
    """
    LABEL_WORDS = {"MATCH", "NO MATCH", "CLEAN", "WARNING"}

    raw_parts = [p.strip() for p in content.split("###")]
    raw_parts = [p for p in raw_parts if p]

    # Expand parts: if a part starts with a label on its first line,
    # split into [label, rest_of_part]
    parts = []
    for rp in raw_parts:
        lines = rp.split("\n", 1)
        first_line = lines[0].strip().upper()
        if first_line in LABEL_WORDS:
            parts.append(first_line)  # the label
            if len(lines) > 1 and lines[1].strip():
                parts.append(lines[1].strip())  # the next message
        else:
            parts.append(rp)

    messages = []
    label_dict = {}  # msg_index -> label

    for part in parts:
        part_upper = part.upper()
        if part_upper in LABEL_WORDS:
            # Label applies to the most recently added message
            if messages:
                msg_idx = len(messages) - 1
                if msg_idx not in label_dict:
                    if part_upper in ("MATCH", "WARNING"):
                        label_dict[msg_idx] = "MATCH"
                    else:
                        label_dict[msg_idx] = "NO MATCH"
        else:
            messages.append(part)

    labels = [label_dict.get(i) for i in range(len(messages))]

    # Check if ANY labels were provided
    has_labels = any(l is not None for l in labels)
    if not has_labels:
        labels = [None] * len(messages)

    return messages, labels


def _verdict_color(verdict):
    if verdict == "MATCH":
        return RED
    elif verdict == "WARNING":
        return YELLOW
    elif verdict == "ERROR":
        return "\033[91m"
    return GREEN


def score_all_modes(message, model, embs, texts, cats,
                    neg_embs=None, neg_texts=None, neutral_embs=None,
                    use_extraction=True, include_llm=False):
    """
    Run all scoring modes on a single message.

    Returns dict of mode_name -> {
        verdict: str, score: float, top3: list,
        reasoning: str (llm only), neg_cos: float
    }
    """
    results = {}

    # --- Cosine ---
    cos_results, cos_extraction, cos_knn = score_message(
        model, message, embs, texts, cats,
        neg_embeddings=neg_embs, neg_texts=neg_texts,
        neutral_embeddings=neutral_embs,
        use_extraction=use_extraction)
    knn_score = cos_knn["knn_score"]
    results["cosine"] = {
        "verdict": _verdict_for_cosine_gap(knn_score),
        "score": knn_score,
        "top3": cos_results[:3],
        "neg_cos": cos_knn["max_neg"],
        "knn": cos_knn,
    }

    # --- NLI ---
    nli_results, nli_corr, nli_corrections, nli_ext, nli_neg, nli_prop, nli_debug = score_message_nli(
        message, texts, cats, model=model, all_embeddings=embs,
        neg_texts=neg_texts, neg_embeddings=neg_embs,
        neutral_embeddings=neutral_embs,
        use_extraction=use_extraction)
    top_nli = nli_results[0][0]
    results["nli"] = {
        "verdict": _verdict_for_score(top_nli),
        "score": top_nli,
        "top3": nli_results[:3],
        "neg_cos": nli_neg,
        "corrections": nli_corrections,
        "prop_score": nli_prop,
        "debug": nli_debug,
    }

    # --- Hybrid ---
    hyb_results, hyb_corr, hyb_corrections, hyb_ext, hyb_neg, hyb_debug = score_message_hybrid(
        message, texts, cats, model=model, all_embeddings=embs,
        neg_texts=neg_texts, neg_embeddings=neg_embs,
        neutral_embeddings=neutral_embs,
        use_extraction=use_extraction)
    top_hyb = hyb_results[0][0]
    results["hybrid"] = {
        "verdict": _verdict_for_score(top_hyb),
        "score": top_hyb,
        "top3": hyb_results[:3],
        "neg_cos": hyb_neg,
        "debug": hyb_debug,
    }

    # --- LLM (optional) ---
    if include_llm:
        llm_result, llm_corr, llm_corrections, llm_cos = score_message_llm(
            message, PROPOSITION, model=model, all_embeddings=embs,
            all_texts=texts, all_categories=cats,
            neg_embeddings=neg_embs, neutral_embeddings=neutral_embs,
            use_extraction=use_extraction)
        results["llm"] = {
            "verdict": llm_result.get("verdict", "ERROR"),
            "score": 0.0,  # LLM returns decision, not score
            "top3": [],
            "reasoning": llm_result.get("reasoning", ""),
            "neg_cos": 0.0,
        }

    return results


def display_compare(message, mode_results, index=None, total=None):
    """Display side-by-side comparison table for one message."""
    # Header
    if index is not None and total is not None:
        disp = message if len(message) <= 80 else message[:77] + "..."
        print("\n  {}[{}/{}]{} \"{}\"".format(BOLD, index, total, RESET, disp))
    else:
        disp = message if len(message) <= 90 else message[:87] + "..."
        print("\n  \"{}\"".format(disp))

    # Spell corrections (from NLI — same for all)
    corrections = mode_results.get("nli", {}).get("corrections", [])
    if corrections:
        print("  {}Spell corrected:{} {}".format(CYAN, RESET,
            ", ".join("{} \u2192 {}".format(o, c) for o, c in corrections)))

    # === Verdict summary row ===
    print()
    mode_order = ["cosine", "nli", "hybrid", "llm"]
    active_modes = [m for m in mode_order if m in mode_results]

    # Build column widths
    col_w = 26
    header = "  {:<12}".format("")
    for m in active_modes:
        header += " {:^{}}".format(m.upper(), col_w)
    print(BOLD + header + RESET)
    print("  " + "\u2500" * (12 + (col_w + 1) * len(active_modes)))

    # Verdict row
    row_verdict = "  {:<12}".format("Verdict")
    for m in active_modes:
        v = mode_results[m]["verdict"]
        vc = _verdict_color(v)
        cell = "{}{}\u25a0 {}{}".format(vc, BOLD, v, RESET)
        # Pad for alignment (color codes don't take visual space)
        pad = col_w - len("\u25a0 " + v)
        row_verdict += " " + cell + " " * max(0, pad)
    print(row_verdict)

    # Score row
    row_score = "  {:<12}".format("Score")
    for m in active_modes:
        if m == "llm":
            cell = "--"
        elif m == "cosine":
            knn = mode_results[m].get("knn", {})
            pos_k = knn.get("pos_in_k", 0)
            k = knn.get("k", 0)
            ks = knn.get("knn_score", 0)
            cell = "{:.0f}/{} ({:.0%})".format(pos_k, k, ks)
        elif m == "nli":
            nli_debug = mode_results[m].get("debug")
            if nli_debug:
                nli_pos = nli_debug.get("pos_in_k", 0)
                nli_neg = nli_debug.get("neg_in_k", 0)
                nli_vk = nli_debug.get("vote_k", 20)
                s = mode_results[m]["score"]
                cell = "{}+/{}− ({:.0%})".format(nli_pos, nli_neg, s)
            else:
                s = mode_results[m]["score"]
                cell = "{:.4f}".format(s)
        else:
            s = mode_results[m]["score"]
            cell = "{:.4f}".format(s)
        row_score += " {:^{}}".format(cell, col_w)
    print(row_score)

    # Max Pos / Neg row
    row_neg = "  {:<12}".format("Pos / Neg")
    for m in active_modes:
        if m == "llm":
            cell = "--"
        elif m == "cosine":
            knn = mode_results[m].get("knn", {})
            mp = knn.get("max_pos", 0.0)
            mn = knn.get("max_neg", 0.0)
            cell = "{:.3f} / {:.3f}".format(mp, mn)
        else:
            nc = mode_results[m].get("neg_cos", 0.0)
            cell = "-- / {:.3f}".format(nc) if nc > 0.01 else "--"
        row_neg += " {:^{}}".format(cell, col_w)
    print(row_neg)

    print("  " + "\u2500" * (12 + (col_w + 1) * len(active_modes)))

    # === Top 3 anchors per mode ===
    print("  {}Top 3 Anchors:{}".format(BOLD, RESET))

    for rank in range(3):
        row = "  {:<12}".format("  #{}".format(rank + 1))
        for m in active_modes:
            mr = mode_results[m]
            is_llm = (m == "llm")
            if is_llm:
                if rank == 0:
                    reasoning = mr.get("reasoning", "")
                    cell = reasoning if len(reasoning) <= (col_w - 2) else reasoning[:col_w - 5] + "..."
                else:
                    cell = ""
            else:
                top3 = mr.get("top3", [])
                if rank < len(top3):
                    s, t, c, *rest = top3[rank]
                    max_t = col_w - 9  # space for score "[0.XXX] "
                    dt = t if len(t) <= max_t else t[:max_t - 3] + "..."
                    cell = "{:.3f} \"{}\"".format(s, dt)
                    if len(cell) > col_w:
                        cell = cell[:col_w - 3] + "..."
                else:
                    cell = ""
            row += " {:<{}}".format(cell, col_w)
        print(row)

    # Category for top anchor per mode
    row_cat = "  {:<12}".format("  Category")
    for m in active_modes:
        if m == "llm":
            row_cat += " {:<{}}".format("", col_w)
        else:
            top3 = mode_results[m].get("top3", [])
            if top3:
                cat = top3[0][2]
                dc = cat if len(cat) <= (col_w - 2) else cat[:col_w - 5] + "..."
                row_cat += " {}{:<{}}{}".format(DIM, dc, col_w, RESET)
            else:
                row_cat += " {:<{}}".format("", col_w)
    print(row_cat)

    # Hybrid debug line
    hyb = mode_results.get("hybrid", {})
    debug = hyb.get("debug")
    if debug:
        nli_sc = debug.get("nli_score", 0)
        knn_sc = debug.get("knn_score", 0)
        nli_w = debug.get("nli_weight", 0.75)
        cos_w = debug.get("cos_weight", 0.25)
        combined = debug.get("combined", 0)
        cos_pos = debug.get("cos_pos_in_k", 0)
        cos_k = debug.get("cos_k", 0)
        nli_pos = debug.get("nli_pos_in_k", 0)
        nli_neg = debug.get("nli_neg_in_k", 0)
        nli_action = debug.get("nli_action", "")
        print("  {}Hybrid: nli={:.3f} ({}+/{}−) x {:.0f}% + cos_knn={:.3f} ({}/{}) x {:.0f}% = {:.3f} [{}]{}".format(
            DIM, nli_sc, nli_pos, nli_neg, nli_w * 100,
            knn_sc, cos_pos, cos_k, cos_w * 100, combined,
            nli_action, RESET))
        adapt = debug.get("adaptation", "none")
        if adapt and adapt != "none":
            print("  {}Adaptive: {}{}".format(DIM, adapt, RESET))

    print()


def display_compare_summary(all_mode_results, messages, labels=None):
    """Display final summary grid across all messages and modes."""
    modes = ["cosine", "nli", "hybrid"]
    if any("llm" in mr for mr in all_mode_results):
        modes.append("llm")

    has_labels = labels is not None and any(l is not None for l in labels)

    # Count per mode
    counts = {}
    correct = {}
    for m in modes:
        counts[m] = {"MATCH": 0, "WARNING": 0, "NO MATCH": 0, "ERROR": 0}
        correct[m] = 0

    for idx, mr in enumerate(all_mode_results):
        for m in modes:
            if m in mr:
                v = mr[m]["verdict"]
                counts[m][v] = counts[m].get(v, 0) + 1
                # Compute accuracy against ground truth
                if has_labels and labels[idx] is not None:
                    predicted_match = v in ("MATCH", "WARNING")
                    expected_match = labels[idx] == "MATCH"
                    if predicted_match == expected_match:
                        correct[m] += 1

    total = len(messages)
    labeled_total = sum(1 for l in (labels or []) if l is not None) if has_labels else 0

    print("\n  {}Comparison Summary ({} messages{}){}".format(
        BOLD, total,
        ", {} labeled".format(labeled_total) if has_labels else "",
        RESET))
    print("  " + "=" * 75)

    if has_labels:
        print("  {}{:<10} {:>10} {:>10} {:>10} {:>10} {:>12}{}".format(
            BOLD, "Mode", "Matches", "Warnings", "No Match", "Errors", "Accuracy", RESET))
    else:
        print("  {}{:<10} {:>10} {:>10} {:>10} {:>10}{}".format(
            BOLD, "Mode", "Matches", "Warnings", "No Match", "Errors", RESET))
    print("  " + "-" * 75)

    for m in modes:
        c = counts[m]
        matches = c["MATCH"]
        warns = c["WARNING"]
        clean = c["NO MATCH"]
        errs = c["ERROR"]

        if has_labels and labeled_total > 0:
            acc_pct = correct[m] / labeled_total * 100
            print("  {:<10} {}{:>10}{} {}{:>10}{} {}{:>10}{} {:>10} {:>11.1f}%".format(
                m.upper(),
                RED, matches, RESET,
                YELLOW, warns, RESET,
                GREEN, clean, RESET,
                errs,
                acc_pct))
        else:
            print("  {:<10} {}{:>10}{} {}{:>10}{} {}{:>10}{} {:>10}".format(
                m.upper(),
                RED, matches, RESET,
                YELLOW, warns, RESET,
                GREEN, clean, RESET,
                errs))

    if has_labels:
        print()
        print("  {}Accuracy = correct predictions / labeled messages (WARNING counts as MATCH){}".format(
            DIM, RESET))

    # Per-message verdict grid
    print("\n  {}Per-Message Verdict Grid{}".format(BOLD, RESET))
    print("  " + "-" * (105 if has_labels else 95))

    # Header row
    hdr = "  {:<4} {:<50}".format("#", "Message")
    if has_labels:
        hdr += " {:>8}".format("EXPECTED")
    for m in modes:
        hdr += " {:>10}".format(m.upper())
    print(hdr)
    print("  " + "-" * (105 if has_labels else 95))

    for i, (msg, mr) in enumerate(zip(messages, all_mode_results), 1):
        disp = msg if len(msg) <= 48 else msg[:45] + "..."
        row = "  {:<4} {:<50}".format(i, disp)

        # Expected column
        if has_labels:
            lbl = labels[i - 1] if labels[i - 1] is not None else "--"
            if lbl == "MATCH":
                row += " {}  {:>5}  {}".format(RED, "MATCH", RESET)
            elif lbl == "NO MATCH":
                row += " {}  {:>5}  {}".format(GREEN, "CLEAN", RESET)
            else:
                row += " {:>8}".format("--")

        for m in modes:
            if m in mr:
                v = mr[m]["verdict"]
                vc = _verdict_color(v)
                short = v[:5] if v != "NO MATCH" else "CLEAN"
                # Mark wrong predictions
                if has_labels and labels[i - 1] is not None:
                    predicted_match = v in ("MATCH", "WARNING")
                    expected_match = labels[i - 1] == "MATCH"
                    if predicted_match != expected_match:
                        short += " \u2717"  # ✗ mark
                row += " {}  {:>7}  {}".format(vc, short, RESET)
            else:
                row += " {:>10}".format("--")
        print(row)

    # Agreement stats
    if len(modes) >= 2:
        agree = 0
        for mr in all_mode_results:
            verdicts = set()
            for m in modes:
                if m in mr:
                    verdicts.add(mr[m]["verdict"])
            if len(verdicts) <= 1:
                agree += 1
        print("\n  {}All-mode agreement: {}/{} ({:.1f}%){}".format(
            DIM, agree, total, agree / total * 100 if total > 0 else 0, RESET))

    print()


def run_compare(filepath, model, embs, texts, cats,
                use_extraction=True,
                neg_embs=None, neg_texts=None, neutral_embs=None,
                include_llm=False):
    """Run all modes on a file and display comparison."""
    if not os.path.exists(filepath):
        print("  ERROR: File not found: {}".format(filepath)); sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    sentences, labels = _parse_labeled_file(content)

    has_labels = any(l is not None for l in labels)
    modes_str = "cosine, nli, hybrid"
    if include_llm:
        modes_str += ", llm"
    label_str = " (with ground truth)" if has_labels else ""
    print("  {}Compare mode:{} {} on {} messages{} from: {}".format(
        BOLD, RESET, modes_str, len(sentences), label_str, filepath))
    print("  " + "=" * 100)

    all_mode_results = []
    for idx, sent in enumerate(sentences, 1):
        mr = score_all_modes(
            sent, model, embs, texts, cats,
            neg_embs=neg_embs, neg_texts=neg_texts, neutral_embs=neutral_embs,
            use_extraction=use_extraction, include_llm=include_llm)
        all_mode_results.append(mr)
        display_compare(sent, mr, index=idx, total=len(sentences))

    display_compare_summary(all_mode_results, sentences, labels=labels)


def run_compare_interactive(model, embs, texts, cats,
                            use_extraction=True,
                            neg_embs=None, neg_texts=None, neutral_embs=None,
                            include_llm=False):
    """Interactive compare mode — type messages, see all modes side-by-side."""
    global SPELLCHECK_ENABLED
    modes_str = "cosine, nli, hybrid"
    if include_llm:
        modes_str += ", llm"
    print("{}Compare Mode: {}{}".format(BOLD, modes_str, RESET))
    print("  Type a message to compare all scoring modes. Commands: /quit  /llm\n")
    print("  Thresholds: match={}{}{}  warning={}{}{}".format(
        RED, MATCH_THRESHOLD, RESET, YELLOW, WARNING_THRESHOLD, RESET))
    print("  " + "-" * 70)

    all_mode_results = []
    all_messages = []

    while True:
        try:
            msg = _read_message("\n  {}Message>{} ".format(BOLD, RESET))
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break
        if not msg:
            continue

        if msg.startswith("/"):
            c = msg.lower().split()
            if c[0] in ("/quit", "/exit", "/q"):
                break
            elif c[0] == "/llm":
                include_llm = not include_llm
                if include_llm:
                    try:
                        _load_llm_config()
                        print("  LLM: ON")
                    except SystemExit:
                        include_llm = False
                        print("  LLM: OFF (config not found)")
                else:
                    print("  LLM: OFF")
                continue
            elif c[0] == "/spellcheck":
                SPELLCHECK_ENABLED = not SPELLCHECK_ENABLED
                if SPELLCHECK_ENABLED:
                    _load_spellchecker()
                print("  Spell correction: {}".format(
                    "ON" if SPELLCHECK_ENABLED else "OFF"))
                continue
            elif c[0] == "/summary":
                if all_mode_results:
                    display_compare_summary(all_mode_results, all_messages)
                else:
                    print("  No messages scored yet.")
                continue
            elif c[0] == "/help":
                print("  /llm        \u2014 toggle LLM judge in comparison")
                print("  /spellcheck \u2014 toggle autocorrect spell checking")
                print("  /summary    \u2014 show summary of all messages so far")
                print("  /quit       \u2014 exit (shows summary)")
                continue
            else:
                print("  Unknown command. /help for options.")
                continue

        mr = score_all_modes(
            msg, model, embs, texts, cats,
            neg_embs=neg_embs, neg_texts=neg_texts, neutral_embs=neutral_embs,
            use_extraction=use_extraction, include_llm=include_llm)
        all_mode_results.append(mr)
        all_messages.append(msg)
        display_compare(msg, mr)

    if all_mode_results:
        display_compare_summary(all_mode_results, all_messages)


# ---------------------------------------------------------------------------
# 3D Visualization
# ---------------------------------------------------------------------------

def show_graph(model):
    """3D interactive scatter plot of anchor embeddings using PCA."""
    try:
        import matplotlib
        from sklearn.decomposition import PCA
        import numpy as np
    except ImportError as e:
        print("\n  ERROR: Missing dependency for --graph: {}".format(e))
        print("  Run: pip install matplotlib scikit-learn")
        sys.exit(1)

    backend_set = False
    for backend in ["macosx", "Qt5Agg", "TkAgg"]:
        try:
            matplotlib.use(backend)
            backend_set = True
            break
        except Exception:
            continue
    if not backend_set:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    using_agg = matplotlib.get_backend().lower() == "agg"

    # Collect positive anchors
    pos_texts = []
    for cat, examples in ANCHORS.items():
        for ex in examples:
            pos_texts.append(ex)

    # Collect negative anchors
    neg_texts = []
    for cat, examples in NEGATIVE_ANCHORS.items():
        for ex in examples:
            neg_texts.append(ex)

    n_pos = len(pos_texts)
    n_neg = len(neg_texts)
    all_texts = pos_texts + neg_texts

    print("  Embedding {} positive + {} negative anchors...".format(n_pos, n_neg))
    all_embs = model.encode(all_texts, show_progress_bar=False)
    prop_emb = model.encode([PROPOSITION])

    combined = np.vstack([all_embs, prop_emb])

    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(combined)
    pos_coords = coords_3d[:n_pos]
    neg_coords = coords_3d[n_pos:n_pos + n_neg]
    prop_coord = coords_3d[-1]

    explained = pca.explained_variance_ratio_
    print("  PCA variance explained: {:.1f}% + {:.1f}% + {:.1f}% = {:.1f}%".format(
        explained[0]*100, explained[1]*100, explained[2]*100, sum(explained)*100))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Positive anchors (colored)
    ax.scatter(pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2],
               c="#2196F3", s=50, alpha=0.7,
               edgecolors="white", linewidths=0.5)

    # Negative anchors (black)
    ax.scatter(neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2],
               c="black", s=35, alpha=0.5,
               edgecolors="white", linewidths=0.5, marker="x")

    # Proposition (red star)
    ax.scatter([prop_coord[0]], [prop_coord[1]], [prop_coord[2]],
               c="red", s=300, marker="*", edgecolors="black",
               linewidths=1, zorder=10)

    # Light lines from proposition to positive anchors
    for i in range(n_pos):
        ax.plot([prop_coord[0], pos_coords[i, 0]],
                [prop_coord[1], pos_coords[i, 1]],
                [prop_coord[2], pos_coords[i, 2]],
                color="red", alpha=0.08, linewidth=0.5)

    ax.set_xlabel("PC1 ({:.1f}%)".format(explained[0]*100))
    ax.set_ylabel("PC2 ({:.1f}%)".format(explained[1]*100))
    ax.set_zlabel("PC3 ({:.1f}%)".format(explained[2]*100))
    ax.set_title("Semantic Anchors: {} \u2014 {} positive (blue) + {} negative (black)".format(
        SCRIPT_NAME, n_pos, n_neg), fontsize=13, fontweight="bold")

    norms = np.linalg.norm(all_embs[:n_pos], axis=1, keepdims=True)
    normed = all_embs[:n_pos] / (norms + 1e-10)
    sim_m = normed @ normed.T
    np.fill_diagonal(sim_m, 0)
    stats = "Positive avg sim: {:.4f} | Min: {:.4f} | Max: {:.4f}".format(
        sim_m.mean(), sim_m[sim_m > 0].min(), sim_m.max())
    ax.text2D(0.02, 0.96, stats, transform=ax.transAxes, fontsize=8,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if using_agg:
        out_file = "semantic_anchor_{}_3d.png".format(SCRIPT_NAME)
        plt.savefig(out_file, dpi=150)
        print("  Backend is non-interactive. Saved plot to: {}".format(out_file))
        import subprocess, platform
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", out_file])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", out_file])
            elif platform.system() == "Windows":
                os.startfile(out_file)
        except Exception:
            pass
    else:
        print("  Showing 3D plot (drag to rotate)...")
        plt.show()


# ---------------------------------------------------------------------------
# Multi-line input handling
# ---------------------------------------------------------------------------

def _read_message(prompt):
    """
    Read a complete message from the user. Handles both single-line and
    multi-line input reliably without timing-based paste detection.

    Rules:
      - If the line ends with sentence-ending punctuation (. ? ! " ' ) ]),
        it's treated as complete and evaluated immediately.
      - If it doesn't (indicating a line break mid-sentence from a paste),
        a continuation prompt is shown. The user keeps pasting/typing until
        they enter a blank line.
      - Commands (starting with /) are always single-line.
    """
    first_line = input(prompt).strip()
    if not first_line:
        return first_line

    # Commands are always single-line
    if first_line.startswith("/"):
        return first_line

    # Check if the line looks complete (ends with sentence punctuation)
    if first_line and first_line[-1] in '.?!"\')]':
        # Try to drain any paste buffer quickly (best-effort)
        try:
            import select, sys
            lines = [first_line]
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.08)
                if ready:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        lines.append(line)
                else:
                    break
            return " ".join(lines)
        except (ImportError, OSError):
            return first_line

    # Line doesn't end with punctuation — it's a multi-line paste
    # Show continuation prompt and read until blank line
    lines = [first_line]
    while True:
        try:
            line = input("  {}...>{} ".format(BOLD, RESET))
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        lines.append(line.strip())

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(model, embs, texts, cats, verbose=False, mode="cosine",
                    use_extraction=True,
                    neg_embs=None, neg_texts=None, neutral_embs=None):
    global SPELLCHECK_ENABLED
    mode_labels = {
        "cosine": "Cosine similarity",
        "nli": "NLI entailment (all anchors)",
        "hybrid": "Hybrid ({:.0f}% NLI + {:.0f}% KNN)".format(
            HYBRID_NLI_WEIGHT * 100, (1 - HYBRID_NLI_WEIGHT) * 100),
        "llm": "LLM Judge",
    }
    mode_label = mode_labels.get(mode, mode)
    extract_label = " + extraction" if use_extraction else ""
    neg_label = " + contrastive" if neg_texts else ""
    print("{}Interactive Mode: {}{}{}{}".format(BOLD, mode_label, extract_label, neg_label, RESET))
    print("  Type a message to evaluate. Commands: /verbose  /top N  /mode MODE  /extract  /quit\n")
    print("  Thresholds: match={}{}{}  warning={}{}{}".format(
        RED, MATCH_THRESHOLD, RESET, YELLOW, WARNING_THRESHOLD, RESET))
    print("  Mode: {}{}{}".format(GREEN, mode_label, RESET))
    print("  Extraction: {}{}{}  (long-message defense, tau={})".format(
        GREEN if use_extraction else YELLOW,
        "ON" if use_extraction else "OFF",
        RESET, EXTRACTION_RELEVANCE_TAU))
    if neg_texts:
        neutral_count = len(NEUTRAL_ANCHORS) if NEUTRAL_ANCHORS else 0
        print("  Contrastive: {}ON{} ({} negative, {} neutral anchors)".format(
            GREEN, RESET, len(neg_texts), neutral_count))
    else:
        print("  Contrastive: {}OFF{} (no negative anchors)".format(YELLOW, RESET))
    print("  " + "-" * 70)
    top_n = 3
    verbose_extract = False

    while True:
        try:
            msg = _read_message("\n  {}Message>{} ".format(BOLD, RESET))
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
            elif c[0] == "/mode" and len(c) > 1:
                new_mode = c[1]
                if new_mode in ("cosine", "nli", "hybrid", "llm"):
                    mode = new_mode
                    if mode in ("nli", "hybrid"):
                        _load_cross_encoder()
                    if mode == "llm":
                        _load_llm_config()
                    print("  Mode: {}".format(mode_labels.get(mode, mode)))
                else:
                    print("  Available modes: cosine, nli, hybrid, llm")
                continue
            elif c[0] == "/extract":
                use_extraction = not use_extraction
                print("  Extraction: {} (tau={})".format(
                    "ON" if use_extraction else "OFF", EXTRACTION_RELEVANCE_TAU)); continue
            elif c[0] == "/extractv":
                verbose_extract = not verbose_extract
                print("  Verbose extraction: {}".format(
                    "ON" if verbose_extract else "OFF")); continue
            elif c[0] == "/spellcheck":
                SPELLCHECK_ENABLED = not SPELLCHECK_ENABLED
                if SPELLCHECK_ENABLED:
                    _load_spellchecker()
                print("  Spell correction: {}".format(
                    "ON" if SPELLCHECK_ENABLED else "OFF")); continue
            elif c[0] == "/help":
                print("  /verbose   \u2014 toggle full table")
                print("  /top N     \u2014 show top N results")
                print("  /mode MODE \u2014 switch scoring: cosine, nli, hybrid, llm")
                print("  /extract   \u2014 toggle proposition-guided extraction (long msgs)")
                print("  /extractv  \u2014 toggle verbose extraction diagnostics")
                print("  /spellcheck\u2014 toggle autocorrect spell checking")
                print("  /quit      \u2014 exit"); continue
            else:
                print("  Unknown command. /help for options."); continue

        # --- Dispatch by mode ---
        if mode == "llm":
            llm_result, corrected, corrections, cosine_results = score_message_llm(
                msg, PROPOSITION, model=model, all_embeddings=embs,
                all_texts=texts, all_categories=cats,
                neg_embeddings=neg_embs, neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            display_llm_result(llm_result, corrected, corrections, cosine_results, top_n)

        elif mode == "hybrid":
            results, corrected, corrections, extraction, neg_score, debug = score_message_hybrid(
                msg, texts, cats, model=model, all_embeddings=embs,
                neg_texts=neg_texts, neg_embeddings=neg_embs,
                neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            if verbose:
                display_verbose_hybrid(results, corrected, corrections, extraction,
                                       neg_score=neg_score, debug=debug)
            else:
                display_default_hybrid(results, corrected, corrections, extraction,
                                       top_n, neg_score=neg_score, debug=debug)

        elif mode == "nli":
            results, corrected, corrections, extraction, neg_score, prop_score, nli_debug = score_message_nli(
                msg, texts, cats, model=model, all_embeddings=embs,
                neg_texts=neg_texts, neg_embeddings=neg_embs,
                neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            if verbose:
                display_verbose_nli(results, corrected, corrections, extraction,
                                    neg_score=neg_score, debug=nli_debug)
            else:
                display_default_nli(results, corrected, corrections, extraction,
                                    top_n, neg_score=neg_score, debug=nli_debug)

        else:  # cosine
            results, extraction, knn_info = score_message(model, msg, embs, texts, cats,
                                               neg_embeddings=neg_embs, neg_texts=neg_texts,
                                               neutral_embeddings=neutral_embs,
                                               use_extraction=use_extraction)
            if verbose:
                display_verbose(results, extraction, knn_info=knn_info)
            else:
                display_default(results, extraction, top_n, verbose_extract, knn_info=knn_info)


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def run_file(filepath, model, embs, texts, cats, verbose=False, mode="cosine",
             use_extraction=True,
             neg_embs=None, neg_texts=None, neutral_embs=None):
    if not os.path.exists(filepath):
        print("  ERROR: File not found: {}".format(filepath)); sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    sentences, labels = _parse_labeled_file(content)
    mode_labels = {
        "cosine": "cosine", "nli": "NLI-only",
        "hybrid": "hybrid", "llm": "LLM judge",
    }
    mode_label = " ({})".format(mode_labels.get(mode, mode))
    extract_label = " + extraction" if use_extraction else ""
    neg_label = " + contrastive" if neg_texts else ""
    print("  Processing {} sentences from: {}{}{}{}\n".format(
        len(sentences), filepath, mode_label, extract_label, neg_label))
    print("  " + "=" * 100)
    all_top_scores = []
    all_verdicts = []

    for idx, sent in enumerate(sentences, 1):
        disp = sent[:80] + ("..." if len(sent) > 80 else "")
        print("\n  {}[{}/{}] Message:{} \"{}\"".format(BOLD, idx, len(sentences), RESET, disp))

        if mode == "llm":
            llm_result, corrected, corrections, cosine_results = score_message_llm(
                sent, PROPOSITION, model=model, all_embeddings=embs,
                all_texts=texts, all_categories=cats,
                neg_embeddings=neg_embs, neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            display_llm_result(llm_result, corrected, corrections, cosine_results)
            verdict = llm_result.get("verdict", "ERROR")
            all_top_scores.append(1.0 if llm_result.get("match") else 0.0)
            all_verdicts.append(verdict)

        elif mode == "hybrid":
            results, corrected, corrections, extraction, neg_score, debug = score_message_hybrid(
                sent, texts, cats, model=model, all_embeddings=embs,
                neg_texts=neg_texts, neg_embeddings=neg_embs,
                neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            top_score = results[0][0]
            if verbose:
                display_verbose_hybrid(results, corrected, corrections, extraction,
                                       neg_score=neg_score, debug=debug)
            else:
                display_default_hybrid(results, corrected, corrections, extraction,
                                       neg_score=neg_score, debug=debug)
            all_top_scores.append(top_score)

        elif mode == "nli":
            results, corrected, corrections, extraction, neg_score, prop_score, nli_debug = score_message_nli(
                sent, texts, cats, model=model, all_embeddings=embs,
                neg_texts=neg_texts, neg_embeddings=neg_embs,
                neutral_embeddings=neutral_embs,
                use_extraction=use_extraction)
            top_score = results[0][0]
            if verbose:
                display_verbose_nli(results, corrected, corrections, extraction,
                                    neg_score=neg_score, debug=nli_debug)
            else:
                display_default_nli(results, corrected, corrections, extraction,
                                    neg_score=neg_score, debug=nli_debug)
            all_top_scores.append(top_score)

        else:  # cosine
            results, extraction, knn_info = score_message(model, sent, embs, texts, cats,
                                               neg_embeddings=neg_embs, neg_texts=neg_texts,
                                               neutral_embeddings=neutral_embs,
                                               use_extraction=use_extraction)
            top_score = results[0][0]
            if verbose:
                display_verbose(results, extraction, knn_info=knn_info)
            else:
                display_default(results, extraction, knn_info=knn_info)
            all_top_scores.append(knn_info["knn_score"])

    # Summary
    has_labels = any(l is not None for l in labels)

    if mode == "llm":
        matches = sum(1 for v in all_verdicts if v == "MATCH")
        warns = sum(1 for v in all_verdicts if v == "WARNING")
        errors = sum(1 for v in all_verdicts if v == "ERROR")
        clean = sum(1 for v in all_verdicts if v == "NO MATCH")
    elif mode == "cosine":
        matches = sum(1 for s in all_top_scores if s >= MATCH_THRESHOLD)
        warns = sum(1 for s in all_top_scores if WARNING_THRESHOLD <= s < MATCH_THRESHOLD)
        clean = sum(1 for s in all_top_scores if s < WARNING_THRESHOLD)
        errors = 0
    else:
        matches = sum(1 for s in all_top_scores if s >= MATCH_THRESHOLD)
        warns = sum(1 for s in all_top_scores if WARNING_THRESHOLD <= s < MATCH_THRESHOLD)
        clean = sum(1 for s in all_top_scores if s < WARNING_THRESHOLD)
        errors = 0

    print("\n  {}Summary{}".format(BOLD, RESET))
    print("  " + "=" * 60)
    print("  Total sentences:  {}".format(len(sentences)))
    print("  {}\u25a0 Matches:         {}{}".format(RED, matches, RESET))
    print("  {}\u25a0 Warnings:        {}{}".format(YELLOW, warns, RESET))
    print("  {}\u25a0 No match:        {}{}".format(GREEN, clean, RESET))
    if errors:
        print("  \u25a0 Errors:          {}".format(errors))

    # Ground truth accuracy
    if has_labels:
        correct_count = 0
        labeled_count = 0
        for i, lbl in enumerate(labels):
            if lbl is None:
                continue
            labeled_count += 1
            if mode == "llm":
                predicted_match = all_verdicts[i] in ("MATCH", "WARNING")
            elif mode == "cosine":
                predicted_match = all_top_scores[i] >= WARNING_THRESHOLD  # KNN ratio above warning
            else:
                predicted_match = all_top_scores[i] >= WARNING_THRESHOLD
            expected_match = lbl == "MATCH"
            if predicted_match == expected_match:
                correct_count += 1
        if labeled_count > 0:
            print("  {}Accuracy:         {}/{} ({:.1f}%){}".format(
                BOLD, correct_count, labeled_count,
                correct_count / labeled_count * 100, RESET))
    print()


# ---------------------------------------------------------------------------
# Logging — tee stdout to log file
# ---------------------------------------------------------------------------

class TeeLogger:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.log_path = log_path

    def write(self, message):
        self.terminal.write(message)
        # Strip ANSI escape codes for clean log file
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', message)
        self.log_file.write(clean)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return self.terminal.encoding

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def _start_logging():
    """Start logging to a timestamped file. Returns the TeeLogger or None."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, "log_{}_{}.txt".format(SCRIPT_NAME, ts))
        tee = TeeLogger(log_path)
        sys.stdout = tee
        return tee
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Start logging to file
    _tee = _start_logging()

    parser = argparse.ArgumentParser(
        description="Semantic Anchor Evaluator: " + SCRIPT_NAME)
    parser.add_argument("--file", "-f", type=str, default=None,
        help="Input file with sentences separated by ###")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Show full scored table (all anchors sorted)")
    parser.add_argument("--show-examples", action="store_true",
        help="Print all positive examples by category")
    parser.add_argument("--graph", "-g", action="store_true",
        help="Show 3D interactive visualization of anchor spread")
    parser.add_argument("--mode", "-m", type=str, default="cosine",
        choices=["cosine", "nli", "hybrid", "llm"],
        help="Scoring mode: cosine, nli, hybrid (NLI+KNN blend), llm (LLM judge)")
    # Backward-compatible aliases
    parser.add_argument("--nli", action="store_true",
        help="Shortcut for --mode nli")
    parser.add_argument("--hybrid", action="store_true",
        help="Shortcut for --mode hybrid")
    parser.add_argument("--llm", action="store_true",
        help="Shortcut for --mode llm")
    parser.add_argument("--no-extract", action="store_true",
        help="Disable proposition-guided extraction for long messages")
    parser.add_argument("--spellcheck", action="store_true",
        help="Enable autocorrect spell checking (disabled by default)")
    parser.add_argument("--compare", "-c", action="store_true",
        help="Compare all scoring modes side-by-side (cosine, nli, hybrid, +llm)")
    args = parser.parse_args()

    # Resolve mode from flags
    mode = args.mode
    if args.compare:
        mode = "compare"
    elif args.nli:
        mode = "nli"
    elif args.hybrid:
        mode = "hybrid"
    elif args.llm:
        mode = "llm"

    use_extraction = not args.no_extract

    # --- Spellcheck: CLI flag or config_<name>.ini ---
    global SPELLCHECK_ENABLED
    if args.spellcheck:
        SPELLCHECK_ENABLED = True
    else:
        # Check config_<name>.ini for spellcheck setting
        try:
            import configparser as _cp
            _cfg = _cp.ConfigParser()
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "config_{}.ini".format(SCRIPT_NAME))
            if os.path.exists(_cfg_path):
                _cfg.read(_cfg_path)
                if _cfg.has_option("options", "spellcheck"):
                    SPELLCHECK_ENABLED = _cfg.get("options", "spellcheck").lower() in ("true", "1", "yes", "on")
                elif _cfg.has_option("llm_judge", "spellcheck"):
                    SPELLCHECK_ENABLED = _cfg.get("llm_judge", "spellcheck").lower() in ("true", "1", "yes", "on")
        except Exception:
            pass

    print_banner()

    if args.show_examples:
        print_examples()
        if not args.file and not args.graph:
            try:
                r = input("  Enter interactive mode? [Y/n] ").strip().lower()
                if r == "n":
                    if _tee: _tee.close(); print("  Log saved: {}".format(_tee.log_path))
                    return
            except (EOFError, KeyboardInterrupt):
                print()
                if _tee: _tee.close(); print("  Log saved: {}".format(_tee.log_path))
                return

    model = load_model()

    if args.graph:
        show_graph(model)
        if _tee: _tee.close(); print("  Log saved: {}".format(_tee.log_path))
        return

    (all_texts, all_cats, all_embs,
     neg_texts, neg_cats, neg_embs,
     neutral_embs) = prepare_anchors(model)

    if use_extraction:
        print("  Extraction: {}ON{} (tau={}, min_words={})".format(
            GREEN, RESET, EXTRACTION_RELEVANCE_TAU, EXTRACTION_MIN_WORDS))

    # Log model sources
    if _model_sources:
        emb_name, emb_src = _model_sources.get("embedding", (EMBEDDING_MODEL, "default"))
        nli_name, nli_src = _model_sources.get("nli", (NLI_MODEL, "default"))
        print("  Embedding:  {} {}(from {}){}".format(emb_name, DIM, emb_src, RESET))
        print("  NLI model:  {} {}(from {}){}".format(nli_name, DIM, nli_src, RESET))

    spell_label = "{}ON{}".format(GREEN, RESET) if SPELLCHECK_ENABLED else "{}OFF{} (enable with --spellcheck)".format(DIM, RESET)
    print("  Spell correction: {}\n".format(spell_label))

    # Load cross-encoder for modes that need it
    if mode in ("nli", "hybrid", "compare"):
        _load_cross_encoder()
        total_anchors = sum(len(v) for v in ANCHORS.values())
        if mode == "compare":
            print("  Compare mode: {}ON{} (all scoring methods side-by-side)".format(
                GREEN, RESET))
        elif mode == "hybrid":
            print("  Hybrid mode: {}ON{} ({:.0f}% NLI + {:.0f}% cosine KNN)".format(
                GREEN, RESET, HYBRID_NLI_WEIGHT * 100, (1 - HYBRID_NLI_WEIGHT) * 100))
        else:
            print("  NLI-only mode: {}ON{}".format(GREEN, RESET))
            print("  Scoring: NLI entailment on ALL {} anchors".format(total_anchors))
            print("  Long messages: proposition-guided extraction + multi-view NLI\n")

    elif mode == "llm":
        _load_llm_config()
        print("  LLM Judge: {}ON{} ({} / {})".format(
            GREEN, RESET, _llm_config["provider"], _llm_config["model"]))
        prov = _llm_config["provider"].lower()
        if prov in ("ollama", "local", "lmstudio", "vllm"):
            print("  Base URL:   {}".format(_llm_config.get("base_url", "")))
        if prov == "ollama":
            _ollama_ensure_model(_llm_config["model"],
                                _llm_config.get("base_url"))
        print()

    # Check if LLM is available for compare mode
    include_llm = False
    if mode == "compare":
        try:
            _load_llm_config(silent=True)
            if _llm_config.get("api_key", "").startswith("YOUR_"):
                print("  LLM Judge: {}SKIPPED{} (edit config_{}.ini [llm_judge] api_key)".format(
                    YELLOW, RESET, SCRIPT_NAME))
            else:
                include_llm = True
                print("  LLM Judge: {}ON{} ({} / {})".format(
                    GREEN, RESET, _llm_config["provider"], _llm_config["model"]))
                if _llm_config["provider"].lower() == "ollama":
                    _ollama_ensure_model(_llm_config["model"],
                                        _llm_config.get("base_url"))
        except SystemExit:
            print("  LLM Judge: {}SKIPPED{} (no config_{}.ini found)".format(
                YELLOW, RESET, SCRIPT_NAME))
        print()

    # Dispatch
    if mode == "compare":
        if args.file:
            run_compare(args.file, model, all_embs, all_texts, all_cats,
                        use_extraction, neg_embs=neg_embs, neg_texts=neg_texts,
                        neutral_embs=neutral_embs, include_llm=include_llm)
        else:
            run_compare_interactive(model, all_embs, all_texts, all_cats,
                                    use_extraction, neg_embs=neg_embs,
                                    neg_texts=neg_texts, neutral_embs=neutral_embs,
                                    include_llm=include_llm)
    elif args.file:
        run_file(args.file, model, all_embs, all_texts, all_cats,
                 args.verbose, mode, use_extraction,
                 neg_embs=neg_embs, neg_texts=neg_texts, neutral_embs=neutral_embs)
    else:
        run_interactive(model, all_embs, all_texts, all_cats,
                        args.verbose, mode, use_extraction,
                        neg_embs=neg_embs, neg_texts=neg_texts, neutral_embs=neutral_embs)

    # Close logger and report log path
    if _tee is not None:
        log_path = _tee.log_path
        _tee.close()
        print("  Log saved: {}".format(log_path))


if __name__ == "__main__":
    main()
'''


def save_anchors_json(output_dir, name, proposition, anchors_dict,
                      match_thresh, warn_thresh, config,
                      negative_dict=None, neutral_list=None, role="user"):
    """Save anchors and metadata to a JSON file."""
    import datetime
    total = sum(len(v) for v in anchors_dict.values())
    neg_total = sum(len(v) for v in (negative_dict or {}).values())
    neutral_total = len(neutral_list) if neutral_list else 0
    data = {
        "proposition": proposition,
        "name": name,
        "role": role,
        "match_threshold": match_thresh,
        "warning_threshold": warn_thresh,
        "anchors": anchors_dict,
        "negative_anchors": negative_dict or {},
        "neutral_anchors": neutral_list or [],
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "provider": config.get("llm", "provider"),
            "model": config.get("llm", "model"),
            "embedding_model": config.get("anchors", "embedding_model"),
            "nli_model": config.get("anchors", "nli_model",
                                    fallback="cross-encoder/nli-deberta-v3-large"),
            "total_anchors": total,
            "total_negative_anchors": neg_total,
            "total_neutral_anchors": neutral_total,
        },
    }
    path = os.path.join(output_dir, "anchors_list_{}.json".format(name))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_existing_anchors(output_dir, name):
    """Load an existing anchors JSON file. Returns (data_dict, path) or (None, None)."""
    path = os.path.join(output_dir, "anchors_list_{}.json".format(name))
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, path


def generate_script(name):
    """Generate standalone evaluator script. Only injects the name — everything
    else is loaded from anchors_list_<n>.json at runtime."""
    title_underline = "=" * (29 + len(name))
    script = EVALUATOR_TEMPLATE
    script = script.replace("%%NAME%%", name)
    script = script.replace("%%TITLE_UNDERLINE%%", title_underline)
    return script


# ---------------------------------------------------------------------------
# Generator main
# ---------------------------------------------------------------------------

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
              %(prog)s -p "The user requests weapon instructions." -name weapons
              %(prog)s -p "..." -name weapons -n 50 --rounds 3
              %(prog)s -name weapons -gs         # Regenerate script only (reuse anchors)
              %(prog)s -p "..." -name weapons -ga # Regenerate anchors only
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

    name = name.replace(" ", "_").replace("-", "_").lower()

    # --- Create output directory ---
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

    # --- Generate anchors ---
    categories = parse_categories(config, role=role)
    anchors_dict = generate_diverse_anchors(
        proposition, categories, target_n, num_rounds, config, role=role)

    total = sum(len(v) for v in anchors_dict.values())
    print("\n  {}\u2713 Final: {} diversity-optimized positive anchors{}".format(GREEN, total, RESET))

    # --- Generate negative (contrastive) anchors ---
    # 2:1 negative-to-positive ratio by default — the benign space is much
    # larger than the harmful space and needs broader coverage for KNN voting.
    neg_ratio = float(config.get("anchors", "negative_ratio", fallback="2.0"))
    neg_target = max(target_n, int(target_n * neg_ratio))
    negative_dict = generate_negative_anchors(
        proposition, anchors_dict, neg_target, config, num_rounds=num_rounds, role=role)

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
        config, negative_dict=negative_dict, neutral_list=neutral_list, role=role)
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
                f.write("# Score >= match_threshold → MATCH\n")
                f.write("match_threshold = 0.85\n\n")
                f.write("# Score >= warning_threshold (but < match) → WARNING\n")
                f.write("warning_threshold = 0.70\n\n")
                f.write("# KNN neighborhood size for cosine voting (default: 20)\n")
                f.write("# Higher = more stable, lower = more sensitive\n")
                f.write("knn_size = 20\n\n")
                f.write("# Hybrid mode blend: weight for NLI component (default: 0.75)\n")
                f.write("# Cosine KNN weight = 1.0 - hybrid_nli_weight\n")
                f.write("hybrid_nli_weight = 0.75\n\n")
                f.write("# NLI KNN abstain margin (default: 0.15)\n")
                f.write("# When positive/negative evidence margin is below this, flag as abstain.\n")
                f.write("# Abstain = ambiguous, consider routing to LLM judge.\n")
                f.write("# Higher = more abstains (safer), lower = force more decisions.\n")
                f.write("nli_abstain_margin = 0.15\n\n")
                f.write("# NLI cosine pre-filter size (default: 40)\n")
                f.write("# How many candidates to retrieve by cosine before NLI re-ranking.\n")
                f.write("# Increase for broad propositions spanning many subtopics (60-80).\n")
                f.write("nli_retrieve_k = 40\n\n")
                f.write("# NLI vote neighborhood size (default: 20)\n")
                f.write("# After NLI re-ranking, vote on the top K candidates.\n")
                f.write("nli_vote_k = 20\n\n")
                f.write("# NLI asymmetric forward weight (default: 0.7)\n")
                f.write("# 0.7 = 70%% forward (message->anchor) + 30%% backward (anchor->message)\n")
                f.write("nli_fwd_weight = 0.7\n\n")
                f.write("# Adaptive hybrid weighting (default: true)\n")
                f.write("# When true, shifts weight toward cosine when NLI is uncertain.\n")
                f.write("# When false, uses fixed hybrid_nli_weight for all messages.\n")
                f.write("adaptive_hybrid = true\n")
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
            f.write("# Score >= match_threshold → MATCH\n")
            f.write("match_threshold = 0.85\n\n")
            f.write("# Score >= warning_threshold (but < match) → WARNING\n")
            f.write("warning_threshold = 0.70\n\n")
            f.write("# KNN neighborhood size for cosine voting (default: 20)\n")
            f.write("# Higher = more stable, lower = more sensitive\n")
            f.write("knn_size = 20\n\n")
            f.write("# Hybrid mode blend: weight for NLI component (default: 0.75)\n")
            f.write("# Cosine KNN weight = 1.0 - hybrid_nli_weight\n")
            f.write("# Example: 0.75 = 75%% NLI + 25%% cosine KNN\n")
            f.write("hybrid_nli_weight = 0.75\n\n")
            f.write("# NLI KNN abstain margin (default: 0.15)\n")
            f.write("# When positive/negative evidence margin is below this, flag as abstain.\n")
            f.write("# Abstain cases are ambiguous — consider routing to LLM judge.\n")
            f.write("# Higher = more abstains, lower = force more decisions.\n")
            f.write("nli_abstain_margin = 0.15\n\n")
            f.write("# NLI cosine pre-filter size (default: 40)\n")
            f.write("# How many candidates to retrieve by cosine before NLI re-ranking.\n")
            f.write("# Increase for broad propositions spanning many subtopics (60-80).\n")
            f.write("nli_retrieve_k = 40\n\n")
            f.write("# NLI vote neighborhood size (default: 20)\n")
            f.write("# After NLI re-ranking, vote on the top K candidates.\n")
            f.write("nli_vote_k = 20\n\n")
            f.write("# NLI asymmetric forward weight (default: 0.7)\n")
            f.write("# 0.7 = 70%% forward (message->anchor) + 30%% backward (anchor->message)\n")
            f.write("nli_fwd_weight = 0.7\n\n")
            f.write("# Adaptive hybrid weighting (default: true)\n")
            f.write("# When true, dynamically shifts weight toward cosine when NLI confidence\n")
            f.write("# is low (KNN vote near 50/50), and toward NLI when confidence is high.\n")
            f.write("# Also applies a max-lift: if either mode is strongly confident,\n")
            f.write("# the hybrid score gets a floor boost.\n")
            f.write("# Set false to use fixed weights for all messages.\n")
            f.write("adaptive_hybrid = true\n\n\n")
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
    print("    python {} --mode hybrid             # Weighted blend: NLI + cosine KNN (recommended)".format(script_path))
    print("    python {} --mode llm                # LLM-as-judge scoring".format(script_path))
    print("    python {} --file input.txt          # Evaluate file".format(script_path))
    print("    python {} --compare                 # Compare all modes side-by-side".format(script_path))
    print("    python {} --compare -f input.txt    # Compare all modes on file".format(script_path))
    print("    python {} --verbose                 # Full table".format(script_path))
    print("    python {} --show-examples           # View anchors by category".format(script_path))


if __name__ == "__main__":
    main()
