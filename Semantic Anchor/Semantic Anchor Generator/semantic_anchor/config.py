import configparser
import os

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
        "hard_negative_ratio": "0.3",
        "orthogonal_axes": "true",
        "mmr_anneal": "true",
        "adversarial_filter": "true",
        "variance_threshold": "0.15",
    },
    "thresholds": {
        "match_threshold": "0.70",
        "warning_threshold": "0.50",
    },
    # ── AMO Training Defaults ──────────────────────────────────────────────
    # These are read from config.ini [training] section. Override any value
    # by adding it to your config.ini. Reasoning for key defaults:
    #
    # temperature = 0.05
    #   Controls softmax sharpness in the differentiable KNN scorer.
    #   Low (0.01-0.1): the closest 2-3 neighbors dominate the vote, creating
    #     clear score separation between positive and negative queries.
    #     This gives strong gradient signal for anchor movement.
    #   High (0.5+): weights spread evenly across all K neighbors, so every
    #     query scores ≈ label_mean (pos_anchors / total_anchors). Scores
    #     become indistinguishable, gradients vanish, nothing trains.
    #   Rule of thumb: keep temperature < 0.1 for anchor sets > 100.
    #
    # regularization = 0.01
    #   L2 penalty pulling anchors back toward original positions.
    #   Too high (0.1+): anchors can't move enough to improve boundaries.
    #   Too low (0.001): anchors drift freely, risk losing semantic meaning.
    #   The cosine anchoring constraint (min_similarity) is the hard safety
    #   net; regularization is the soft spring that keeps drift gradual.
    #
    # train_knn_k = 20
    #   Matches the evaluator's default knn_size. Training with K=20 means
    #   the scorer learns to optimize the same neighborhood the evaluator
    #   will use at inference time. K=40 dilutes the signal by including
    #   distant anchors that won't participate in inference voting.
    # ───────────────────────────────────────────────────────────────────────
    "training": {
        "learning_rate": "0.001",
        "epochs": "20",
        "regularization": "0.01",
        "temperature": "0.05",
        "batch_size": "8",
        "train_knn_k": "20",
        "kfold": "5",
        "drift_limit": "0.15",
        "min_similarity": "0.85",
        "synthetic_diversity_threshold": "0.95",
        "diversity_temp": "0.8",
        "adversarial_depth": "high",
        "context_variety": "true",
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
