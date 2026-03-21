"""Default grounding prompt templates and builder utilities.

Ported from the TemporalGuard web-app backend
(``Temporal Guard/backend/models/settings.py`` and
``Temporal Guard/backend/engine/grounding.py``).
"""

from __future__ import annotations

from temporalguard.policy import Proposition

# ---------------------------------------------------------------------------
# Default prompt constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a precise content classifier. Your task is to determine whether "
    "a message matches a given proposition. You must distinguish between:\n"
    "- Messages that genuinely request/intend what the proposition describes\n"
    "- Messages that use similar vocabulary but have legitimate/benign intent\n"
    "- Educational questions ABOUT the topic (not requesting to perform it)\n\n"
    "Respond with ONLY valid JSON, no other text."
)

DEFAULT_USER_PROMPT_TEMPLATE_USER: str = (
    'PROPOSITION: "{proposition_description}"\n\n'
    "Few-shot examples:\n"
    "{few_shot_examples}\n\n"
    'USER MESSAGE: "{message_text}"\n'
    "Does this message match the proposition? Consider:\n"
    "1. Does the user intend to perform the action described in the proposition?\n"
    "2. Or are they asking for education, defense, their own account, or general info?\n\n"
    "Respond with JSON:\n"
    "{{\n"
    '  "match": true or false,\n'
    '  "reasoning": "brief explanation"\n'
    "}}"
)

DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT: str = (
    'PROPOSITION: "{proposition_description}"\n\n'
    "Few-shot examples:\n"
    "{few_shot_examples}\n\n"
    'ASSISTANT MESSAGE: "{message_text}"\n'
    "Does this message match the proposition? Consider:\n"
    "1. Does the assistant response actually perform/provide what the proposition describes?\n"
    "2. Or is it general, defensive, refusal-oriented, or safety-focused discussion?\n"
    "3. Distinguish direct actionable assistance from high-level or preventive information\n\n"
    "Respond with JSON:\n"
    "{{\n"
    '  "match": true or false,\n'
    '  "reasoning": "brief explanation"\n'
    "}}"
)

# ---------------------------------------------------------------------------
# Helper: normalise few-shot fields to lists
# ---------------------------------------------------------------------------


def _to_list(value: list[str] | None) -> list[str]:
    """Normalise a few-shot field to a list (handles None gracefully)."""
    return list(value) if value else []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_few_shots(proposition: Proposition, role: str) -> str:
    """Render few-shot examples for a proposition into a prompt fragment.

    Returns ``"NONE"`` when the proposition has no examples.
    """
    role_label = "USER" if (role or "").strip().lower() != "assistant" else "ASSISTANT"
    positives = _to_list(proposition.few_shot_positive)
    negatives = _to_list(proposition.few_shot_negative)
    if not positives and not negatives:
        return "NONE"
    lines: list[str] = []
    i = 1
    for txt in positives:
        lines.append("Example {}:\nLabel: MATCH\n{} MESSAGE: {}".format(i, role_label, txt))
        i += 1
    for txt in negatives:
        lines.append("Example {}:\nLabel: NO_MATCH\n{} MESSAGE: {}".format(i, role_label, txt))
        i += 1
    return "\n\n".join(lines)


def build_grounding_prompts(
    proposition: Proposition,
    message_role: str,
    message_text: str,
    system_prompt: str | None,
    user_prompt_template_user: str | None,
    user_prompt_template_assistant: str | None,
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for a grounding LLM call.

    Returns a 2-tuple of ``(final_system_prompt, rendered_user_prompt)``.
    """
    role = (proposition.role or message_role or "user").strip().lower()
    if role == "assistant":
        final_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_template = user_prompt_template_assistant or DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT
    else:
        final_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_template = user_prompt_template_user or DEFAULT_USER_PROMPT_TEMPLATE_USER

    few_shot_examples = render_few_shots(proposition, role)
    user_prompt = user_template.format(
        proposition_description=proposition.description,
        proposition_role=proposition.role,
        message_role=message_role,
        message_role_upper=(message_role or "user").strip().upper(),
        message_text=message_text,
        few_shot_examples=few_shot_examples,
    )
    return final_system_prompt, user_prompt
