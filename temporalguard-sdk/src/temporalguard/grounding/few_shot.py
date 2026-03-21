"""Auto-generation of few-shot examples for propositions.

When a proposition has no user-provided few-shot examples, the SDK can
generate them automatically using the grounding LLM.  This mirrors the
behaviour of the Temporal Guard web application.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

from temporalguard.policy import Proposition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts (ported from Temporal Guard web-app)
# ---------------------------------------------------------------------------

FEW_SHOT_GENERATION_SYSTEM_PROMPT: str = (
    "You generate synthetic few-shot examples for proposition matching. "
    "Return ONLY valid JSON."
)


def _few_shot_generation_prompt(prop_description: str, role: str) -> str:
    """Build the user prompt for few-shot example generation."""
    role_norm = (role or "user").strip().lower()
    role_desc = "user messages" if role_norm == "user" else "assistant messages"
    return (
        "Create few-shot examples for proposition classification.\n\n"
        'PROPOSITION: "{}"\n'
        "ROLE: {}\n\n"
        "Generate exactly:\n"
        "1) 5 positive examples where proposition is clearly true.\n"
        "2) 5 negative examples that are tricky: same domain/terms, but proposition is false.\n\n"
        "Examples must be realistic {} and 1-2 sentences.\n\n"
        "Return JSON exactly:\n"
        "{{\n"
        '  "positive_examples": ["...", "...", "...", "...", "..."],\n'
        '  "negative_examples": ["...", "...", "...", "...", "..."]\n'
        "}}"
    ).format(prop_description, role_norm, role_desc)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict | None:
    """Try to pull a JSON object out of *text*."""
    text = text.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_few_shot_response(raw: str) -> tuple[list[str], list[str]]:
    """Parse the LLM response into ``(positive_examples, negative_examples)``.

    Returns ``([], [])`` on parse failure.
    """
    obj = _extract_json_object(raw)
    if not obj:
        logger.warning("Could not parse JSON from few-shot generation response")
        return [], []

    pos = obj.get("positive_examples", obj.get("positive", []))
    neg = obj.get("negative_examples", obj.get("negative", []))
    if not isinstance(pos, list) or not isinstance(neg, list):
        logger.warning("Missing positive_examples/negative_examples arrays")
        return [], []

    positives = [str(x).strip() for x in pos if str(x).strip()]
    negatives = [str(x).strip() for x in neg if str(x).strip()]
    return positives[:5], negatives[:5]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Type alias for an LLM call function.
LLMCallable = Callable[..., str]


def generate_few_shots(
    propositions: list[Proposition],
    call_llm: LLMCallable,
) -> None:
    """Fill in missing few-shot examples for propositions **in-place**.

    Only generates for propositions where *both* ``few_shot_positive`` and
    ``few_shot_negative`` are empty.  If the user has provided their own
    examples, they are left untouched.

    Parameters
    ----------
    propositions:
        The list of propositions to fill.  Mutated in-place.
    call_llm:
        A callable with signature
        ``(system_prompt: str, user_prompt: str, *, max_tokens: int) -> str``
        that sends a prompt to an LLM and returns the raw text response.
        Typically ``LLMGrounding._call_llm``.
    """
    for prop in propositions:
        if prop.few_shot_positive or prop.few_shot_negative:
            continue  # user-provided examples — keep as-is

        system = FEW_SHOT_GENERATION_SYSTEM_PROMPT
        user = _few_shot_generation_prompt(prop.description, prop.role)
        try:
            raw = call_llm(system, user, max_tokens=1000)
            positives, negatives = _parse_few_shot_response(raw)
            if positives or negatives:
                prop.few_shot_positive = positives
                prop.few_shot_negative = negatives
                logger.info(
                    "Generated %d positive + %d negative few-shot examples for '%s'",
                    len(positives), len(negatives), prop.prop_id,
                )
            else:
                logger.warning(
                    "Few-shot generation returned no examples for '%s'", prop.prop_id
                )
        except Exception:
            logger.warning(
                "Failed to generate few-shot examples for '%s'", prop.prop_id,
                exc_info=True,
            )
