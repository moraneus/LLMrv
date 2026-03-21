"""Tests for temporalguard.grounding.prompts."""

from __future__ import annotations

import pytest

from temporalguard.grounding.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_USER_PROMPT_TEMPLATE_USER,
    build_grounding_prompts,
    render_few_shots,
)
from temporalguard.policy import Proposition


# --- Default prompt constants ---


class TestDefaultPrompts:
    def test_system_prompt_contains_content_classifier(self):
        assert "content classifier" in DEFAULT_SYSTEM_PROMPT

    def test_user_template_has_placeholders(self):
        for placeholder in ("proposition_description", "message_text", "few_shot_examples"):
            assert "{" + placeholder + "}" in DEFAULT_USER_PROMPT_TEMPLATE_USER

    def test_assistant_template_has_placeholders(self):
        for placeholder in ("proposition_description", "message_text", "few_shot_examples"):
            assert "{" + placeholder + "}" in DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT


# --- render_few_shots ---


class TestRenderFewShots:
    def test_empty_examples_returns_none_string(self):
        prop = Proposition(prop_id="p", role="user", description="desc")
        assert render_few_shots(prop, "user") == "NONE"

    def test_with_examples_contains_match_and_no_match(self):
        prop = Proposition(
            prop_id="p",
            role="user",
            description="desc",
            few_shot_positive="good message",
            few_shot_negative="bad message",
        )
        result = render_few_shots(prop, "user")
        assert "MATCH" in result
        assert "NO_MATCH" in result
        assert "USER MESSAGE" in result

    def test_assistant_role_label(self):
        prop = Proposition(
            prop_id="p",
            role="assistant",
            description="desc",
            few_shot_positive="good message",
        )
        result = render_few_shots(prop, "assistant")
        assert "ASSISTANT MESSAGE" in result


# --- build_grounding_prompts ---


class TestBuildGroundingPrompts:
    def test_user_role(self):
        prop = Proposition(prop_id="p", role="user", description="asks for help")
        system, user = build_grounding_prompts(
            proposition=prop,
            message_role="user",
            message_text="Can you help me?",
            system_prompt=None,
            user_prompt_template_user=None,
            user_prompt_template_assistant=None,
        )
        assert system == DEFAULT_SYSTEM_PROMPT
        assert "asks for help" in user
        assert "Can you help me?" in user

    def test_assistant_role(self):
        prop = Proposition(prop_id="p", role="assistant", description="provides code")
        system, user = build_grounding_prompts(
            proposition=prop,
            message_role="assistant",
            message_text="Here is the code.",
            system_prompt=None,
            user_prompt_template_user=None,
            user_prompt_template_assistant=None,
        )
        assert system == DEFAULT_SYSTEM_PROMPT
        assert "provides code" in user
        assert "Here is the code." in user
        assert "ASSISTANT MESSAGE" in user

    def test_custom_system_prompt(self):
        prop = Proposition(prop_id="p", role="user", description="desc")
        custom = "You are a custom classifier."
        system, _ = build_grounding_prompts(
            proposition=prop,
            message_role="user",
            message_text="hello",
            system_prompt=custom,
            user_prompt_template_user=None,
            user_prompt_template_assistant=None,
        )
        assert system == custom
