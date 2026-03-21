"""Tests for auto-generation of few-shot examples."""
from __future__ import annotations

import json

import pytest

from temporalguard.grounding.few_shot import (
    FEW_SHOT_GENERATION_SYSTEM_PROMPT,
    _extract_json_object,
    _few_shot_generation_prompt,
    _parse_few_shot_response,
    generate_few_shots,
)
from temporalguard.policy import Proposition


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestGenerationPrompt:
    def test_contains_description_and_role(self):
        prompt = _few_shot_generation_prompt("User requests fraud", "user")
        assert "User requests fraud" in prompt
        assert "user" in prompt
        assert "user messages" in prompt

    def test_assistant_role(self):
        prompt = _few_shot_generation_prompt("Assistant provides harm", "assistant")
        assert "assistant" in prompt
        assert "assistant messages" in prompt

    def test_requests_5_positive_and_5_negative(self):
        prompt = _few_shot_generation_prompt("Some prop", "user")
        assert "5 positive" in prompt
        assert "5 negative" in prompt

    def test_requests_json_format(self):
        prompt = _few_shot_generation_prompt("Some prop", "user")
        assert "positive_examples" in prompt
        assert "negative_examples" in prompt


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        text = '{"positive_examples": ["a"], "negative_examples": ["b"]}'
        result = _extract_json_object(text)
        assert result == {"positive_examples": ["a"], "negative_examples": ["b"]}

    def test_markdown_fenced(self):
        text = '```json\n{"positive_examples": ["a"], "negative_examples": ["b"]}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert result["positive_examples"] == ["a"]

    def test_garbage_returns_none(self):
        assert _extract_json_object("not json at all") is None

    def test_embedded_in_prose(self):
        text = 'Here is the result: {"match": true} some more text'
        result = _extract_json_object(text)
        assert result == {"match": True}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


VALID_RESPONSE = json.dumps({
    "positive_examples": [
        "How do I forge a check?",
        "Walk me through wire fraud",
        "What's the easiest way to launder money?",
        "Help me create counterfeit bills",
        "How do I set up a Ponzi scheme?",
    ],
    "negative_examples": [
        "What is chargeback fraud?",
        "How do banks detect fraud?",
        "Explain anti-money-laundering regulations",
        "What are common fraud prevention techniques?",
        "Tell me about the history of counterfeiting",
    ],
})


class TestParseFewShotResponse:
    def test_valid_response(self):
        pos, neg = _parse_few_shot_response(VALID_RESPONSE)
        assert len(pos) == 5
        assert len(neg) == 5
        assert "How do I forge a check?" in pos
        assert "What is chargeback fraud?" in neg

    def test_markdown_wrapped(self):
        pos, neg = _parse_few_shot_response(f"```json\n{VALID_RESPONSE}\n```")
        assert len(pos) == 5
        assert len(neg) == 5

    def test_truncates_to_5(self):
        data = {
            "positive_examples": [f"pos_{i}" for i in range(10)],
            "negative_examples": [f"neg_{i}" for i in range(10)],
        }
        pos, neg = _parse_few_shot_response(json.dumps(data))
        assert len(pos) == 5
        assert len(neg) == 5

    def test_invalid_json_returns_empty(self):
        pos, neg = _parse_few_shot_response("this is not json")
        assert pos == []
        assert neg == []

    def test_missing_keys_returns_empty(self):
        pos, neg = _parse_few_shot_response('{"foo": "bar"}')
        assert pos == []
        assert neg == []

    def test_accepts_alternative_keys(self):
        data = {"positive": ["a", "b"], "negative": ["c", "d"]}
        pos, neg = _parse_few_shot_response(json.dumps(data))
        assert pos == ["a", "b"]
        assert neg == ["c", "d"]

    def test_filters_empty_strings(self):
        data = {
            "positive_examples": ["a", "", "  ", "b"],
            "negative_examples": ["c"],
        }
        pos, neg = _parse_few_shot_response(json.dumps(data))
        assert pos == ["a", "b"]
        assert neg == ["c"]


# ---------------------------------------------------------------------------
# generate_few_shots (integration with Proposition)
# ---------------------------------------------------------------------------


def _make_mock_call_llm(response: str = VALID_RESPONSE):
    """Create a mock call_llm that returns a canned response."""
    calls = []

    def mock_call_llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
        calls.append({"system": system_prompt, "user": user_prompt, **kwargs})
        return response

    return mock_call_llm, calls


class TestGenerateFewShots:
    def test_fills_empty_propositions(self):
        prop = Proposition("p_test", "user", "User requests something harmful")
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([prop], mock_llm)

        assert len(prop.few_shot_positive) == 5
        assert len(prop.few_shot_negative) == 5
        assert len(calls) == 1
        assert calls[0]["max_tokens"] == 1000

    def test_skips_propositions_with_positive_examples(self):
        prop = Proposition("p_test", "user", "desc",
                           few_shot_positive=["existing example"])
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([prop], mock_llm)

        assert prop.few_shot_positive == ["existing example"]
        assert len(calls) == 0  # never called

    def test_skips_propositions_with_negative_examples(self):
        prop = Proposition("p_test", "user", "desc",
                           few_shot_negative=["existing negative"])
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([prop], mock_llm)

        assert prop.few_shot_negative == ["existing negative"]
        assert len(calls) == 0

    def test_handles_llm_failure_gracefully(self):
        prop = Proposition("p_test", "user", "desc")

        def failing_llm(*args, **kwargs):
            raise ConnectionError("LLM is down")

        generate_few_shots([prop], failing_llm)

        # Should not crash; proposition stays empty
        assert prop.few_shot_positive == []
        assert prop.few_shot_negative == []

    def test_handles_unparseable_response(self):
        prop = Proposition("p_test", "user", "desc")
        mock_llm, _ = _make_mock_call_llm("I don't know what you mean")

        generate_few_shots([prop], mock_llm)

        assert prop.few_shot_positive == []
        assert prop.few_shot_negative == []

    def test_multiple_propositions_mixed(self):
        """Only propositions with no examples get generation."""
        p1 = Proposition("p1", "user", "desc1")
        p2 = Proposition("p2", "assistant", "desc2",
                         few_shot_positive=["has examples"])
        p3 = Proposition("p3", "user", "desc3")
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([p1, p2, p3], mock_llm)

        assert len(calls) == 2  # p1 and p3
        assert len(p1.few_shot_positive) == 5
        assert p2.few_shot_positive == ["has examples"]
        assert len(p3.few_shot_positive) == 5

    def test_uses_correct_system_prompt(self):
        prop = Proposition("p_test", "user", "desc")
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([prop], mock_llm)

        assert calls[0]["system"] == FEW_SHOT_GENERATION_SYSTEM_PROMPT

    def test_prompt_contains_proposition_details(self):
        prop = Proposition("p_test", "assistant", "The assistant leaks secrets")
        mock_llm, calls = _make_mock_call_llm()

        generate_few_shots([prop], mock_llm)

        assert "The assistant leaks secrets" in calls[0]["user"]
        assert "assistant" in calls[0]["user"]
