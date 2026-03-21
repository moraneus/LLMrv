"""Tests for temporalguard.engine.grounding."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from temporalguard.engine.grounding import (
    GroundingResult,
    LLMGrounding,
    _extract_json,
)
from temporalguard.engine.trace import MessageEvent
from temporalguard.policy import Proposition


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        raw = '{"match": true, "reasoning": "ok"}'
        result = _extract_json(raw)
        assert result == {"match": True, "reasoning": "ok"}

    def test_markdown_code_block(self):
        raw = '```json\n{"match": false, "reasoning": "nope"}\n```'
        result = _extract_json(raw)
        assert result == {"match": False, "reasoning": "nope"}

    def test_json_embedded_in_text(self):
        raw = 'Sure! Here is the result:\n{"match": true, "reasoning": "yes"}\nHope that helps!'
        result = _extract_json(raw)
        assert result == {"match": True, "reasoning": "yes"}

    def test_invalid_json_returns_none(self):
        assert _extract_json("this is not json at all") is None

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None

    def test_code_block_without_json_tag(self):
        raw = '```\n{"match": true}\n```'
        result = _extract_json(raw)
        assert result == {"match": True}


# ---------------------------------------------------------------------------
# GroundingResult
# ---------------------------------------------------------------------------


class TestGroundingResult:
    def test_to_dict(self):
        gr = GroundingResult(
            match=True,
            confidence=0.95,
            reasoning="looks good",
            method="llm",
            prop_id="p1",
        )
        d = gr.to_dict()
        assert d == {
            "match": True,
            "confidence": 0.95,
            "reasoning": "looks good",
            "method": "llm",
            "prop_id": "p1",
        }

    def test_to_dict_is_plain_dict(self):
        gr = GroundingResult(
            match=False, confidence=0.0, reasoning="", method="llm", prop_id="x"
        )
        assert isinstance(gr.to_dict(), dict)


# ---------------------------------------------------------------------------
# LLMGrounding — protocol detection
# ---------------------------------------------------------------------------


class TestLLMGroundingProtocol:
    def test_ollama_detected_by_port(self):
        g = LLMGrounding(base_url="http://localhost:11434")
        assert g._is_ollama is True

    def test_ollama_detected_by_api_chat_path(self):
        g = LLMGrounding(base_url="http://myserver:8080/api/chat")
        assert g._is_ollama is True

    def test_openai_detected(self):
        g = LLMGrounding(base_url="https://api.openai.com", model="gpt-4o", api_key="sk-test")
        assert g._is_ollama is False


# ---------------------------------------------------------------------------
# LLMGrounding._parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    @pytest.fixture()
    def grounding(self) -> LLMGrounding:
        return LLMGrounding()

    def test_valid_match_true(self, grounding: LLMGrounding):
        raw = json.dumps({"match": True, "reasoning": "it matches", "confidence": 0.9})
        result = grounding._parse_response(raw, "p1")
        assert result.match is True
        assert result.confidence == 0.9
        assert result.reasoning == "it matches"
        assert result.prop_id == "p1"
        assert result.method == "llm"

    def test_valid_match_false(self, grounding: LLMGrounding):
        raw = json.dumps({"match": False, "reasoning": "no match"})
        result = grounding._parse_response(raw, "p2")
        assert result.match is False
        assert result.confidence == 0.0  # default for False
        assert result.reasoning == "no match"

    def test_missing_confidence_defaults_true(self, grounding: LLMGrounding):
        raw = json.dumps({"match": True, "reasoning": "yes"})
        result = grounding._parse_response(raw, "p1")
        assert result.confidence == 1.0

    def test_missing_confidence_defaults_false(self, grounding: LLMGrounding):
        raw = json.dumps({"match": False, "reasoning": "no"})
        result = grounding._parse_response(raw, "p1")
        assert result.confidence == 0.0

    def test_missing_reasoning(self, grounding: LLMGrounding):
        raw = json.dumps({"match": True})
        result = grounding._parse_response(raw, "p1")
        assert result.reasoning == ""

    def test_non_boolean_match_returns_false(self, grounding: LLMGrounding):
        raw = json.dumps({"match": "yes", "reasoning": "oops"})
        result = grounding._parse_response(raw, "p1")
        assert result.match is False
        assert result.confidence == 0.0
        assert "Non-boolean" in result.reasoning

    def test_unparseable_response(self, grounding: LLMGrounding):
        result = grounding._parse_response("not json at all", "p1")
        assert result.match is False
        assert result.confidence == 0.0
        assert "Could not parse" in result.reasoning

    def test_match_as_integer_is_non_boolean(self, grounding: LLMGrounding):
        raw = json.dumps({"match": 1, "reasoning": "truthy int"})
        result = grounding._parse_response(raw, "p1")
        assert result.match is False
        assert "Non-boolean" in result.reasoning


# ---------------------------------------------------------------------------
# LLMGrounding.evaluate — fail-open behaviour
# ---------------------------------------------------------------------------


class TestEvaluateFailOpen:
    def test_exception_returns_fail_open(self):
        grounding = LLMGrounding(base_url="http://localhost:11434")
        msg = MessageEvent(role="user", text="hello", index=0)
        prop = Proposition(prop_id="p1", role="user", description="greeting")

        # _call_llm will fail because there's no server running
        with patch.object(grounding, "_call_llm", side_effect=RuntimeError("boom")):
            result = grounding.evaluate(msg, prop)

        assert result.match is False
        assert result.confidence == 0.0
        assert "Grounding failed" in result.reasoning
        assert "boom" in result.reasoning
        assert result.method == "llm"
        assert result.prop_id == "p1"


# ---------------------------------------------------------------------------
# LLMGrounding._call_llm — mocked HTTP
# ---------------------------------------------------------------------------


class TestCallLlm:
    def test_ollama_call(self):
        grounding = LLMGrounding(base_url="http://localhost:11434", model="mistral")
        ollama_response = {
            "message": {"content": '{"match": true, "reasoning": "ok"}'}
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = ollama_response
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = grounding._call_llm("system", "user prompt")

        assert result == '{"match": true, "reasoning": "ok"}'
        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["stream"] is False
        assert payload["model"] == "mistral"

    def test_openai_call_with_api_key(self):
        grounding = LLMGrounding(
            base_url="https://api.openai.com", model="gpt-4o", api_key="sk-test"
        )
        openai_response = {
            "choices": [{"message": {"content": '{"match": false}'}}]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = openai_response
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = grounding._call_llm("system", "user prompt")

        assert result == '{"match": false}'
        call_args = mock_client.post.call_args
        assert "/v1/chat/completions" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"
