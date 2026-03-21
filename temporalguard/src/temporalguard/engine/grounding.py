"""Grounding engine: evaluate whether a message matches a proposition.

Provides an abstract ``GroundingMethod`` base class and a concrete
``LLMGrounding`` implementation that calls OpenAI-compatible or Ollama
servers over HTTP.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import httpx

from temporalguard.engine.trace import MessageEvent
from temporalguard.grounding.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_USER_PROMPT_TEMPLATE_USER,
    build_grounding_prompts,
)
from temporalguard.policy import Proposition


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GroundingResult:
    """Outcome of evaluating a single proposition against a message."""

    match: bool
    confidence: float
    reasoning: str
    method: str
    prop_id: str

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class GroundingMethod(ABC):
    """Interface for grounding strategies."""

    @abstractmethod
    def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        """Evaluate whether *message* satisfies *proposition*."""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Try to pull a JSON object out of *text*.

    Handles plain JSON, markdown fenced code-blocks, and JSON embedded in
    surrounding prose.  Returns ``None`` when no valid object can be found.
    """
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


# ---------------------------------------------------------------------------
# LLM-based grounding
# ---------------------------------------------------------------------------


class LLMGrounding(GroundingMethod):
    """Grounding via an LLM (OpenAI-compatible or Ollama native API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        api_key: str = "",
        system_prompt: str = "",
        user_prompt_template: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or ""
        self.user_prompt_template = user_prompt_template or ""

        # Protocol detection
        self._is_ollama = "11434" in self.base_url or "/api/chat" in self.base_url

    # -- public interface ---------------------------------------------------

    def evaluate(self, message: MessageEvent, proposition: Proposition) -> GroundingResult:
        """Ground *proposition* against *message* via an LLM call."""
        try:
            system_prompt, user_prompt = build_grounding_prompts(
                proposition=proposition,
                message_role=message.role,
                message_text=message.text,
                system_prompt=self.system_prompt or None,
                user_prompt_template_user=(
                    self.user_prompt_template
                    if self.user_prompt_template and proposition.role != "assistant"
                    else None
                ),
                user_prompt_template_assistant=(
                    self.user_prompt_template
                    if self.user_prompt_template and proposition.role == "assistant"
                    else None
                ),
            )
            raw = self._call_llm(system_prompt, user_prompt)
            return self._parse_response(raw, proposition.prop_id)
        except Exception as e:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"Grounding failed: {e}",
                method="llm",
                prop_id=proposition.prop_id,
            )

    # -- LLM transport ------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to the configured LLM and return the raw text reply."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self._is_ollama:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            }
            headers: dict[str, str] = {}
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0,
            }
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if self._is_ollama:
            return data.get("message", {}).get("content", "")
        return data["choices"][0]["message"]["content"]

    # -- response parsing ---------------------------------------------------

    def _parse_response(self, raw: str, prop_id: str) -> GroundingResult:
        """Parse the raw LLM reply into a ``GroundingResult``."""
        parsed = _extract_json(raw)
        if parsed is None:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"Could not parse LLM response: {raw!r}",
                method="llm",
                prop_id=prop_id,
            )

        match_val = parsed.get("match")
        if not isinstance(match_val, bool):
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"Non-boolean match field: {match_val!r}",
                method="llm",
                prop_id=prop_id,
            )

        confidence = parsed.get("confidence")
        if confidence is None:
            confidence = 1.0 if match_val else 0.0

        reasoning = parsed.get("reasoning", "")

        return GroundingResult(
            match=match_val,
            confidence=float(confidence),
            reasoning=reasoning,
            method="llm",
            prop_id=prop_id,
        )
