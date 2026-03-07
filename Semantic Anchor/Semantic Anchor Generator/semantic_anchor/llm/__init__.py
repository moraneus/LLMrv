"""LLM integration subpackage."""
from .api import call_llm, call_llm_raw
from .prompts import (
    _role_context, _system_prompt_round1, _system_prompt_diversity,
    build_round1_prompt, build_diversity_prompt,
)
