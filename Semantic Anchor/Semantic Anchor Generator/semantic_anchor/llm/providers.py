"""LLM provider functions for various APIs."""
import json
import os
import sys

# Terminal color constants
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Shared flags dict — used by api.py to communicate feature-skip signals
# without circular imports. api.py sets these; provider functions read them.
_flags = {}


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
    if not _flags.get('skip_temperature', False):
        api_params["temperature"] = 0.95
    if not _flags.get('skip_response_format', False):
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
    if not _flags.get('skip_temperature', False):
        api_params["temperature"] = 0.95
    response = client.chat.completions.create(**api_params)
    msg = response.choices[0].message
    finish = response.choices[0].finish_reason
    raw = msg.content if msg.content else ""
    return raw.strip(), finish, None


def _call_openrouter(api_key, model, system_prompt, user_prompt, attempt):
    """OpenRouter API call (OpenAI-compatible, routes to 200+ models)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    api_params = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=16384,
    )
    if not _flags.get('skip_temperature', False):
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
                    chunk = json.loads(line)
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
PROVIDER_FUNCTIONS["openrouter"] = _call_openrouter
PROVIDER_FUNCTIONS["ollama"] = _call_ollama
PROVIDER_FUNCTIONS["local"] = _call_local_openai_compat
PROVIDER_FUNCTIONS["lmstudio"] = _call_local_openai_compat
PROVIDER_FUNCTIONS["vllm"] = _call_local_openai_compat
