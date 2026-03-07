"""High-level LLM call wrappers with retry logic and JSON extraction."""
import json
import re
import sys

from .providers import PROVIDER_FUNCTIONS, _flags


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
        supported = "openai, anthropic/claude, gemini/google, grok/xai, openrouter, ollama, lmstudio, vllm"
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
                _flags['skip_temperature'] = True
                if attempt < max_retries:
                    print("    NOTE: Model doesn't support custom temperature, retrying...")
                    continue
            # Detect response_format not supported
            if "response_format" in err_str and ("unsupported" in err_str.lower() or
                                                  "not supported" in err_str.lower() or
                                                  "invalid" in err_str.lower()):
                _flags['skip_response_format'] = True
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


def call_llm_raw(config, system_prompt, user_prompt, max_retries=3):
    """Call LLM and return raw text response (no JSON parsing).

    Unlike call_llm(), this returns the raw string directly.
    Used for synthetic data generation where output is plain text, not JSON.

    Returns: str (raw text) or "" on failure/refusal.
    """
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
        supported = "openai, anthropic/claude, gemini/google, grok/xai, openrouter, ollama, lmstudio, vllm"
        print("\n  ERROR: Unknown provider '{}'. Supported: {}".format(
            provider, supported))
        sys.exit(1)

    for attempt in range(1, max_retries + 1):
        try:
            raw, finish, refusal = call_fn(api_key, model, system_prompt, user_prompt, attempt)

            if refusal:
                print("    WARNING: Model refused (attempt {}/{}): {}".format(
                    attempt, max_retries, str(refusal)[:200]))
                if attempt >= max_retries:
                    print("    The model is refusing this content. Try a different model "
                          "or lower adversarial_depth in config.ini [training].")
                    return ""
                continue

            if finish == "content_filter":
                print("    WARNING: Content filtered (attempt {}/{}).".format(
                    attempt, max_retries))
                if attempt >= max_retries:
                    return ""
                continue

            if not raw:
                if attempt < max_retries:
                    print("    WARNING: Empty response (attempt {}/{}), retrying...".format(
                        attempt, max_retries))
                    continue
                return ""

            # Return raw text directly — no JSON parsing
            if isinstance(raw, str):
                return raw
            return str(raw)

        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "authentication" in err_str.lower():
                print("    ERROR: Authentication failed. Check your API key.")
                return ""
            if attempt < max_retries:
                print("    WARNING: API error (attempt {}/{}): {}, retrying...".format(
                    attempt, max_retries, e))
            else:
                print("    ERROR: API call failed after {} attempts: {}".format(
                    max_retries, e))
                return ""
    return ""
