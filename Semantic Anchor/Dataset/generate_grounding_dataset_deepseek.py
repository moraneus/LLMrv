#!/usr/bin/env python3
"""
Generate grounding_dataset_deepseek.csv from existing propositions.

For each proposition in grounding_dataset.csv, this script asks DeepSeek
through OpenRouter to generate:
- 5 positive prompts (proposition holds)
- 5 negative prompts (proposition does not hold but same domain terms)

Output schema matches the original dataset:
proposition_id,role,proposition,example_type,example_id,text
"""

import argparse
import configparser
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from urllib import error as urlerror
from urllib import request as urlrequest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_COLUMNS = {
    "proposition_id",
    "role",
    "proposition",
    "example_type",
    "example_id",
    "text",
}


def normalize_header(name):
    return (name or "").replace("\ufeff", "").strip().lower()


def proposition_sort_key(pid):
    m = re.fullmatch(r"p(\d+)", (pid or "").strip().lower())
    if m:
        return (0, int(m.group(1)))
    return (1, pid or "")


def parse_json_from_response(raw):
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model output.")


def load_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Dataset has no header row.")
        reader.fieldnames = [normalize_header(h) for h in reader.fieldnames]
        missing = REQUIRED_COLUMNS.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(
                "Dataset is missing required columns: {}".format(", ".join(sorted(missing)))
            )
        for row in reader:
            normalized = {normalize_header(k): (v or "").strip() for k, v in row.items()}
            rows.append(normalized)
    if not rows:
        raise ValueError("Dataset has no rows.")
    return rows


def extract_propositions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["proposition_id"]].append(row)

    result = []
    for proposition_id in sorted(grouped.keys(), key=proposition_sort_key):
        bucket = grouped[proposition_id]
        roles = {r["role"] for r in bucket}
        propositions = {r["proposition"] for r in bucket}
        if len(roles) != 1:
            raise ValueError(
                "proposition_id {} has multiple roles: {}".format(
                    proposition_id, ", ".join(sorted(roles))
                )
            )
        if len(propositions) != 1:
            raise ValueError(
                "proposition_id {} has multiple propositions.".format(proposition_id)
            )
        result.append(
            {
                "proposition_id": proposition_id,
                "role": next(iter(roles)),
                "proposition": next(iter(propositions)),
            }
        )
    return result


def sanitize_text(text):
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t


def dedupe_and_take(items, n):
    seen = set()
    out = []
    for item in items:
        clean = sanitize_text(item)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) == n:
            break
    return out


def load_openrouter_config(config_path):
    cfg = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise ValueError("Config file not found: {}".format(config_path))
    cfg.read(config_path, encoding="utf-8")

    if not cfg.has_section("openrouter"):
        raise ValueError("Config is missing [openrouter] section: {}".format(config_path))

    api_key = cfg.get("openrouter", "api_key", fallback="").strip()
    model = cfg.get("openrouter", "model", fallback="deepseek/deepseek-chat-v3.2").strip()
    base_url = cfg.get("openrouter", "base_url", fallback="https://openrouter.ai/api/v1").strip()
    site_url = cfg.get("openrouter", "site_url", fallback="").strip()
    site_name = cfg.get("openrouter", "site_name", fallback="").strip()
    temperature = float(cfg.get("openrouter", "temperature", fallback="0.7"))

    if api_key.startswith("or-"):
        api_key = "sk-" + api_key

    if (
        not api_key
        or api_key.lower().startswith("or-your-")
        or api_key.lower().startswith("sk-or-your-")
    ):
        raise ValueError(
            "OpenRouter API key is missing. Set [openrouter].api_key in {}".format(config_path)
        )

    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "site_url": site_url,
        "site_name": site_name,
        "temperature": temperature,
    }


def build_prompt(proposition, role):
    speaker_line = (
        "The prompts must look like messages a USER would write."
        if role.lower() == "user"
        else "The prompts must look like ASSISTANT responses to a user."
    )
    return (
        "Proposition:\n"
        "{}\n\n"
        "Role context:\n"
        "{}\n\n"
        "Instruction:\n"
        "please generate 5 prompts that the proposition holds for them, and 5 other prompts "
        "that the proposition does not hold for them, but they use some terms that are related "
        "to the proposition domain. it should be clear from the negative examples that they do not hold for the proposition." 
        "the prompts should be 1-2 sentences long.\n\n"
        "Additional constraints:\n"
        "- No placeholders like [REDACTED], <NAME>, <...>, or template tokens.\n"
        "- No numbering, bullets, special characters or markdown.\n"
        "- Avoid duplicates.\n"
        "- Keep language realistic and natural.\n"
        "Return ONLY valid JSON in this exact schema:\n"
        '{{\n'
        '  "positive": ["...","...","...","...","..."],\n'
        '  "negative": ["...","...","...","...","..."]\n'
        "}}"
    ).format(proposition, speaker_line)


def openrouter_chat_completion(api_cfg, messages, temperature, max_tokens, force_json=False):
    url = api_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer {}".format(api_cfg["api_key"]),
        "Content-Type": "application/json",
    }
    if api_cfg["site_url"]:
        headers["HTTP-Referer"] = api_cfg["site_url"]
    if api_cfg["site_name"]:
        headers["X-Title"] = api_cfg["site_name"]

    payload = {
        "model": api_cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    req = urlrequest.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {}: {}".format(e.code, body[:500]))
    except urlerror.URLError as e:
        raise RuntimeError("Network error: {}".format(e.reason))

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices in OpenRouter response.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Missing text content in OpenRouter response.")
    return content.strip()


def request_examples(api_cfg, proposition, role, temperature, max_retries):
    system_prompt = (
        "You generate benchmark prompts for semantic proposition evaluation. "
        "Return strict JSON only."
    )
    user_prompt = build_prompt(proposition, role)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                content = openrouter_chat_completion(
                    api_cfg=api_cfg,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1400,
                    force_json=True,
                )
            except Exception:
                content = openrouter_chat_completion(
                    api_cfg=api_cfg,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1400,
                    force_json=False,
                )

            data = parse_json_from_response(content)
            positive = dedupe_and_take(data.get("positive", []), 5)
            negative = dedupe_and_take(data.get("negative", []), 5)
            if len(positive) == 5 and len(negative) == 5:
                return positive, negative
            raise ValueError(
                "Model returned insufficient examples (positive={}, negative={}).".format(
                    len(positive), len(negative)
                )
            )
        except Exception as exc:
            last_error = exc
            print("    Attempt {}/{} failed: {}".format(attempt, max_retries, exc))

    raise RuntimeError("Failed after {} attempts: {}".format(max_retries, last_error))


def build_rows(propositions, generated):
    rows = []
    for item in propositions:
        pid = item["proposition_id"]
        role = item["role"]
        proposition = item["proposition"]
        positive = generated[pid]["positive"]
        negative = generated[pid]["negative"]

        for i, text in enumerate(positive, 1):
            rows.append(
                {
                    "proposition_id": pid,
                    "role": role,
                    "proposition": proposition,
                    "example_type": "positive",
                    "example_id": "{}_pos_{}".format(pid, i),
                    "text": text,
                }
            )
        for i, text in enumerate(negative, 1):
            rows.append(
                {
                    "proposition_id": pid,
                    "role": role,
                    "proposition": proposition,
                    "example_type": "negative",
                    "example_id": "{}_neg_{}".format(pid, i),
                    "text": text,
                }
            )
    return rows


def write_dataset(path, rows):
    fields = [
        "proposition_id",
        "role",
        "proposition",
        "example_type",
        "example_id",
        "text",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate grounding_dataset_deepseek.csv using OpenRouter DeepSeek."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(SCRIPT_DIR, "grounding_dataset.csv"),
        help="Input dataset used only to read proposition_id/role/proposition.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "grounding_dataset_deepseek.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "config.ini"),
        help="Path to config.ini with [openrouter] settings.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per proposition when output is invalid.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.4,
        help="Sleep between propositions to reduce burst traffic.",
    )
    args = parser.parse_args()

    rows = load_dataset(args.input)
    propositions = extract_propositions(rows)
    cfg = load_openrouter_config(args.config)

    print("Generating new dataset with model: {}".format(cfg["model"]))
    print("Propositions: {}".format(len(propositions)))

    generated = {}
    for idx, item in enumerate(propositions, 1):
        pid = item["proposition_id"]
        role = item["role"]
        proposition = item["proposition"]
        print("[{}/{}] {} ({})".format(idx, len(propositions), pid, role))
        positive, negative = request_examples(
            api_cfg=cfg,
            proposition=proposition,
            role=role,
            temperature=cfg["temperature"],
            max_retries=args.max_retries,
        )
        generated[pid] = {"positive": positive, "negative": negative}
        if idx < len(propositions):
            time.sleep(max(0.0, args.sleep_seconds))

    out_rows = build_rows(propositions, generated)
    write_dataset(args.output, out_rows)
    print("Wrote {} rows to {}".format(len(out_rows), args.output))


if __name__ == "__main__":
    main()
