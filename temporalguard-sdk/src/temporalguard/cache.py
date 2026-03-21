"""Persistent YAML cache for propositions with generated few-shot examples.

After few-shot examples are auto-generated, they are saved to a YAML file so
subsequent runs can reuse them without hitting the LLM again.  The cache is
keyed by a fingerprint derived from proposition IDs, descriptions, roles, and
policy formulas — if any of these change, the cache is invalidated.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import yaml

from temporalguard.policy import Policy, Proposition

logger = logging.getLogger(__name__)

# Default cache directory (relative to cwd)
DEFAULT_CACHE_DIR = ".temporalguard"


def _fingerprint(propositions: list[Proposition], policies: list[Policy]) -> str:
    """Compute a short hash of proposition + policy definitions.

    Any change to IDs, descriptions, roles, or formulas invalidates the cache.
    Few-shot examples are deliberately excluded so they don't trigger
    re-generation when only the cached examples differ.
    """
    parts: list[str] = []
    for p in sorted(propositions, key=lambda x: x.prop_id):
        parts.append(f"prop:{p.prop_id}:{p.role}:{p.description}")
    for pol in sorted(policies, key=lambda x: x.name):
        parts.append(f"policy:{pol.name}:{pol.formula}")
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return digest


def _propositions_to_yaml(propositions: list[Proposition]) -> list[dict]:
    """Serialise propositions to YAML-ready dicts."""
    items = []
    for p in propositions:
        d: dict = {
            "id": p.prop_id,
            "role": p.role,
            "description": p.description,
        }
        if p.few_shot_positive:
            d["few_shot_positive"] = list(p.few_shot_positive)
        if p.few_shot_negative:
            d["few_shot_negative"] = list(p.few_shot_negative)
        items.append(d)
    return items


def _policies_to_yaml(policies: list[Policy]) -> list[dict]:
    """Serialise policies to YAML-ready dicts."""
    items = []
    for pol in policies:
        d: dict = {"name": pol.name, "formula": pol.formula}
        if not pol.enabled:
            d["enabled"] = False
        items.append(d)
    return items


def save_cache(
    propositions: list[Proposition],
    policies: list[Policy],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Write propositions and policies (with few-shot examples) to a cached YAML.

    Returns the path to the written file.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fp = _fingerprint(propositions, policies)
    file_path = cache_path / f"{fp}.yaml"

    data = {
        "propositions": _propositions_to_yaml(propositions),
        "policies": _policies_to_yaml(policies),
    }

    file_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Saved few-shot cache to %s", file_path)
    return file_path


def load_cache(
    propositions: list[Proposition],
    policies: list[Policy],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> bool:
    """Try to load cached few-shot examples into propositions **in-place**.

    Returns ``True`` if a valid cache was found and applied, ``False`` otherwise.
    """
    cache_path = Path(cache_dir)
    fp = _fingerprint(propositions, policies)
    file_path = cache_path / f"{fp}.yaml"

    if not file_path.exists():
        return False

    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return False

        # Build lookup: prop_id → cached few-shot examples
        cached: dict[str, dict] = {}
        for item in data.get("propositions", []):
            if isinstance(item, dict) and "id" in item:
                cached[item["id"]] = item

        applied = 0
        for prop in propositions:
            if prop.few_shot_positive or prop.few_shot_negative:
                continue  # user-provided, skip
            entry = cached.get(prop.prop_id)
            if entry:
                pos = entry.get("few_shot_positive", [])
                neg = entry.get("few_shot_negative", [])
                if pos or neg:
                    prop.few_shot_positive = list(pos)
                    prop.few_shot_negative = list(neg)
                    applied += 1

        if applied > 0:
            logger.info(
                "Loaded cached few-shot examples for %d proposition(s) from %s",
                applied, file_path,
            )
            return True

        return False

    except Exception:
        logger.warning("Failed to load few-shot cache from %s", file_path, exc_info=True)
        return False
