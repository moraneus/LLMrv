"""YAML policy loader."""
from __future__ import annotations

from pathlib import Path

import yaml

from temporalguard.policy import Policy, Proposition


def load_yaml(path: str) -> tuple[list[Proposition], list[Policy]]:
    """Load propositions and policies from a YAML file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with open(file_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")

    propositions = []
    for item in data.get("propositions", []):
        if not isinstance(item, dict):
            raise ValueError(f"Expected proposition mapping, got {type(item).__name__}")
        if "id" not in item:
            raise ValueError("Proposition missing required 'id' field")
        if "role" not in item:
            raise ValueError(f"Proposition '{item['id']}' missing required 'role' field")
        if "description" not in item:
            raise ValueError(f"Proposition '{item['id']}' missing required 'description' field")
        propositions.append(Proposition(
            prop_id=item["id"],
            role=item["role"],
            description=item["description"],
            few_shot_positive=item.get("few_shot_positive", []),
            few_shot_negative=item.get("few_shot_negative", []),
        ))

    policies = []
    for item in data.get("policies", []):
        if not isinstance(item, dict):
            raise ValueError(f"Expected policy mapping, got {type(item).__name__}")
        if "name" not in item:
            raise ValueError("Policy missing required 'name' field")
        if "formula" not in item:
            raise ValueError(f"Policy '{item['name']}' missing required 'formula' field")
        policies.append(Policy(
            name=item["name"],
            formula=item["formula"],
            propositions=item.get("propositions", []),
            enabled=item.get("enabled", True),
        ))

    return propositions, policies
