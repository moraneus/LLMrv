"""Tests for the YAML policy loader."""
from __future__ import annotations

import pytest

from temporalguard.loader import load_yaml


SAMPLE_YAML = """\
propositions:
  - id: consent_given
    role: user
    description: "User has given explicit consent"
    few_shot_positive: "Yes, I consent to sharing my data."
    few_shot_negative: "I don't want to share anything."
  - id: data_accessed
    role: assistant
    description: "Assistant accesses user data"

policies:
  - name: consent_before_access
    formula: "data_accessed -> O(consent_given)"
    propositions:
      - consent_given
      - data_accessed
"""

MINIMAL_YAML = """\
propositions:
  - id: greeted
    role: user
    description: "User greeted the assistant"

policies:
  - name: greeting_policy
    formula: "greeted"
"""

DISABLED_POLICY_YAML = """\
propositions:
  - id: action_done
    role: assistant
    description: "Action was performed"

policies:
  - name: disabled_one
    formula: "action_done"
    enabled: false
  - name: enabled_one
    formula: "action_done"
"""


@pytest.fixture()
def sample_yaml_file(tmp_path):
    p = tmp_path / "sample.yaml"
    p.write_text(SAMPLE_YAML)
    return p


@pytest.fixture()
def minimal_yaml_file(tmp_path):
    p = tmp_path / "minimal.yaml"
    p.write_text(MINIMAL_YAML)
    return p


@pytest.fixture()
def disabled_policy_file(tmp_path):
    p = tmp_path / "disabled.yaml"
    p.write_text(DISABLED_POLICY_YAML)
    return p


class TestLoadPropositions:
    def test_count(self, sample_yaml_file):
        props, _ = load_yaml(str(sample_yaml_file))
        assert len(props) == 2

    def test_fields(self, sample_yaml_file):
        props, _ = load_yaml(str(sample_yaml_file))
        p = props[0]
        assert p.prop_id == "consent_given"
        assert p.role == "user"
        assert p.description == "User has given explicit consent"
        assert p.few_shot_positive == "Yes, I consent to sharing my data."
        assert p.few_shot_negative == "I don't want to share anything."

    def test_few_shot_defaults_to_none(self, sample_yaml_file):
        props, _ = load_yaml(str(sample_yaml_file))
        p = props[1]  # data_accessed has no few-shots
        assert p.few_shot_positive is None
        assert p.few_shot_negative is None

    def test_minimal(self, minimal_yaml_file):
        props, _ = load_yaml(str(minimal_yaml_file))
        assert len(props) == 1
        assert props[0].prop_id == "greeted"


class TestLoadPolicies:
    def test_count(self, sample_yaml_file):
        _, policies = load_yaml(str(sample_yaml_file))
        assert len(policies) == 1

    def test_fields(self, sample_yaml_file):
        _, policies = load_yaml(str(sample_yaml_file))
        pol = policies[0]
        assert pol.name == "consent_before_access"
        assert pol.formula == "data_accessed -> O(consent_given)"
        assert pol.propositions == ["consent_given", "data_accessed"]
        assert pol.enabled is True

    def test_auto_extract_propositions(self, minimal_yaml_file):
        """When propositions list is omitted, Policy.__post_init__ extracts them."""
        _, policies = load_yaml(str(minimal_yaml_file))
        assert policies[0].propositions == ["greeted"]

    def test_disabled_policy(self, disabled_policy_file):
        _, policies = load_yaml(str(disabled_policy_file))
        assert policies[0].name == "disabled_one"
        assert policies[0].enabled is False
        assert policies[1].name == "enabled_one"
        assert policies[1].enabled is True


class TestErrors:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/path/to/file.yaml")

    def test_invalid_yaml_not_mapping(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list\n- not a mapping\n")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            load_yaml(str(p))

    def test_missing_proposition_id(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("propositions:\n  - role: user\n    description: oops\npolicies: []\n")
        with pytest.raises(ValueError, match="missing required 'id' field"):
            load_yaml(str(p))

    def test_missing_proposition_role(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("propositions:\n  - id: foo\n    description: oops\npolicies: []\n")
        with pytest.raises(ValueError, match="missing required 'role' field"):
            load_yaml(str(p))

    def test_missing_proposition_description(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("propositions:\n  - id: foo\n    role: user\npolicies: []\n")
        with pytest.raises(ValueError, match="missing required 'description' field"):
            load_yaml(str(p))

    def test_missing_policy_name(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("propositions: []\npolicies:\n  - formula: 'x'\n")
        with pytest.raises(ValueError, match="missing required 'name' field"):
            load_yaml(str(p))

    def test_missing_policy_formula(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("propositions: []\npolicies:\n  - name: test\n")
        with pytest.raises(ValueError, match="missing required 'formula' field"):
            load_yaml(str(p))
