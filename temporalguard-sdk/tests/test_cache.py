"""Tests for the persistent YAML cache."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from temporalguard.cache import (
    _fingerprint,
    _policies_to_yaml,
    _propositions_to_yaml,
    load_cache,
    save_cache,
)
from temporalguard.policy import Policy, Proposition


@pytest.fixture
def tmp_cache(tmp_path):
    """Return a temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def sample_props():
    return [
        Proposition("p1", "user", "User requests harm",
                    few_shot_positive=["Do harm"], few_shot_negative=["What is harm?"]),
        Proposition("p2", "assistant", "Assistant complies"),
    ]


@pytest.fixture
def sample_policies():
    return [
        Policy(name="Safety", formula="H(P(p1) -> !p2)"),
    ]


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_deterministic(self, sample_props, sample_policies):
        fp1 = _fingerprint(sample_props, sample_policies)
        fp2 = _fingerprint(sample_props, sample_policies)
        assert fp1 == fp2

    def test_changes_with_description(self, sample_policies):
        props_a = [Proposition("p1", "user", "Description A")]
        props_b = [Proposition("p1", "user", "Description B")]
        assert _fingerprint(props_a, sample_policies) != _fingerprint(props_b, sample_policies)

    def test_changes_with_role(self, sample_policies):
        props_a = [Proposition("p1", "user", "desc")]
        props_b = [Proposition("p1", "assistant", "desc")]
        assert _fingerprint(props_a, sample_policies) != _fingerprint(props_b, sample_policies)

    def test_changes_with_formula(self, sample_props):
        pol_a = [Policy(name="P", formula="H(p1)")]
        pol_b = [Policy(name="P", formula="P(p1)")]
        assert _fingerprint(sample_props, pol_a) != _fingerprint(sample_props, pol_b)

    def test_order_independent(self, sample_policies):
        p1 = Proposition("p1", "user", "desc1")
        p2 = Proposition("p2", "user", "desc2")
        assert _fingerprint([p1, p2], sample_policies) == _fingerprint([p2, p1], sample_policies)

    def test_ignores_few_shot_content(self, sample_policies):
        """Few-shot examples should NOT affect the fingerprint."""
        props_a = [Proposition("p1", "user", "desc", few_shot_positive=["a"])]
        props_b = [Proposition("p1", "user", "desc", few_shot_positive=["b"])]
        assert _fingerprint(props_a, sample_policies) == _fingerprint(props_b, sample_policies)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_propositions_to_yaml(self):
        props = [
            Proposition("p1", "user", "desc",
                        few_shot_positive=["pos1"], few_shot_negative=["neg1"]),
            Proposition("p2", "assistant", "desc2"),
        ]
        result = _propositions_to_yaml(props)
        assert len(result) == 2
        assert result[0]["id"] == "p1"
        assert result[0]["few_shot_positive"] == ["pos1"]
        assert result[0]["few_shot_negative"] == ["neg1"]
        # p2 has no few-shot examples, so keys should be absent
        assert "few_shot_positive" not in result[1]
        assert "few_shot_negative" not in result[1]

    def test_policies_to_yaml(self):
        policies = [
            Policy(name="P1", formula="H(p1)"),
            Policy(name="P2", formula="P(p2)", enabled=False),
        ]
        result = _policies_to_yaml(policies)
        assert len(result) == 2
        assert result[0] == {"name": "P1", "formula": "H(p1)"}
        assert result[1] == {"name": "P2", "formula": "P(p2)", "enabled": False}


# ---------------------------------------------------------------------------
# Save + load round-trip
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_save_creates_file(self, sample_props, sample_policies, tmp_cache):
        path = save_cache(sample_props, sample_policies, tmp_cache)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_saved_file_is_valid_yaml(self, sample_props, sample_policies, tmp_cache):
        path = save_cache(sample_props, sample_policies, tmp_cache)
        data = yaml.safe_load(path.read_text())
        assert "propositions" in data
        assert "policies" in data
        assert len(data["propositions"]) == 2
        assert len(data["policies"]) == 1

    def test_saved_file_contains_few_shots(self, sample_props, sample_policies, tmp_cache):
        path = save_cache(sample_props, sample_policies, tmp_cache)
        data = yaml.safe_load(path.read_text())
        p1 = data["propositions"][0]
        assert p1["few_shot_positive"] == ["Do harm"]
        assert p1["few_shot_negative"] == ["What is harm?"]

    def test_load_fills_empty_propositions(self, sample_policies, tmp_cache):
        """Save with few-shots, then load into propositions that lack them."""
        props_with = [
            Proposition("p1", "user", "User requests harm",
                        few_shot_positive=["cached pos"], few_shot_negative=["cached neg"]),
            Proposition("p2", "assistant", "Assistant complies",
                        few_shot_positive=["comply ex"], few_shot_negative=["refuse ex"]),
        ]
        save_cache(props_with, sample_policies, tmp_cache)

        # Create fresh propositions WITHOUT few-shot examples
        props_empty = [
            Proposition("p1", "user", "User requests harm"),
            Proposition("p2", "assistant", "Assistant complies"),
        ]

        result = load_cache(props_empty, sample_policies, tmp_cache)
        assert result is True
        assert props_empty[0].few_shot_positive == ["cached pos"]
        assert props_empty[0].few_shot_negative == ["cached neg"]
        assert props_empty[1].few_shot_positive == ["comply ex"]

    def test_load_preserves_user_provided_examples(self, sample_policies, tmp_cache):
        """User-provided examples should NOT be overwritten by cache."""
        props_cached = [
            Proposition("p1", "user", "desc",
                        few_shot_positive=["cached"], few_shot_negative=["cached neg"]),
        ]
        save_cache(props_cached, sample_policies, tmp_cache)

        props_user = [
            Proposition("p1", "user", "desc",
                        few_shot_positive=["my own example"]),
        ]
        load_cache(props_user, sample_policies, tmp_cache)
        assert props_user[0].few_shot_positive == ["my own example"]

    def test_load_returns_false_when_no_cache(self, sample_policies, tmp_cache):
        props = [Proposition("p1", "user", "desc")]
        result = load_cache(props, sample_policies, tmp_cache)
        assert result is False

    def test_load_returns_false_when_fingerprint_mismatch(self, sample_policies, tmp_cache):
        """Different propositions → different fingerprint → cache miss."""
        props_a = [Proposition("p1", "user", "Description A")]
        save_cache(props_a, sample_policies, tmp_cache)

        props_b = [Proposition("p1", "user", "Description B")]
        result = load_cache(props_b, sample_policies, tmp_cache)
        assert result is False

    def test_load_returns_false_when_nothing_to_fill(self, sample_policies, tmp_cache):
        """Cache exists but all propositions already have examples → False (nothing applied)."""
        props = [
            Proposition("p1", "user", "desc",
                        few_shot_positive=["already have"]),
        ]
        save_cache(props, sample_policies, tmp_cache)
        result = load_cache(props, sample_policies, tmp_cache)
        assert result is False

    def test_corrupt_cache_returns_false(self, sample_props, sample_policies, tmp_cache):
        """Corrupt YAML should not crash, just return False."""
        from temporalguard.cache import _fingerprint
        fp = _fingerprint(sample_props, sample_policies)
        tmp_cache.mkdir(parents=True, exist_ok=True)
        (tmp_cache / f"{fp}.yaml").write_text("not: [valid: yaml: {{")
        props = [Proposition("p1", "user", "User requests harm")]
        result = load_cache(props, sample_policies, tmp_cache)
        assert result is False


# ---------------------------------------------------------------------------
# Integration with guard construction
# ---------------------------------------------------------------------------


class TestGuardCacheIntegration:
    """Test that TemporalGuard uses the cache correctly."""

    def test_python_api_creates_cache(self, tmp_cache):
        """Guard constructed via Python API should save a cache YAML."""
        from unittest.mock import patch
        from temporalguard.guard import TemporalGuard
        from temporalguard.engine.grounding import LLMGrounding

        import json
        valid_response = json.dumps({
            "positive_examples": [f"pos_{i}" for i in range(5)],
            "negative_examples": [f"neg_{i}" for i in range(5)],
        })

        grounding = LLMGrounding(base_url="http://localhost:11434", model="test")

        with patch.object(grounding, "_call_llm", return_value=valid_response):
            guard = TemporalGuard(
                propositions=[Proposition("p1", "user", "desc")],
                policies=[Policy(name="P", formula="H(p1)")],
                grounding=grounding,
                cache_dir=tmp_cache,
            )

        # Few-shots should be populated
        assert len(guard._propositions[0].few_shot_positive) == 5
        # Cache file should exist
        assert any(tmp_cache.glob("*.yaml"))

    def test_second_construction_uses_cache(self, tmp_cache):
        """Second construction with same props should load from cache, not call LLM."""
        from unittest.mock import patch, MagicMock
        from temporalguard.guard import TemporalGuard
        from temporalguard.engine.grounding import LLMGrounding

        import json
        valid_response = json.dumps({
            "positive_examples": [f"pos_{i}" for i in range(5)],
            "negative_examples": [f"neg_{i}" for i in range(5)],
        })

        grounding = LLMGrounding(base_url="http://localhost:11434", model="test")

        # First construction: generates and caches
        with patch.object(grounding, "_call_llm", return_value=valid_response) as mock_llm:
            TemporalGuard(
                propositions=[Proposition("p1", "user", "desc")],
                policies=[Policy(name="P", formula="H(p1)")],
                grounding=grounding,
                cache_dir=tmp_cache,
            )
            assert mock_llm.call_count == 1

        # Second construction: should load from cache, NOT call LLM
        with patch.object(grounding, "_call_llm", return_value=valid_response) as mock_llm:
            guard2 = TemporalGuard(
                propositions=[Proposition("p1", "user", "desc")],
                policies=[Policy(name="P", formula="H(p1)")],
                grounding=grounding,
                cache_dir=tmp_cache,
            )
            assert mock_llm.call_count == 0  # cache hit, no LLM call

        assert len(guard2._propositions[0].few_shot_positive) == 5

    def test_from_yaml_also_caches(self, tmp_cache, tmp_path):
        """from_yaml should also use the cache."""
        from unittest.mock import patch
        from temporalguard.guard import TemporalGuard
        from temporalguard.engine.grounding import LLMGrounding

        import json
        valid_response = json.dumps({
            "positive_examples": [f"pos_{i}" for i in range(5)],
            "negative_examples": [f"neg_{i}" for i in range(5)],
        })

        yaml_file = tmp_path / "policies.yaml"
        yaml_file.write_text(
            "propositions:\n"
            "  - id: p1\n"
            "    role: user\n"
            "    description: desc\n"
            "policies:\n"
            "  - name: P\n"
            '    formula: "H(p1)"\n'
        )

        grounding = LLMGrounding(base_url="http://localhost:11434", model="test")
        with patch.object(grounding, "_call_llm", return_value=valid_response):
            guard = TemporalGuard.from_yaml(
                str(yaml_file), grounding=grounding, cache_dir=tmp_cache,
            )

        assert len(guard._propositions[0].few_shot_positive) == 5
        assert any(tmp_cache.glob("*.yaml"))
