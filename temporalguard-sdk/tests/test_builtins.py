from temporalguard.builtins import (
    BUILTIN_USER_TURN,
    BUILTIN_PROPOSITIONS,
    is_builtin_proposition,
)

def test_user_turn_constant():
    assert BUILTIN_USER_TURN == "user_turn"

def test_user_turn_in_set():
    assert "user_turn" in BUILTIN_PROPOSITIONS

def test_is_builtin_true():
    assert is_builtin_proposition("user_turn") is True

def test_is_builtin_false():
    assert is_builtin_proposition("p_fraud") is False

def test_is_builtin_whitespace():
    assert is_builtin_proposition("  user_turn  ") is True

def test_is_builtin_empty():
    assert is_builtin_proposition("") is False

def test_is_builtin_none():
    assert is_builtin_proposition(None) is False
