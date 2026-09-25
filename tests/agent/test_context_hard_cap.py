"""Invariants for the pre-API context hard cap."""

from types import SimpleNamespace

from agent.turn_preflight_gate import context_hard_cap_verdict
from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_default_hard_cap_is_conservative_and_enabled():
    config = DEFAULT_CONFIG["compression"]
    assert config["hard_cap_enabled"] is True
    assert config["hard_cap_tokens"] == 220000
    assert config["hard_cap_action"] in {"new_session", "reset"}


def test_hard_cap_blocks_request_at_limit_and_requests_new_session():
    verdict = context_hard_cap_verdict(
        SimpleNamespace(context_hard_cap_enabled=True, context_hard_cap_tokens=220_000,
                        context_hard_cap_action="new_session"),
        220_000,
    )
    assert verdict == "new_session"


def test_hard_cap_is_disabled_without_config_and_does_not_block():
    assert context_hard_cap_verdict(SimpleNamespace(), 999_999) is None


def test_hard_cap_never_changes_for_under_limit_request():
    agent = SimpleNamespace(context_hard_cap_enabled=True, context_hard_cap_tokens=220_000,
                            context_hard_cap_action="reset")
    assert context_hard_cap_verdict(agent, 219_999) is None
