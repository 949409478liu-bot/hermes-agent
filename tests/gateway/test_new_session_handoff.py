"""Focused tests for the minimal Telegram /new handoff path."""
import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.handoff import (
    HANDOFF_FIELDS,
    MAX_TRANSCRIPT_CHARS,
    build_handoff_prompt,
    format_transcript,
    generate_handoff,
    parse_handoff_response,
)
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key


def _response(fields=None):
    fields = fields or {field: f"内容-{i}" for i, field in enumerate(HANDOFF_FIELDS)}
    msg = SimpleNamespace(content=__import__("json").dumps(fields, ensure_ascii=False))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_transcript_is_bounded_and_keeps_edges():
    text = format_transcript([{"role": "user", "content": "A" * 20_000}, {"role": "assistant", "content": "Z" * 20_000}])
    assert len(text) <= MAX_TRANSCRIPT_CHARS
    assert text.startswith("[user]")
    assert text.endswith("Z" * 20)
    assert "省略" in text


def test_handoff_response_requires_exact_five_nonempty_fields():
    fields = {field: field for field in HANDOFF_FIELDS}
    assert parse_handoff_response(__import__("json").dumps(fields, ensure_ascii=False)) == fields
    with pytest.raises(ValueError):
        parse_handoff_response("{}")


def test_generate_handoff_calls_auxiliary_once_and_uses_runtime():
    with patch("agent.auxiliary_client.call_llm", return_value=_response()) as call:
        result = generate_handoff([{"role": "user", "content": "继续部署"}], main_runtime={"model": "m"})
    call.assert_called_once()
    assert all(field in result for field in HANDOFF_FIELDS)
    assert call.call_args.kwargs["main_runtime"] == {"model": "m"}
    assert call.call_args.kwargs["timeout"] <= 45


def _runner(history, handoff_result=None):
    runner = object.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm")
    key = build_session_key(source)
    old = SessionEntry(key, "old", datetime.now(), datetime.now(), platform=Platform.TELEGRAM)
    new = SessionEntry(key, "new", datetime.now(), datetime.now(), platform=Platform.TELEGRAM)
    store = MagicMock()
    store._entries = {key: old}
    store.reset_session.return_value = new
    store.load_transcript_for_handoff.return_value = history
    runner.session_store = store
    runner._async_session_store = None
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")})
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._is_user_authorized = lambda source: True
    runner._format_session_info = lambda: ""
    runner._telegram_topic_new_header = lambda source: "new"
    runner._is_telegram_topic_lane = lambda source: False
    runner._resolve_session_agent_runtime = lambda **kwargs: ("m", {"provider": "p", "api_key": "k"})
    return runner, MessageEvent(text="/new", source=source)


@pytest.mark.asyncio
async def test_telegram_new_success_persists_pending_handoff():
    runner, event = _runner([{"role": "user", "content": "old"}])
    with patch("gateway.handoff.generate_handoff", return_value="## 当前任务\n继续"):
        result = await runner._handle_reset_command(event)
    assert "new" in str(result)
    kwargs = runner.session_store.reset_session.call_args.kwargs
    assert kwargs["expected_session_id"] == "old"
    assert kwargs["metadata"]["telegram_new_handoff"]["pending"] is True


@pytest.mark.asyncio
async def test_telegram_new_failure_keeps_old_session():
    runner, event = _runner([{"role": "user", "content": "old"}])
    with patch("gateway.handoff.generate_handoff", side_effect=RuntimeError("boom")):
        result = await runner._handle_reset_command(event)
    assert "旧 Session 已保留" in str(result)
    runner.session_store.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_reset_deadlock_regression_still_selected():
    # The existing regression suite owns the detailed off-loop assertions;
    # this smoke test ensures the native handler remains callable.
    runner, event = _runner([])
    await runner._handle_reset_command(event)
    runner.session_store.reset_session.assert_called_once()
