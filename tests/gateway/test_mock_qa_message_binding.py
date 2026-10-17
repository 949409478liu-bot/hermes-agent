import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_flush_preserves_real_platform_id(tmp_path):
    from hermes_state import SessionDB
    from tests.run_agent.test_identity_flush import _make_agent, SESSION_ID
    db = SessionDB(db_path=tmp_path / 'test.db')
    try:
        agent = _make_agent(db)
        messages = [{'role': 'user', 'content': 'same text', 'message_id': '471'},
                    {'role': 'assistant', 'content': 'reply'}]
        agent._flush_messages_to_session_db(messages, [])
        agent._flush_messages_to_session_db(messages, [])
        rows = db._conn.execute('SELECT role, platform_message_id FROM messages WHERE session_id=? ORDER BY id', (SESSION_ID,)).fetchall()
        assert [(r['role'], r['platform_message_id']) for r in rows] == [('user', '471'), ('assistant', None)]
    finally:
        db.close()


def test_turn_id_keyword_propagates():
    import inspect
    from run_agent import AIAgent
    from agent.conversation_loop import run_conversation
    from agent.turn_context import build_turn_context
    for fn in (AIAgent.run_conversation, run_conversation, build_turn_context):
        assert 'persist_user_message_id' in inspect.signature(fn).parameters


def test_mock_delivery_transaction(tmp_path, monkeypatch):
    from gateway import mock_qa_delivery as qa
    monkeypatch.setattr(qa, 'MOCK_ROOT', tmp_path / 'TASK-FZ-QA-MOBILE-001' / 'mock')
    root = qa.MOCK_ROOT
    root.mkdir(parents=True)
    card = root / 'card.md'
    card.write_text('# MOCK card')
    state = dict(task_id='TASK-FZ-QA-MOBILE-001', MOCK=True, mode='MOCK', status='NEED_FOLLOWUP', revision=4,
                 last_answer_card=str(card), next_question_id='MOCK-Q-PLACE', next_question='Where?', next_question_status='NOT_SENT',
                 pending_transaction=dict(kind='telegram_followup', session_key='session', chat_id='chat', source_message_id='471',
                                          card_path=str(card), revision=4, next_question_id='MOCK-Q-PLACE', next_question='Where?'))
    path = root / 'state.json'
    def write(): path.write_text(json.dumps(state))
    def read(): return json.loads(path.read_text())
    event = SimpleNamespace(message_id='471', source=SimpleNamespace(platform=SimpleNamespace(value='telegram'), chat_id='chat'))
    write()
    ticket = qa.prepare(event, 'session', 'Next: Where?')
    assert ticket is not None
    assert qa.complete(ticket, SimpleNamespace(success=False, message_id=None)) is False
    assert read()['next_question_status'] == 'NOT_SENT'
    assert qa.complete(ticket, SimpleNamespace(success=True, message_id=None)) is False
    assert qa.complete(ticket, SimpleNamespace(success=True, message_id='900')) is True
    result = read()
    assert result['next_question_status'] == 'SENT'
    assert result['next_question_delivery']['message_id'] == '900'
    assert result['next_question_delivery']['source_message_id'] == '471'
    assert result['revision'] == 5
    assert qa.complete(ticket, SimpleNamespace(success=True, message_id='901')) is False

    # Ordinary chat, wrong source, wrong card, stale revision and non-MOCK never bind.
    for field, value in [('MOCK', False), ('mode', 'LIVE'), ('revision', 99), ('last_answer_card', '/outside/card.md')]:
        original = state[field]; state[field] = value; write()
        assert qa.prepare(event, 'session', 'Next: Where?') is None
        state[field] = original
    write()
    assert qa.prepare(event, 'session', 'ordinary chat') is None
    assert qa.prepare(event, 'other-session', 'Where?') is None
    event.message_id = '472'
    assert qa.prepare(event, 'session', 'Where?') is None
    event.message_id = '471'
    # Paused records cannot bind. Pausing after preparation blocks the write.
    ticket = qa.prepare(event, 'session', 'Where?')
    state['status'] = 'PAUSED'; write()
    assert qa.prepare(event, 'session', 'Where?') is None
    assert qa.complete(ticket, SimpleNamespace(success=True, message_id='902')) is False
    assert read()['next_question_status'] == 'NOT_SENT'


def test_real_turn_prologue_carries_id():
    from tests.agent.test_turn_context import _FakeAgent, _build
    from unittest.mock import patch
    with patch('agent.auxiliary_client.set_runtime_main', lambda *a, **k: None):
        agent = _FakeAgent()
        ctx = _build(agent, persist_user_message_id='471')
        assert ctx.messages[ctx.current_turn_user_idx]['message_id'] == '471'
        ctx = _build(agent)
        assert 'message_id' not in ctx.messages[ctx.current_turn_user_idx]


@pytest.mark.parametrize('success, paused, ordinary', [(True, False, False), (False, False, False), (True, True, False), (True, False, True)])
def test_real_final_send_pipeline(tmp_path, monkeypatch, success, paused, ordinary):
    import asyncio
    from unittest.mock import AsyncMock
    from gateway import mock_qa_delivery as qa
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType, SendResult
    from gateway.session import SessionSource
    from tests.gateway.test_delivery_ledger_producer import _Adapter
    monkeypatch.setattr(qa, 'MOCK_ROOT', tmp_path / 'mock')
    qa.MOCK_ROOT.mkdir()
    card = qa.MOCK_ROOT / 'card.md'; card.write_text('# MOCK')
    state = dict(task_id=qa.TASK_ID, MOCK=True, mode='MOCK', status='PAUSED' if paused else 'NEED_FOLLOWUP', revision=4,
                 last_answer_card=str(card), next_question_id='PLACE', next_question='Where?', next_question_status='NOT_SENT',
                 pending_transaction=dict(kind='telegram_followup', session_key='session', chat_id='chat', source_message_id='471',
                                          card_path=str(card), revision=4, next_question_id='PLACE', next_question='Where?'))
    path = qa.MOCK_ROOT / 'state.json'; path.write_text(json.dumps(state))
    adapter = _Adapter(); adapter.platform = Platform.TELEGRAM
    adapter.config.typing_indicator = False
    adapter.send = AsyncMock(return_value=SendResult(success=success, message_id='900' if success else None, retryable=False, error=None if success else 'chat_not_found'))
    adapter._message_handler = AsyncMock(return_value='ordinary chat' if ordinary else 'Next: Where?')
    event = MessageEvent(text='MOCK answer', message_type=MessageType.TEXT, message_id='471',
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id='chat', chat_type='dm', message_id='471'))
    asyncio.run(adapter._process_message_background(event, 'session'))
    result = json.loads(path.read_text())
    assert result['next_question_status'] == ('SENT' if success and not paused and not ordinary else 'NOT_SENT')
    # The existing send pipeline attempts a plain-text fallback on failure.
    assert adapter.send.await_count == (0 if paused and not ordinary else (1 if success else 2))
