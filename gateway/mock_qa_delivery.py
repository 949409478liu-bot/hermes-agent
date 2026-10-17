"""Fail-closed delivery receipt for the single authorized, out-of-vault MOCK.

No discovery by reply text: a producer must explicitly stage pending_transaction
with the current session/chat/inbound ID, card, revision and next question.
Streaming/direct tool sends are deliberately not inferred here.
"""
import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MOCK_ROOT = Path('/Users/mima0000/AI-Test/TASK-FZ-QA-MOBILE-001/mock')
TASK_ID = 'TASK-FZ-QA-MOBILE-001'


def _bound(event, session_key, content):
    if getattr(getattr(event.source, 'platform', None), 'value', None) != 'telegram':
        return None
    root = MOCK_ROOT.resolve()
    path = root / 'state.json'
    # Never follow a state/card symlink out of the authorized MOCK directory.
    if path.is_symlink() or (root / 'card.md').is_symlink():
        return None
    try:
        state = json.loads(path.read_text())
        tx = state.get('pending_transaction')
        if not isinstance(tx, dict):
            return None
        question = state.get('next_question')
        card = str(root / 'card.md')
        if not (state.get('task_id') == TASK_ID and state.get('MOCK') is True
                and state.get('mode') == 'MOCK' and state.get('next_question_status') == 'NOT_SENT'
                and tx.get('kind') == 'telegram_followup'
                and tx.get('session_key') == session_key
                and tx.get('chat_id') == str(event.source.chat_id)
                and event.message_id and tx.get('source_message_id') == str(event.message_id)
                and state.get('last_answer_card') == tx.get('card_path') == card
                and Path(card).is_file()
                and isinstance(state.get('revision'), int)
                and tx.get('revision') == state['revision']
                and state.get('next_question_id') and tx.get('next_question_id') == state['next_question_id']
                and isinstance(question, str) and question and tx.get('next_question') == question
                and question in content):
            return None
        return {'path': path, 'state': state, 'transaction': tx}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def should_block(event, session_key, content):
    # An explicit MOCK inbound turn is blocked while paused even when the
    # producer correctly refused to create a transaction. No reply guessing.
    source = getattr(event, 'source', None)
    from tools.mock_qa_tool import eligible
    if eligible(source, getattr(event, 'text', None)):
        try:
            root = MOCK_ROOT
            path = root / 'state.json'
            if root.is_symlink() or path.is_symlink():
                return True
            state = json.loads(path.read_text())
            if (state.get('task_id') == TASK_ID and state.get('MOCK') is True
                    and state.get('mode') == 'MOCK' and state.get('status') == 'PAUSED'):
                return True
        except (OSError, ValueError, TypeError):
            return True
    ticket = _bound(event, session_key, content)
    return bool(ticket and ticket['state'].get('status') == 'PAUSED')


def prepare(event, session_key, content):
    ticket = _bound(event, session_key, content)
    if ticket and ticket['state'].get('status') in {'ACTIVE', 'NEED_FOLLOWUP'}:
        return ticket
    return None


def complete(ticket, result):
    """Only a successful final text send with a real returned ID is evidence."""
    if not ticket or getattr(result, 'success', False) is not True or not getattr(result, 'message_id', None):
        return False
    root = MOCK_ROOT.resolve()
    path = root / 'state.json'
    if ticket['path'] != path or path.is_symlink():
        return False
    # Serialize our receipts. Compare the entire snapshot so an intervening
    # pause/edit invalidates this receipt instead of overwriting newer data.
    with (root / '.delivery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        if state != ticket['state'] or state.get('status') not in {'ACTIVE', 'NEED_FOLLOWUP'}:
            return False
        tx = ticket['transaction']
        state['next_question_status'] = 'SENT'
        state['next_question_delivery'] = {
            'platform': 'telegram', 'chat_id': tx['chat_id'],
            'message_id': str(result.message_id), 'source_message_id': tx['source_message_id'],
            'next_question_id': tx['next_question_id'], 'card_path': tx['card_path'],
            'sent_at': datetime.now(timezone.utc).isoformat(),
        }
        state['pending_transaction'] = None
        state['revision'] += 1
        state['updated_at'] = state['next_question_delivery']['sent_at']
        fd, name = tempfile.mkstemp(prefix='.delivery-', dir=root)
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(state, out, ensure_ascii=False, indent=2)
                out.write('\n'); out.flush(); os.fsync(out.fileno())
            # A producer not using this lock can still race: recheck immediately
            # before replacement, and document the shared-lock prerequisite.
            if json.loads(path.read_text()) != ticket['state']:
                return False
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return True
