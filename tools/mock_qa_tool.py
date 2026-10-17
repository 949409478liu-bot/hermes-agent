"""Explicit, turn-bound producer for the one authorized out-of-vault MOCK.

No path, answer or source identifiers are accepted from the model. No resume.
Only explicitly MOCK-prefixed inbound turns may stage a transaction.
"""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from gateway import mock_qa_delivery as qa
from tools.registry import registry

_current = ContextVar('mock_qa_producer_turn', default=None)


def eligible(source, message):
    return (getattr(getattr(source, 'platform', None), 'value', None) == 'telegram'
            and bool(getattr(source, 'message_id', None))
            and isinstance(message, str) and message.lstrip().startswith('【MOCK】'))


@contextmanager
def turn(source, session_key, message):
    token = _current.set((source, session_key, message) if eligible(source, message) else None)
    try:
        guide = ''
        if _current.get():
            try:
                root = qa.MOCK_ROOT
                if root.is_symlink() or (root/'state.json').is_symlink():
                    raise ValueError('unsafe MOCK path')
                state = json.loads((root/'state.json').read_text())
                if state.get('status') == 'PAUSED':
                    guide = '\n[MOCK collection is PAUSED. Do not stage, resume or ask a follow-up.]'
                elif state.get('status') in {'ACTIVE', 'NEED_FOLLOWUP'}:
                    guide = ('\n[MOCK producer: call mock_qa_stage with expected_revision='
                             + str(state.get('revision'))
                             + ' and one next_question_id/next_question. Do not claim saved before success. '
                             'After success output exactly the returned next_question; final delivery records its receipt.]')
            except (OSError, ValueError, TypeError):
                guide = '\n[MOCK state unavailable. Do not collect or ask a follow-up.]'
        yield guide
    finally:
        _current.reset(token)


def _atomic(path, content):
    fd, name = tempfile.mkstemp(prefix='.producer-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(content); out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def stage(args):
    try:
        bound = _current.get()
        if not bound:
            raise ValueError('No explicit MOCK inbound turn')
        source, session, answer = bound
        if not eligible(source, answer) or not session or not getattr(source, 'chat_id', None):
            raise ValueError('Unbound source')
        rev = args.get('expected_revision')
        qid, question = args.get('next_question_id'), args.get('next_question')
        if type(rev) is not int or not isinstance(qid, str) or not qid.strip() or not isinstance(question, str) or not question.strip():
            raise ValueError('Invalid revision or next question')
        if len(qid) > 256 or len(question) > 4000 or any(x in question for x in ('MEDIA:', '[[', '<')):
            raise ValueError('Follow-up must be plain text')
        root = qa.MOCK_ROOT
        if root.is_symlink() or root.resolve() != root.absolute():
            raise ValueError('Unsafe MOCK root')
        path, card, lockpath = root/'state.json', root/'card.md', root/'.delivery.lock'
        if any(p.is_symlink() for p in (path, card, lockpath)):
            raise ValueError('Unsafe MOCK file')
        with lockpath.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text())
            if not (state.get('task_id') == qa.TASK_ID and state.get('MOCK') is True
                    and state.get('mode') == 'MOCK' and state.get('status') in {'ACTIVE', 'NEED_FOLLOWUP'}
                    and type(state.get('revision')) is int and state['revision'] == rev
                    and state.get('last_answer_card') == str(card) and card.is_file()
                    and state.get('pending_transaction') is None):
                raise ValueError('Paused, stale, non-MOCK or outstanding transaction')
            mid = str(source.message_id)
            history = state.get('answer_history', [])
            if not isinstance(history, list) or any(h.get('source_message_id') == mid for h in history if isinstance(h, dict)):
                raise ValueError('Duplicate inbound answer or invalid history')
            now = datetime.now(timezone.utc).isoformat()
            history.append(dict(answer_raw=answer, source_message_id=mid, source_message_id_status='KNOWN',
                                question_id=state.get('active_question_id', state.get('question_id')), recorded_at=now))
            state.update(answer_history=history, answer_raw=answer, source_message_id=mid,
                         source_message_id_status='KNOWN', revision=rev+1, updated_at=now,
                         next_question_id=qid, next_question=question, next_question_status='NOT_SENT',
                         next_question_delivery=None, status='NEED_FOLLOWUP', followup_required=True)
            state['pending_transaction'] = dict(kind='telegram_followup', session_key=session,
                chat_id=str(source.chat_id), source_message_id=mid, card_path=str(card), revision=rev+1,
                next_question_id=qid, next_question=question)
            oldcard = card.read_text()
            addition = ('\n\n## MOCK inbound answer (revision '+str(rev+1)+')\n'
                        + 'source_message_id: '+mid+'\n'+answer+'\n\nNext question ['+qid+']: '+question+'\n')
            # Card first, state last: a failed card write can never register delivery.
            _atomic(card, oldcard+addition)
            try:
                _atomic(path, json.dumps(state, ensure_ascii=False, indent=2)+'\n')
            except Exception:
                _atomic(card, oldcard)
                raise
        return json.dumps(dict(success=True, revision=rev+1, next_question=question,
                               next_question_status='NOT_SENT'), ensure_ascii=False)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return json.dumps(dict(success=False, error=str(exc)), ensure_ascii=False)


registry.register(
    name='mock_qa_stage', toolset='file',
    schema={'name':'mock_qa_stage', 'description':'Save the current explicit MOCK inbound answer and register one follow-up transaction. Turn-bound, no paths or source IDs accepted; refuses PAUSED and ordinary chat. Does not send.',
            'parameters': {'type':'object', 'properties': {
                'expected_revision': {'type':'integer'}, 'next_question_id': {'type':'string'},
                'next_question': {'type':'string'}},
                'required':['expected_revision','next_question_id','next_question'], 'additionalProperties':False}},
    handler=lambda args, **kw: stage(args),
    check_fn=lambda: qa.MOCK_ROOT.is_dir() and not qa.MOCK_ROOT.is_symlink(),
)
