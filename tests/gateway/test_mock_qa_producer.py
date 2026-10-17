import json
from types import SimpleNamespace
import pytest
from gateway import mock_qa_delivery as qa

@pytest.fixture
def production(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, 'MOCK_ROOT', tmp_path / 'mock')
    qa.MOCK_ROOT.mkdir()
    (qa.MOCK_ROOT / 'card.md').write_text('# MOCK\n')
    s=dict(task_id=qa.TASK_ID, MOCK=True, mode='MOCK', status='ACTIVE', revision=3,
           last_answer_card=str(qa.MOCK_ROOT/'card.md'), pending_transaction=None, answer_history=[])
    (qa.MOCK_ROOT/'state.json').write_text(json.dumps(s))
    return SimpleNamespace(platform=SimpleNamespace(value='telegram'), chat_id='chat', message_id='471')

def test_registered_producer_to_receipt(production):
    from tools import mock_qa_tool as producer
    from model_tools import get_tool_definitions, handle_function_call
    names=[t['function']['name'] for t in get_tool_definitions(['file'], quiet_mode=True)]
    assert 'mock_qa_stage' in names
    with producer.turn(production, 'session', '【MOCK】测试答案') as guide:
        assert 'mock_qa_stage' in guide
        result=json.loads(handle_function_call('mock_qa_stage', dict(expected_revision=3,next_question_id='PLACE',next_question='Where?')))
    assert result['success'] is True
    s=json.loads((qa.MOCK_ROOT/'state.json').read_text())
    assert s['source_message_id']=='471'
    assert s['revision']==4 and s['next_question_status']=='NOT_SENT'
    assert s['answer_history'][-1]['answer_raw']=='【MOCK】测试答案'
    assert '【MOCK】测试答案' in (qa.MOCK_ROOT/'card.md').read_text()
    event=SimpleNamespace(source=production,message_id='471')
    assert qa.complete(qa.prepare(event,'session','Where?'),SimpleNamespace(success=False,message_id=None)) is False
    assert qa.complete(qa.prepare(event,'session','Where?'),SimpleNamespace(success=True,message_id='900')) is True
    assert json.loads((qa.MOCK_ROOT/'state.json').read_text())['next_question_delivery']['message_id']=='900'

@pytest.mark.parametrize('kind',['paused','ordinary','stale','unbound','symlink'])
def test_producer_fail_closed(production,kind,tmp_path):
    from tools import mock_qa_tool as producer
    path=qa.MOCK_ROOT/'state.json'
    if kind=='paused':
        s=json.loads(path.read_text());s['status']='PAUSED';path.write_text(json.dumps(s))
    if kind=='symlink':
        card=qa.MOCK_ROOT/'card.md';card.unlink();card.symlink_to(tmp_path/'outside')
    before=path.read_bytes()
    with producer.turn(production,'session','ordinary' if kind=='ordinary' else '【MOCK】测试答案'):
        if kind=='unbound': production.message_id=None
        result=producer.stage(dict(expected_revision=99 if kind=='stale' else 3,next_question_id='PLACE',next_question='Where?'))
    assert json.loads(result).get('success') is False
    assert path.read_bytes()==before

def test_no_turn_no_producer(production):
    from tools import mock_qa_tool as producer
    assert json.loads(producer.stage(dict(expected_revision=3,next_question_id='P',next_question='Where?')))['success'] is False

def test_runtime_hook_is_wired():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[2]/'gateway/run.py').read_text()
    assert 'with _mock_qa_turn(ctx.source, ctx.session_key, ctx.message)' in text

@pytest.mark.parametrize('success, paused, ordinary', [(True,False,False),(False,False,False),(True,True,False),(True,False,True)])
def test_registered_producer_actual_delivery(production, monkeypatch, success, paused, ordinary):
    import asyncio
    from unittest.mock import AsyncMock
    from tools import mock_qa_tool as producer
    from model_tools import handle_function_call
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType, SendResult
    from gateway.session import SessionSource
    from tests.gateway.test_delivery_ledger_producer import _Adapter
    statepath=qa.MOCK_ROOT/'state.json'
    if paused:
        s=json.loads(statepath.read_text());s['status']='PAUSED';statepath.write_text(json.dumps(s))
    before=statepath.read_bytes()
    adapter=_Adapter();adapter.platform=Platform.TELEGRAM;adapter.config.typing_indicator=False
    adapter.send=AsyncMock(return_value=SendResult(success=success,message_id='900' if success else None,retryable=False,error=None if success else 'chat_not_found'))
    source=SessionSource(platform=Platform.TELEGRAM,chat_id='chat',chat_type='dm',message_id='471')
    text='ordinary' if ordinary else '【MOCK】测试答案'
    async def handler(event):
        with producer.turn(source,'session',text):
            result=json.loads(handle_function_call('mock_qa_stage',dict(expected_revision=3,next_question_id='PLACE',next_question='Where?'),enabled_tools=['mock_qa_stage'],enabled_toolsets=['hermes-telegram']))
        assert result['success'] == (not paused and not ordinary)
        # Even if a model ignores the pause instruction, actual send is blocked.
        return 'Where?' if not ordinary else 'ordinary chat'
    adapter._message_handler=handler
    event=MessageEvent(text=text,message_type=MessageType.TEXT,message_id='471',source=source)
    asyncio.run(adapter._process_message_background(event,'session'))
    s=json.loads(statepath.read_text())
    if ordinary or paused:
        assert statepath.read_bytes()==before
    else:
        assert s['next_question_status']==('SENT' if success else 'NOT_SENT')
        assert s['pending_transaction'] is None if success else s['pending_transaction']['source_message_id']=='471'
        if success: assert s['next_question_delivery']['message_id']=='900'
    assert adapter.send.await_count==(0 if paused else (1 if success else 2))


def test_execute_real_gateway_turn_hook(production):
    import ast
    from pathlib import Path
    from tools.mock_qa_tool import turn as _mock_qa_turn
    from model_tools import handle_function_call, get_tool_definitions
    tree=ast.parse((Path(__file__).resolve().parents[2]/'gateway/run.py').read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call) and isinstance(n.items[0].context_expr.func,ast.Name) and n.items[0].context_expr.func.id=='_mock_qa_turn')
    seen=[]
    class Agent:
        def run_conversation(self,message,**kwargs):
            seen.append(message)
            names=[t['function']['name'] for t in get_tool_definitions(['hermes-telegram'],quiet_mode=True)]
            assert 'mock_qa_stage' in names
            return json.loads(handle_function_call('mock_qa_stage',dict(expected_revision=3,next_question_id='PLACE',next_question='Where?'),enabled_tools=names,enabled_toolsets=['hermes-telegram']))
    ns=dict(_mock_qa_turn=_mock_qa_turn,ctx=SimpleNamespace(source=production,session_key='session',message='【MOCK】测试答案'),_api_run_message='【MOCK】测试答案',_conversation_kwargs={},agent=Agent())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'gateway/run.py','exec'),ns)
    assert ns['result']['success'] is True
    assert 'mock_qa_stage' in seen[0]
    assert ns['_conversation_kwargs']['persist_user_message']=='【MOCK】测试答案'
    from tools.mock_qa_tool import stage
    assert json.loads(stage(dict(expected_revision=4,next_question_id='X',next_question='x')))['success'] is False

@pytest.mark.parametrize('kind',['outstanding','duplicate','write_failure','state_symlink','root_symlink'])
def test_producer_preserves_transaction_boundary(production,monkeypatch,tmp_path,kind):
    from tools import mock_qa_tool as producer
    path=qa.MOCK_ROOT/'state.json';card=qa.MOCK_ROOT/'card.md'
    if kind in {'outstanding','duplicate'}:
        s=json.loads(path.read_text())
        if kind=='outstanding':s['pending_transaction']={'existing':'transaction'}
        else:s['answer_history']=[{'source_message_id':'471','answer_raw':'old'}]
        path.write_text(json.dumps(s))
    if kind=='state_symlink':
        outside=tmp_path/'outside.json';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    if kind=='root_symlink':
        real=qa.MOCK_ROOT;link=tmp_path/'link';link.symlink_to(real,target_is_directory=True);monkeypatch.setattr(qa,'MOCK_ROOT',link)
    before=(path.read_bytes(),card.read_bytes())
    if kind=='write_failure':
        atomic=producer._atomic
        def fail_state(p,content):
            if p.name=='state.json':raise OSError('injected state write failure')
            return atomic(p,content)
        monkeypatch.setattr(producer,'_atomic',fail_state)
    with producer.turn(production,'session','【MOCK】测试答案'):
        result=json.loads(producer.stage(dict(expected_revision=3,next_question_id='PLACE',next_question='Where?')))
    assert result['success'] is False
    assert (path.read_bytes(),card.read_bytes())==before


def test_context_isolation(production):
    from tools import mock_qa_tool as producer
    from concurrent.futures import ThreadPoolExecutor
    args=dict(expected_revision=3,next_question_id='PLACE',next_question='Where?')
    with producer.turn(production,'session','【MOCK】测试答案'):
        with ThreadPoolExecutor() as pool:
            assert json.loads(pool.submit(producer.stage,args).result())['success'] is False
        assert json.loads(producer.stage(args))['success'] is True
