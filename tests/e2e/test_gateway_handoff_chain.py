"""Mocked end-to-end contract for pending Telegram handoff consumption."""
from datetime import datetime
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key


def test_pending_handoff_is_cas_read_and_consumed_once(tmp_path):
    store = object.__new__(SessionStore)
    import threading
    store._lock = threading.RLock()
    store._loaded = True
    key = build_session_key(SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u"))
    entry = SessionEntry(key, "new", datetime.now(), datetime.now(), platform=Platform.TELEGRAM,
                         metadata={"telegram_new_handoff": {"pending": True, "text": "handoff"}})
    store._entries = {key: entry}
    store._db = None
    store._save = MagicMock()
    assert store.get_pending_telegram_handoff(key, "new")["text"] == "handoff"
    assert store.get_pending_telegram_handoff(key, "wrong") is None
    assert store.consume_pending_telegram_handoff(key, "new")
    assert not store.consume_pending_telegram_handoff(key, "new")
    assert store._entries[key].metadata["telegram_new_handoff"]["pending"] is False
