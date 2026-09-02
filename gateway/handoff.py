"""Minimal Telegram /new session handoff generation.

The generated text is durable session metadata and is injected later as an
internal ephemeral system prompt.  Nothing in this module writes transcripts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

MAX_TRANSCRIPT_CHARS = 24_000
HANDOFF_TIMEOUT_S = 40.0
HANDOFF_FIELDS = (
    "当前任务",
    "已确认结论",
    "关键路径/配置",
    "当前未完成事项",
    "下一步",
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content) if content is not None else ""


def format_transcript(messages: Iterable[Dict[str, Any]], max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Render and bound a transcript, retaining both the beginning and end."""
    rows = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        content = _content_text(message.get("content")).strip()
        if content:
            rows.append(f"[{role}]\n{content}")
    text = "\n\n".join(rows)
    if len(text) <= max_chars:
        return text
    marker = "\n\n[...中间内容因长度限制省略...]\n\n"
    available = max(0, max_chars - len(marker))
    head = available // 2
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def build_handoff_prompt(transcript: str) -> List[Dict[str, str]]:
    schema = {field: "非空字符串" for field in HANDOFF_FIELDS}
    return [
        {
            "role": "system",
            "content": (
                "你负责把旧会话压缩成新会话可继续工作的交接。"
                "只输出一个 JSON 对象，不要 Markdown，不要额外字段。"
                f"JSON 必须严格包含这些键：{json.dumps(schema, ensure_ascii=False)}。"
                "内容必须忠于记录；未知内容明确写‘未确认’，不得编造。"
            ),
        },
        {"role": "user", "content": "请生成五段式交接：\n\n" + transcript},
    ]


def _response_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
        return ""


def parse_handoff_response(content: str) -> Dict[str, str]:
    """Validate the exact five-field JSON response or raise ValueError."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != set(HANDOFF_FIELDS):
        raise ValueError("handoff response must contain exactly five fields")
    cleaned: Dict[str, str] = {}
    for field in HANDOFF_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"handoff field {field!r} is empty")
        cleaned[field] = value.strip()
    return cleaned


def format_handoff(fields: Dict[str, str]) -> str:
    return "\n\n".join(f"## {field}\n{fields[field]}" for field in HANDOFF_FIELDS)


def generate_handoff(
    messages: Iterable[Dict[str, Any]],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
    timeout: float = HANDOFF_TIMEOUT_S,
) -> str:
    """Make at most one auxiliary LLM call and return validated handoff text."""
    transcript = format_transcript(messages)
    if not transcript.strip():
        raise ValueError("cannot generate handoff from an empty transcript")

    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="session_handoff",
        messages=build_handoff_prompt(transcript),
        temperature=0.1,
        max_tokens=900,
        timeout=min(float(timeout), HANDOFF_TIMEOUT_S),
        main_runtime=main_runtime,
    )
    return format_handoff(parse_handoff_response(_response_content(response)))
