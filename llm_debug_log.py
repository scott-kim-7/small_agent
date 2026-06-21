"""Optional JSONL logging of LLM request/response payloads for debugging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_CALL_INDEX = 0
_SESSION_LOG_PATH: Path | None = None


def reset_llm_log_state() -> None:
    """Reset session file and call counter (for tests)."""
    global _CALL_INDEX, _SESSION_LOG_PATH
    _CALL_INDEX = 0
    _SESSION_LOG_PATH = None


def llm_log_enabled() -> bool:
    raw = os.environ.get("SMALL_AGENT_LLM_LOG", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def llm_log_dir() -> Path:
    override = os.environ.get("SMALL_AGENT_LLM_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / ".local" / "llm"


def session_log_path(*, reset: bool = False) -> Path | None:
    """Return the JSONL path for this process (created on first use)."""
    global _SESSION_LOG_PATH
    if reset:
        reset_llm_log_state()
    if not llm_log_enabled():
        return None
    if _SESSION_LOG_PATH is not None:
        return _SESSION_LOG_PATH
    log_dir = llm_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _SESSION_LOG_PATH = log_dir / f"{stamp}-{os.getpid()}.jsonl"
    return _SESSION_LOG_PATH


def _message_role(message: AnyMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    return type(message).__name__.lower()


def serialize_message(message: AnyMessage) -> dict[str, Any]:
    role = _message_role(message)
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    payload: dict[str, Any] = {
        "type": type(message).__name__,
        "role": role,
        "content": content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = list(message.tool_calls)
    if isinstance(message, ToolMessage):
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
    return payload


def serialize_response(message: AIMessage) -> dict[str, Any]:
    payload = serialize_message(message)
    payload["tool_calls"] = list(message.tool_calls or [])
    return payload


def _next_call_index() -> int:
    global _CALL_INDEX
    _CALL_INDEX += 1
    return _CALL_INDEX


def log_llm_exchange(
    request_messages: list[AnyMessage],
    response: AIMessage,
    *,
    mode: str,
) -> Path | None:
    """Append one JSONL record with MLX-converted request and final response."""
    path = session_log_path()
    if path is None:
        return None
    record = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(),
        "event": "llm_exchange",
        "mode": mode,
        "call_index": _next_call_index(),
        "request": {"messages": [serialize_message(m) for m in request_messages]},
        "response": serialize_response(response),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
