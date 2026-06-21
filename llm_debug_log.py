"""Optional JSONL logging of raw HTTP payloads to/from the LLM server."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

_CALL_INDEX = 0
_SESSION_LOG_PATH: Path | None = None
_LOGGING_HTTP_CLIENT: httpx.Client | None = None


def reset_llm_log_state() -> None:
    """Reset session file, call counter, and HTTP client (for tests)."""
    global _CALL_INDEX, _SESSION_LOG_PATH, _LOGGING_HTTP_CLIENT
    _CALL_INDEX = 0
    _SESSION_LOG_PATH = None
    if _LOGGING_HTTP_CLIENT is not None:
        _LOGGING_HTTP_CLIENT.close()
        _LOGGING_HTTP_CLIENT = None


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


def _next_call_index() -> int:
    global _CALL_INDEX
    _CALL_INDEX += 1
    return _CALL_INDEX


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    out = dict(headers)
    for key in list(out):
        if key.lower() == "authorization":
            out[key] = "[REDACTED]"
    return out


def _try_parse_json(data: bytes) -> Any:
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _serialize_http_body(content: bytes, content_type: str) -> Any:
    if not content:
        return None
    lowered = content_type.lower()
    if "json" in lowered:
        return _try_parse_json(content)
    return content.decode("utf-8", errors="replace")


def _request_is_stream(content: bytes) -> bool:
    parsed = _try_parse_json(content)
    return isinstance(parsed, dict) and bool(parsed.get("stream"))


def _serialize_request(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": _safe_headers(dict(request.headers)),
        "body": _try_parse_json(request.content),
    }


def _serialize_response(
    response: httpx.Response,
    body: bytes,
    *,
    streamed: bool,
) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    payload: dict[str, Any] = {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "stream": streamed,
        "body": _serialize_http_body(body, content_type),
    }
    return payload


def append_log_record(record: dict[str, Any]) -> Path | None:
    path = session_log_path()
    if path is None:
        return None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class _TeeByteStream(httpx.SyncByteStream):
    def __init__(
        self,
        inner: httpx.SyncByteStream,
        *,
        on_complete: Callable[[bytes], None],
    ) -> None:
        self._inner = inner
        self._on_complete = on_complete
        self._chunks: list[bytes] = []
        self._finished = False

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._finish()

    def close(self) -> None:
        self._inner.close()
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._on_complete(b"".join(self._chunks))


class LoggingTransport(httpx.BaseTransport):
    """Log full OpenAI-compatible HTTP request/response bodies."""

    def __init__(self, wrapped: httpx.BaseTransport | None = None) -> None:
        self._wrapped = wrapped or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        call_index = _next_call_index()
        req_snapshot = _serialize_request(request)
        response = self._wrapped.handle_request(request)

        if _request_is_stream(request.content):
            return self._wrap_streaming_response(response, call_index, req_snapshot, request)

        body = response.read()
        append_log_record(
            {
                "ts": datetime.now(timezone.utc).astimezone().isoformat(),
                "event": "llm_http",
                "call_index": call_index,
                "request": req_snapshot,
                "response": _serialize_response(response, body, streamed=False),
            }
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=body,
            request=request,
        )

    def _wrap_streaming_response(
        self,
        response: httpx.Response,
        call_index: int,
        req_snapshot: dict[str, Any],
        request: httpx.Request,
    ) -> httpx.Response:
        def on_complete(body: bytes) -> None:
            append_log_record(
                {
                    "ts": datetime.now(timezone.utc).astimezone().isoformat(),
                    "event": "llm_http",
                    "call_index": call_index,
                    "request": req_snapshot,
                    "response": _serialize_response(response, body, streamed=True),
                }
            )

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_TeeByteStream(response.stream, on_complete=on_complete),
            request=request,
        )

    def close(self) -> None:
        self._wrapped.close()


def logging_http_client() -> httpx.Client:
    """Shared httpx client that JSONL-logs every chat/completions HTTP exchange."""
    global _LOGGING_HTTP_CLIENT
    if _LOGGING_HTTP_CLIENT is None:
        _LOGGING_HTTP_CLIENT = httpx.Client(
            transport=LoggingTransport(),
            timeout=300.0,
        )
    return _LOGGING_HTTP_CLIENT
