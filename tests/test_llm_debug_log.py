import json

import httpx

import llm_debug_log


def test_llm_log_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SMALL_AGENT_LLM_LOG", raising=False)
    assert llm_debug_log.llm_log_enabled() is False


def test_llm_log_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "1")
    assert llm_debug_log.llm_log_enabled() is True


def test_logging_transport_records_buffered_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "1")
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG_DIR", str(tmp_path))
    llm_debug_log.session_log_path(reset=True)

    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "tool_choice": "required",
        "stream": False,
    }
    response_body = {
        "choices": [{"message": {"role": "assistant", "content": "hi there"}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content.decode())["tool_choice"] == "required"
        return httpx.Response(200, json=response_body)

    transport = llm_debug_log.LoggingTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.post(
            "http://127.0.0.1:8089/v1/chat/completions",
            json=request_body,
        )
        assert response.json() == response_body

    path = llm_debug_log.session_log_path()
    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["event"] == "llm_http"
    assert record["call_index"] == 1
    assert record["request"]["body"]["tool_choice"] == "required"
    assert record["request"]["body"]["messages"][0]["content"] == "hello"
    assert record["response"]["stream"] is False
    assert record["response"]["body"]["choices"][0]["message"]["content"] == "hi there"


def test_logging_transport_records_stream_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "1")
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG_DIR", str(tmp_path))
    llm_debug_log.session_log_path(reset=True)

    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    class ByteStream(httpx.SyncByteStream):
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __iter__(self):
            yield self._payload

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content.decode())["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ByteStream(sse.encode("utf-8")),
        )

    transport = llm_debug_log.LoggingTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with client.stream(
            "POST",
            "http://127.0.0.1:8089/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "stream me"}],
                "stream": True,
            },
        ) as response:
            body = response.read()

    assert body.decode("utf-8") == sse
    record = json.loads(llm_debug_log.session_log_path().read_text(encoding="utf-8").strip())
    assert record["response"]["stream"] is True
    assert record["response"]["body"] == sse
    assert record["request"]["body"]["messages"][0]["content"] == "stream me"


def test_append_log_record_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "0")
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG_DIR", str(tmp_path))
    llm_debug_log.session_log_path(reset=True)

    path = llm_debug_log.append_log_record({"event": "llm_http"})
    assert path is None
    assert list(tmp_path.iterdir()) == []
