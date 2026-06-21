import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import llm_debug_log


def test_llm_log_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SMALL_AGENT_LLM_LOG", raising=False)
    assert llm_debug_log.llm_log_enabled() is False


def test_llm_log_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "1")
    assert llm_debug_log.llm_log_enabled() is True


def test_serialize_message_variants():
    system = llm_debug_log.serialize_message(SystemMessage(content="sys"))
    assert system["role"] == "system"
    assert system["content"] == "sys"

    human = llm_debug_log.serialize_message(HumanMessage(content="hi"))
    assert human["role"] == "user"

    ai = llm_debug_log.serialize_message(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "bash", "args": {"command": "ls"}, "id": "1", "type": "tool_call"}
            ],
        )
    )
    assert ai["role"] == "assistant"
    assert ai["tool_calls"][0]["name"] == "bash"

    tool = llm_debug_log.serialize_message(
        ToolMessage(content="out", tool_call_id="1", name="bash")
    )
    assert tool["role"] == "tool"
    assert tool["name"] == "bash"


def test_log_llm_exchange_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "1")
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG_DIR", str(tmp_path))
    llm_debug_log.session_log_path(reset=True)

    request = [SystemMessage(content="sys"), HumanMessage(content="hello")]
    response = AIMessage(content="hi there")
    path = llm_debug_log.log_llm_exchange(request, response, mode="buffered")

    assert path is not None
    assert path.exists()
    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["event"] == "llm_exchange"
    assert record["mode"] == "buffered"
    assert record["call_index"] == 1
    assert record["request"]["messages"][1]["content"] == "hello"
    assert record["response"]["content"] == "hi there"


def test_log_llm_exchange_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG", "0")
    monkeypatch.setenv("SMALL_AGENT_LLM_LOG_DIR", str(tmp_path))
    llm_debug_log.session_log_path(reset=True)

    path = llm_debug_log.log_llm_exchange(
        [HumanMessage(content="x")],
        AIMessage(content="y"),
        mode="buffered",
    )
    assert path is None
    assert list(tmp_path.iterdir()) == []
