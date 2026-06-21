from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import graph


def test_should_continue_with_tool_calls():
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "pwd"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert graph.should_continue(state) == "tools"


def test_should_continue_without_tool_calls():
    state = {"messages": [AIMessage(content="done")]}
    assert graph.should_continue(state) == graph.END


def test_tools_node_runs_bash(monkeypatch):
    captured: list[str] = []

    class FakeBash:
        def run(self, command: str):
            captured.append(command)

            class Obs:
                def to_llm_string(self):
                    return f"ran:{command}"

            return Obs()

    monkeypatch.setattr(graph, "_BASH", FakeBash())
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "echo hi"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
    result = graph.tools_node(state)
    assert captured == ["echo hi"]
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "ran:echo hi"


def test_parse_tool_args_string_fallback():
    assert graph._parse_tool_args("bash", "pwd") == {"command": "pwd"}
    assert graph._parse_tool_args("python", 'print("x")') == {"code": 'print("x")'}


def test_parse_tool_args_json_string():
    assert graph._parse_tool_args("bash", '{"command": "echo hi"}') == {"command": "echo hi"}


def test_messages_for_mlx_converts_tool_follow_up():
    converted = graph.messages_for_mlx(
        [
            HumanMessage(content="list files"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "ls"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="a.txt", tool_call_id="call_1", name="bash"),
        ]
    )
    assert len(converted) == 2
    assert isinstance(converted[0], HumanMessage)
    assert converted[0].content == "list files"
    assert isinstance(converted[1], HumanMessage)
    assert "[Tool `bash` output]" in str(converted[1].content)


def test_final_assistant_text_skips_tool_calls():
    text = graph.final_assistant_text(
        [
            HumanMessage(content="news?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "news"}, "id": "1", "type": "tool_call"}
                ],
            ),
            ToolMessage(content="search hit", tool_call_id="1", name="web_search"),
            AIMessage(content="Here is the answer."),
        ]
    )
    assert text == "Here is the answer."


def test_final_assistant_text_falls_back_to_tool_output():
    text = graph.final_assistant_text(
        [
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "1", "type": "tool_call"}]),
            ToolMessage(content="raw search results", tool_call_id="1", name="web_search"),
            AIMessage(content=""),
        ]
    )
    assert text == "raw search results"


def test_effective_model_id_auto_resolves(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setattr(graph, "resolve_model_id", lambda base, key: "mlx-main")
    assert graph.effective_model_id("http://127.0.0.1:8089/v1", "local") == "mlx-main"


def test_effective_model_id_prefers_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "custom-model")
    assert graph.effective_model_id("http://127.0.0.1:8089/v1", "local") == "custom-model"


def test_system_prompt_defined():
    assert "bash" in graph.SYSTEM_PROMPT_BODY
    assert "python" in graph.SYSTEM_PROMPT_BODY
    assert "web_search" in graph.SYSTEM_PROMPT_BODY
    assert "context7_search" in graph.SYSTEM_PROMPT_BODY
    assert "Training data is stale" in graph.SYSTEM_PROMPT_BODY
    assert "respond_directly" in graph.SYSTEM_PROMPT_BODY
    assert "Reply in the user's language" in graph.SYSTEM_PROMPT_BODY


def test_build_system_prompt_includes_clock():
    from datetime import timezone

    fixed = datetime(2026, 6, 17, 15, 30, 0, tzinfo=timezone.utc)
    text = graph.build_system_prompt(now=fixed)
    assert "Today's date and time" in text
    assert "2026-06-17 15:30:00 UTC" in text
    assert "Tool output overrides" in text


def test_refresh_system_message_replaces_existing():
    from langchain_core.messages import SystemMessage

    old = SystemMessage(content="old")
    updated = graph.refresh_system_message([old, HumanMessage(content="hi")])
    assert isinstance(updated[0], SystemMessage)
    assert "Today's date and time" in str(updated[0].content)
    assert updated[1].content == "hi"


def test_chunk_to_text():
    assert graph.chunk_to_text(AIMessage(content="hi")) == "hi"
    assert graph.chunk_to_text(AIMessage(content="")) == ""


def test_show_thinking_enabled_default(monkeypatch):
    monkeypatch.delenv("SMALL_AGENT_SHOW_THINKING", raising=False)
    assert graph.show_thinking_enabled() is True


def test_show_thinking_disabled(monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_SHOW_THINKING", "0")
    assert graph.show_thinking_enabled() is False


def test_make_thinking_marker_handlers():
    parts: list[str] = []
    on_reasoning, on_token, on_tool_call, _close = graph.make_thinking_marker_handlers(
        write=parts.append,
    )
    on_reasoning("plan")
    on_token("answer")
    on_tool_call("\n[bash] ls\n")
    text = "".join(parts)
    assert "[thinking]" in text
    assert "[/thinking]" in text
    assert "plan" in text
    assert "answer" in text
    assert "[bash]" in text


def test_thinking_tap_for_sink():
    reasoning: list[str] = []
    sink = graph.StreamSink(on_reasoning=reasoning.append)
    tap = graph._thinking_tap_for_sink(sink)
    tap({"choices": [{"delta": {"reasoning_content": "step"}}]})
    tap(
        {
            "choices": [
                {"message": {"reasoning_content": "full plan", "content": "hi"}},
            ]
        }
    )
    assert reasoning == ["step", "full plan"]


def test_format_tool_call_notice():
    assert "ls" in graph.format_tool_call_notice("bash", {"command": "ls"})
    assert "news" in graph.format_tool_call_notice("web_search", {"query": "news"})
    assert "fastapi" in graph.format_tool_call_notice(
        "context7_search", {"query": "routing", "library": "fastapi"}
    )
    assert graph.format_tool_call_notice("respond_directly", {}) == "\n[respond_directly]\n"


def _patch_fake_llm(monkeypatch, fake):
    monkeypatch.setattr(graph, "llm_for_turn", lambda messages, forced=None: fake)


def test_call_llm_buffered(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content="done")

        def stream(self, messages):
            raise AssertionError("stream should not be called")

    _patch_fake_llm(monkeypatch, FakeLLM())
    out = graph.call_llm([HumanMessage(content="x")], sink=None, forced=False)
    assert out.content == "done"


def test_call_llm_emits_tool_call_notice(monkeypatch):
    notices: list[str] = []

    class FakeLLM:
        def invoke(self, messages):
            raise AssertionError("invoke should not be called")

        def stream(self, messages):
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "news today"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )

    _patch_fake_llm(monkeypatch, FakeLLM())
    sink = graph.StreamSink(on_tool_call=notices.append)
    out = graph.call_llm([HumanMessage(content="x")], sink=sink, forced=False)
    assert out.tool_calls
    assert any("web_search" in line and "news today" in line for line in notices)


def test_call_llm_stream_accumulates(monkeypatch):
    pieces: list[str] = []

    class Chunk:
        def __init__(self, text: str):
            self.content = text
            self.tool_calls = []

        def __add__(self, other):
            return Chunk(self.content + other.content)

    class FakeLLM:
        def invoke(self, messages):
            raise AssertionError("invoke should not be called")

        def stream(self, messages):
            yield Chunk("Hel")
            yield Chunk("lo")

    _patch_fake_llm(monkeypatch, FakeLLM())
    sink = graph.StreamSink(on_token=pieces.append)
    out = graph.call_llm([HumanMessage(content="x")], sink=sink, forced=False)
    assert "".join(pieces) == "Hello"
    assert out.content == "Hello"


def test_tools_include_web_search():
    assert "web_search" in graph.TOOLS


def test_tools_include_context7_search():
    assert "context7_search" in graph.TOOLS


def test_llm_for_turn_forced_uses_required(monkeypatch):
    captured: dict[str, object] = {}

    class FakeBase:
        def bind_tools(self, tools, tool_choice=None):
            captured["tool_choice"] = tool_choice
            captured["tool_names"] = [t.name for t in tools]
            return self

    monkeypatch.setattr(graph, "build_llm_base", lambda: FakeBase())
    graph.llm_for_turn([HumanMessage(content="hi")], forced=True)
    assert captured["tool_choice"] == "required"
    assert "respond_directly" in captured["tool_names"]


def test_llm_for_turn_auto_excludes_respond_directly(monkeypatch):
    captured: dict[str, object] = {}

    class FakeBase:
        def bind_tools(self, tools, tool_choice=None):
            captured["tool_choice"] = tool_choice
            captured["tool_names"] = [t.name for t in tools]
            return self

    monkeypatch.setattr(graph, "build_llm_base", lambda: FakeBase())
    graph.llm_for_turn([HumanMessage(content="hi")], forced=False)
    assert captured["tool_choice"] == "auto"
    assert "respond_directly" not in captured["tool_names"]


def test_agent_respond_directly_bypass(monkeypatch):
    forced_flags: list[bool | None] = []
    invoke_count = 0

    class FakeLLM:
        def invoke(self, messages):
            nonlocal invoke_count
            invoke_count += 1
            if invoke_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "respond_directly",
                            "args": {},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Hello!")

        def stream(self, messages):
            raise AssertionError("stream should not be called")

    def fake_llm_for_turn(messages, forced=None):
        forced_flags.append(forced)
        return FakeLLM()

    monkeypatch.setattr(graph, "llm_for_turn", fake_llm_for_turn)
    node = graph.make_agent_node(sink=None)
    result = node({"messages": [HumanMessage(content="hi")]})
    assert forced_flags[1] is False
    assert invoke_count == 2
    assert result["messages"][0].content == "Hello!"
    assert not result["messages"][0].tool_calls


def test_tools_node_skips_respond_directly(monkeypatch):
    captured: list[str] = []

    class FakeBash:
        def run(self, command: str):
            captured.append(command)

            class Obs:
                def to_llm_string(self):
                    return f"ran:{command}"

            return Obs()

    monkeypatch.setattr(graph, "_BASH", FakeBash())
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "respond_directly",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    },
                    {
                        "name": "bash",
                        "args": {"command": "echo hi"},
                        "id": "2",
                        "type": "tool_call",
                    },
                ],
            )
        ]
    }
    result = graph.tools_node(state)
    assert captured == ["echo hi"]
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "bash"
