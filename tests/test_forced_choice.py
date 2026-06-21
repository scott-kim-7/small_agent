from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import forced_choice


def test_force_choice_enabled_default():
    assert forced_choice.force_choice_enabled() is True


def test_force_choice_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SMALL_AGENT_FORCE_CHOICE", "0")
    assert forced_choice.force_choice_enabled() is False


def test_is_forced_fresh_user_message():
    messages = [SystemMessage(content="sys"), HumanMessage(content="hello")]
    assert forced_choice.is_forced_choice_turn(messages) is True


def test_is_forced_new_user_in_long_conversation():
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old reply"),
        HumanMessage(content="new question"),
    ]
    assert forced_choice.is_forced_choice_turn(messages) is True


def test_is_forced_false_after_tool_call():
    messages = [
        HumanMessage(content="news?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "news"}, "id": "1", "type": "tool_call"}
            ],
        ),
    ]
    assert forced_choice.is_forced_choice_turn(messages) is False


def test_is_forced_false_after_respond_directly_ack():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "respond_directly",
                    "args": {},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="OK", tool_call_id="1", name="respond_directly"),
    ]
    assert forced_choice.is_forced_choice_turn(messages) is False


def test_is_forced_false_after_final_text():
    messages = [HumanMessage(content="hi"), AIMessage(content="Hello!")]
    assert forced_choice.is_forced_choice_turn(messages) is False


def test_is_forced_ignores_tool_output_human_messages():
    messages = [
        HumanMessage(content="list files"),
        AIMessage(
            content="",
            tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "1", "type": "tool_call"}],
        ),
        ToolMessage(content="a.txt", tool_call_id="1", name="bash"),
        HumanMessage(content="[Tool `bash` output]\na.txt\n\nUse this result to continue."),
    ]
    assert forced_choice.is_forced_choice_turn(messages) is False


def test_tools_for_turn_includes_respond_directly_when_forced():
    real = [forced_choice.respond_directly]
    forced_tools = forced_choice.tools_for_turn(real, forced=True)
    names = {tool.name for tool in forced_tools}
    assert "respond_directly" in names


def test_tools_for_turn_excludes_respond_directly_when_not_forced():
    from langchain_core.tools import tool

    @tool
    def sample_tool(x: str) -> str:
        """sample"""
        return x

    normal_tools = forced_choice.tools_for_turn([sample_tool], forced=False)
    names = {tool.name for tool in normal_tools}
    assert names == {"sample_tool"}
    assert "respond_directly" not in names


def test_is_respond_directly_call():
    calls = [{"name": "respond_directly", "args": {}, "id": "1"}]
    assert forced_choice.is_respond_directly_call(calls) is True
    mixed = [{"name": "respond_directly", "id": "1"}, {"name": "bash", "id": "2"}]
    assert forced_choice.is_respond_directly_call(mixed) is False
