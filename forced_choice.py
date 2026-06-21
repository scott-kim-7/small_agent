"""Force the model to pick a tool on the first agent call per user turn."""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

RESPOND_DIRECTLY_NAME = "respond_directly"


def force_choice_enabled() -> bool:
    raw = os.environ.get("SMALL_AGENT_FORCE_CHOICE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


@tool
def respond_directly(reason: str = "") -> str:
    """Answer without tools. Only greetings, opinions, timeless facts.
    Never for today/now/weather/news/sports/prices — use web_search or context7_search."""
    return "OK — answer the user directly."


def respond_directly_ack() -> str:
    return "OK — answer the user directly."


def is_user_human_message(message: AnyMessage) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    content = str(message.content or "")
    return not content.startswith("[Tool ")


def is_forced_choice_turn(messages: list[AnyMessage]) -> bool:
    """True on the first agent call after the latest real user message."""
    last_user_idx = max(
        (i for i, m in enumerate(messages) if is_user_human_message(m)),
        default=-1,
    )
    if last_user_idx < 0:
        return False
    for message in messages[last_user_idx + 1 :]:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                return False
            if str(message.content or "").strip():
                return False
    return True


def tools_for_turn(real_tools: list[BaseTool], *, forced: bool) -> list[BaseTool]:
    if forced:
        return [*real_tools, respond_directly]
    return list(real_tools)


def tool_choice_for_turn(forced: bool) -> Literal["required", "auto"]:
    return "required" if forced else "auto"


def is_respond_directly_call(tool_calls: list[Any]) -> bool:
    if not tool_calls:
        return False
    names = [str(call.get("name") or "") for call in tool_calls if isinstance(call, dict)]
    return bool(names) and all(name == RESPOND_DIRECTLY_NAME for name in names)
