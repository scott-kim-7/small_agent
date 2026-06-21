"""아주 간단한 ReAct 스타일 LangGraph.

구조:
    agent (LLM, litellm/MLX@8089) ──tool_calls?──> tools (로컬 실행) ──> agent
                                  └──없으면────────> END
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from bash_session import BashSession
from context7_search import resolve_context7_api_key, run_context7_search
from exa_search import resolve_exa_api_key, run_web_search
from forced_choice import (
    RESPOND_DIRECTLY_NAME,
    force_choice_enabled,
    is_forced_choice_turn,
    is_respond_directly_call,
    tool_choice_for_turn,
    tools_for_turn,
)
from llm_debug_log import llm_log_enabled, log_llm_exchange, session_log_path
from python_executor import PythonExecutor

_BASH = BashSession(work_dir=os.getcwd())
_PYTHON = PythonExecutor(work_dir=os.getcwd())
_EXA_API_KEY: str | None = None
_CONTEXT7_API_KEY: str | None = None


def set_exa_api_key(key: str | None) -> None:
    global _EXA_API_KEY
    _EXA_API_KEY = key


def set_context7_api_key(key: str | None) -> None:
    global _CONTEXT7_API_KEY
    _CONTEXT7_API_KEY = key


@tool
def bash(command: str) -> str:
    """Run a shell command. Session persists (cd, env, venv carry over)."""
    obs = _BASH.run(command)
    return obs.to_llm_string()


@tool
def python(code: str) -> str:
    """Run Python once in a fresh subprocess (no state between calls)."""
    obs = _PYTHON.run(code)
    return obs.to_llm_string()


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current/recent facts. Not for library docs (use context7_search)."""
    return run_web_search(_EXA_API_KEY, query, max_results)


@tool
def context7_search(query: str, library: str = "") -> str:
    """Fetch library/framework docs. Set library when known. Not for news/weather (use web_search)."""
    return run_context7_search(_CONTEXT7_API_KEY, query, library)


TOOLS = {
    "bash": bash,
    "python": python,
    "web_search": web_search,
    "context7_search": context7_search,
}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def llm_base_url() -> str:
    return os.environ.get("LITELLM_URL", "http://127.0.0.1:8089/v1").strip().rstrip("/")


def llm_api_key() -> str:
    return os.environ.get("LITELLM_KEY", "local")


def _api_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root[: -len("/v1")].rstrip("/")
    return root


def resolve_model_id(base_url: str, api_key: str, *, timeout: float = 10.0) -> str:
    """MLX /health 또는 /v1/models 에서 현재 로드된 모델 id 를 가져온다."""
    root = _api_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=timeout) as client:
        try:
            resp = client.get(f"{root}/health", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                for key in ("loaded_model", "model_id"):
                    raw = data.get(key)
                    if isinstance(raw, str) and raw.strip():
                        return raw.split(",")[0].strip()
        except httpx.HTTPError:
            pass

        models_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        resp = client.get(f"{models_url}/models", headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            raise RuntimeError(
                f"No model from {models_url}/models — is the LLM server running?"
            )
        first = items[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
        raise RuntimeError(f"Unexpected /v1/models response from {models_url}")


def effective_model_id(base_url: str, api_key: str) -> str:
    explicit = os.environ.get("AGENT_MODEL", "").strip()
    if explicit:
        return explicit
    return resolve_model_id(base_url, api_key)


def build_llm_base() -> ChatOpenAI:
    base_url = llm_base_url()
    api_key = llm_api_key()
    model = effective_model_id(base_url, api_key)
    max_tokens = int(os.environ.get("SMALL_AGENT_MAX_TOKENS", "2048"))
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
        timeout=300,
        max_tokens=max_tokens,
    )


def _forced_for_turn(messages: list[AnyMessage], forced: bool | None) -> bool:
    if forced is not None:
        return forced
    return force_choice_enabled() and is_forced_choice_turn(messages)


def llm_for_turn(messages: list[AnyMessage], *, forced: bool | None = None) -> Any:
    use_forced = _forced_for_turn(messages, forced)
    tools = tools_for_turn(list(TOOLS.values()), forced=use_forced)
    choice = tool_choice_for_turn(use_forced)
    return build_llm_base().bind_tools(tools, tool_choice=choice)


def build_llm() -> Any:
    """Backward-compatible: auto tool choice, no respond_directly."""
    return llm_for_turn([], forced=False)


def stream_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    raw = os.environ.get("SMALL_AGENT_STREAM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_LLM_BASE = build_llm_base()


def messages_for_mlx(messages: list[AnyMessage]) -> list[AnyMessage]:
    """MLX/LiteLLM follow-up turns break on tool_calls/ToolMessage in history.

    Keep LangGraph state canonical, but convert tool results to plain user text
    before calling the model again.
    """
    out: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            name = message.name or "tool"
            out.append(
                HumanMessage(
                    content=(
                        f"[Tool `{name}` output]\n{message.content}\n\n"
                        "Use this result to continue."
                    )
                )
            )
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            content = message.content if isinstance(message.content, str) else ""
            if content.strip():
                out.append(AIMessage(content=content))
            continue
        out.append(message)
    return out


def final_assistant_text(messages: list[AnyMessage]) -> str:
    """Skip tool-call-only assistant turns; return the last natural-language reply."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content or "")
        if content.strip():
            return content.strip()
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content and not content.startswith("[ERROR]"):
                return content
    return ""


def chunk_to_text(chunk: AnyMessage) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        return content
    return ""


def _parse_tool_args(name: str, args: object) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        if name == "bash":
            return {"command": args}
        if name == "python":
            return {"code": args}
        if name == "web_search":
            return {"query": args}
        if name == "context7_search":
            return {"query": args}
    return {}


@dataclass
class StreamSink:
    """Terminal stream callbacks (tokens + tool-call notices, not tool output)."""

    on_token: Callable[[str], None] | None = None
    on_tool_call: Callable[[str], None] | None = None


def format_tool_call_notice(name: str, args: dict[str, Any]) -> str:
    if name == "bash":
        command = str(args.get("command") or "").strip()
        preview = command if len(command) <= 80 else command[:77] + "..."
        return f"\n[bash] {preview}\n"
    if name == "python":
        code = str(args.get("code") or "").strip()
        first = code.splitlines()[0] if code else ""
        preview = first if len(first) <= 80 else first[:77] + "..."
        return f"\n[python] {preview}\n"
    if name == "web_search":
        query = str(args.get("query") or "").strip()
        return f"\n[web_search] {query}\n"
    if name == "context7_search":
        query = str(args.get("query") or "").strip()
        library = str(args.get("library") or "").strip()
        suffix = f" ({library})" if library else ""
        return f"\n[context7_search] {query}{suffix}\n"
    if name == RESPOND_DIRECTLY_NAME:
        return "\n[respond_directly]\n"
    return f"\n[{name}]\n"


def _emit_tool_call_notices(
    tool_calls: list[Any],
    sink: StreamSink | None,
    *,
    announced: set[str] | None = None,
) -> None:
    if sink is None or sink.on_tool_call is None or not tool_calls:
        return
    seen = announced if announced is not None else set()
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or name)
        full_key = f"full:{call_id}"
        if full_key in seen:
            continue
        seen.add(full_key)
        args = _parse_tool_args(name, call.get("args"))
        sink.on_tool_call(format_tool_call_notice(name, args))


def _tool_call_chunks(chunk: AnyMessage) -> list[Any]:
    chunks = getattr(chunk, "tool_call_chunks", None)
    return list(chunks) if chunks else []


def call_llm(
    messages: list[AnyMessage],
    *,
    sink: StreamSink | None = None,
    forced: bool | None = None,
) -> AIMessage:
    mlx_messages = messages_for_mlx(messages)
    llm = llm_for_turn(messages, forced=forced)
    on_token = sink.on_token if sink else None
    use_stream = not (
        on_token is None and (sink is None or sink.on_tool_call is None)
    )
    mode = "stream" if use_stream else "buffered"

    if not use_stream:
        response = llm.invoke(mlx_messages)
        result = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
        _emit_tool_call_notices(list(result.tool_calls or []), sink)
        log_llm_exchange(mlx_messages, result, mode=mode)
        return result

    announced: set[str] = set()
    merged = None
    for chunk in llm.stream(mlx_messages):
        piece = chunk_to_text(chunk)
        if piece and on_token:
            on_token(piece)
        for tcc in _tool_call_chunks(chunk):
            if not isinstance(tcc, dict):
                continue
            name = tcc.get("name")
            if not name:
                continue
            idx = str(tcc.get("index", 0))
            key = f"chunk:{idx}:{name}"
            if key in announced:
                continue
            announced.add(key)
            if sink and sink.on_tool_call:
                sink.on_tool_call(f"\n[{name}] …\n")
        merged = chunk if merged is None else merged + chunk
    if merged is None:
        result = AIMessage(content="")
    elif isinstance(merged, AIMessage):
        result = merged
    else:
        tool_calls = list(merged.tool_calls) if getattr(merged, "tool_calls", None) else []
        content = merged.content if isinstance(merged.content, str) else str(merged.content or "")
        result = AIMessage(content=content, tool_calls=tool_calls)
    _emit_tool_call_notices(list(result.tool_calls or []), sink, announced=announced)
    log_llm_exchange(mlx_messages, result, mode=mode)
    return result


def make_agent_node(sink: StreamSink | None = None):
    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        first = call_llm(messages, sink=sink)
        if is_respond_directly_call(list(first.tool_calls or [])):
            final = call_llm(messages, sink=sink, forced=False)
            return {"messages": [final]}
        return {"messages": [first]}

    return agent_node


def tools_node(state: AgentState) -> dict:
    """마지막 AIMessage 의 tool_calls 를 로컬 실행 → ToolMessage 로 회신."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": []}

    out: list[ToolMessage] = []
    for call in last.tool_calls:
        name = str(call.get("name") or "")
        if name == RESPOND_DIRECTLY_NAME:
            continue
        call_id = str(call.get("id") or "")
        args = _parse_tool_args(name, call.get("args"))
        if name not in TOOLS:
            content = f"[ERROR] unknown tool: {name}"
        else:
            content = TOOLS[name].invoke(args)
        out.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
    return {"messages": out}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(sink: StreamSink | None = None):
    g = StateGraph(AgentState)
    g.add_node("agent", make_agent_node(sink))
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

SYSTEM_PROMPT_BODY = """You have: bash, python, web_search, context7_search.

- Training data is stale for present-day facts → web_search first.
- Library/API docs → context7_search, not web_search.
- Tool output overrides memory and prior assistant messages.
- Reply in the user's language. Be concise; don't repeat earlier answers.
- Each user message: pick one tool first (respond_directly only if no tool is needed).
"""


def build_system_prompt(*, now: datetime | None = None) -> str:
    """System prompt with live clock and tool-authority rules."""
    clock = (now or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
    return (
        f"Today's date and time (system clock): {clock}.\n"
        f"{SYSTEM_PROMPT_BODY}"
    )


def refresh_system_message(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Replace or prepend the system message with a fresh clock."""
    from langchain_core.messages import SystemMessage

    prompt = SystemMessage(content=build_system_prompt())
    if messages and isinstance(messages[0], SystemMessage):
        return [prompt, *messages[1:]]
    return [prompt, *messages]


# Backward-compatible name for tests/docs
SYSTEM_PROMPT = SYSTEM_PROMPT_BODY


def main(argv: list[str] | None = None) -> int:
    from langchain_core.messages import HumanMessage, SystemMessage

    parser = argparse.ArgumentParser(description="LangGraph coding agent (MLX stream)")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for full JSON response (stream: false)",
    )
    args = parser.parse_args(argv)
    use_stream = stream_enabled(explicit=not args.no_stream)

    base_url = llm_base_url()
    api_key = llm_api_key()
    try:
        model_id = effective_model_id(base_url, api_key)
    except RuntimeError as exc:
        print(f"모델 확인 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    exa_key = resolve_exa_api_key()
    set_exa_api_key(exa_key)
    c7_key = resolve_context7_api_key()
    set_context7_api_key(c7_key)
    search_note = "web_search ON" if exa_key else "web_search OFF (no EXA_API_KEY / exa.api_key)"
    c7_note = (
        "context7_search ON"
        if c7_key
        else "context7_search OFF (no CONTEXT7_API_KEY / context7.api_key)"
    )
    fc_note = (
        "forced_choice ON"
        if force_choice_enabled()
        else "forced_choice OFF (SMALL_AGENT_FORCE_CHOICE=0)"
    )
    mode = "stream" if use_stream else "buffered"

    streamed_any = False

    def on_token(piece: str) -> None:
        nonlocal streamed_any
        streamed_any = True
        print(piece, end="", flush=True)

    def on_tool_call(line: str) -> None:
        nonlocal streamed_any
        streamed_any = True
        print(line, end="", flush=True)

    sink = (
        StreamSink(on_token=on_token, on_tool_call=on_tool_call)
        if use_stream
        else None
    )
    graph = build_graph(sink=sink)
    log_note = ""
    if llm_log_enabled():
        log_path = session_log_path()
        if log_path is not None:
            log_note = f", llm log → {log_path}"
    print(
        f"agent ready (model={model_id}, {mode}, {fc_note}, {search_note}, {c7_note}{log_note}, /quit to exit)"
    )
    history: list[AnyMessage] = refresh_system_message([])
    try:
        while True:
            try:
                user = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user or user in ("/quit", "/exit"):
                break
            history = refresh_system_message(history)
            history.append(HumanMessage(content=user))
            streamed_any = False
            try:
                if use_stream:
                    print("\nBot> ", end="", flush=True)
                result = graph.invoke(
                    {"messages": history},
                    config={"recursion_limit": 25},
                )
            except Exception as exc:
                print(f"\n오류: {exc}", file=sys.stderr)
                continue
            history = list(result["messages"])
            if use_stream:
                if not streamed_any:
                    content = final_assistant_text(history)
                    if content:
                        print(content, end="")
                print()
            else:
                content = final_assistant_text(history)
                if content:
                    print(f"\nBot> {content}")
                else:
                    print("\nBot> (no reply)")
    finally:
        _BASH.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
