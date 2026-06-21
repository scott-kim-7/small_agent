"""Context7 library documentation search — httpx only, no ada import required."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

CONTEXT7_API_BASE = "https://context7.com/api/v2"
CONTEXT7_VAULT_KEY = "context7.api_key"


def context7_key_from_env() -> str | None:
    key = os.environ.get("CONTEXT7_API_KEY", "").strip()
    return key or None


def context7_key_from_vault() -> str | None:
    try:
        from ada.vault import VaultError
        from ada.vault_secrets import resolve_vault_secret
        from ada.vault_unlock import bootstrap_vault_session
    except ImportError:
        return None
    try:
        session = bootstrap_vault_session()
        return resolve_vault_secret(CONTEXT7_VAULT_KEY, session)
    except VaultError:
        return None


def resolve_context7_api_key() -> str | None:
    return context7_key_from_env() or context7_key_from_vault()


def parse_libs_search_response(data: dict[str, Any]) -> list[dict[str, str]]:
    items = data.get("results") or data.get("libraries") or data.get("data") or []
    out: list[dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        lib_id = str(item.get("libraryId") or item.get("id") or item.get("library_id") or "")
        name = str(item.get("name") or item.get("libraryName") or item.get("title") or "")
        if lib_id or name:
            out.append({"libraryId": lib_id, "name": name})
    return out


def parse_context_response(data: dict[str, Any]) -> str:
    for key in ("context", "content", "text", "data"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = value.get("content") or value.get("text")
            if isinstance(nested, str) and nested.strip():
                return nested
    return json.dumps(data, ensure_ascii=False)[:8000]


def search_library(api_key: str, library_name: str, query: str) -> list[dict[str, str]]:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"libraryName": library_name, "query": query}
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        resp = client.get(f"{CONTEXT7_API_BASE}/libs/search", params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        return []
    return parse_libs_search_response(data)


def fetch_context(api_key: str, library_id: str, query: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"libraryId": library_id, "query": query, "type": "json"}
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        resp = client.get(f"{CONTEXT7_API_BASE}/context", params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        return ""
    return parse_context_response(data)


def format_context7_result(
    library_name: str,
    library_id: str,
    query: str,
    context_text: str,
) -> str:
    body = context_text if len(context_text) <= 8000 else context_text[:8000] + "..."
    lines = [
        f'Context7 documentation for "{library_name}" (query: "{query}")',
        f"libraryId: {library_id}",
        "",
        body,
    ]
    return "\n".join(lines).strip()


def run_context7_search(
    api_key: str | None,
    query: str,
    library: str = "",
) -> str:
    query = query.strip()
    if not query:
        return "[ERROR] context7_search: query is required"
    if not api_key:
        return (
            "[ERROR] context7_search: no API key. "
            "Set CONTEXT7_API_KEY or store context7.api_key in ada vault."
        )

    library_name = library.strip() or query
    try:
        libs = search_library(api_key, library_name, query)
    except httpx.HTTPError as exc:
        return f"[ERROR] context7_search failed: {exc}"

    if not libs:
        return f'(no libraries matched library="{library_name}" query="{query}")'

    first = libs[0]
    lib_id = first.get("libraryId") or first.get("name") or ""
    lib_display = first.get("name") or lib_id
    if not lib_id:
        return f'(no library id for "{library_name}" / "{query}")'

    try:
        context_text = fetch_context(api_key, lib_id, query)
    except httpx.HTTPError as exc:
        return f"[ERROR] context7_search context fetch failed: {exc}"

    if not context_text.strip():
        return f'(no documentation context for {lib_display} / "{query}")'

    return format_context7_result(lib_display, lib_id, query, context_text)
