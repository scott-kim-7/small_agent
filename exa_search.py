"""Exa web search — httpx only, no ada import required."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

EXA_API_BASE = "https://api.exa.ai"
EXA_VAULT_KEY = "exa.api_key"


def exa_key_from_env() -> str | None:
    key = os.environ.get("EXA_API_KEY", "").strip()
    return key or None


def exa_key_from_vault() -> str | None:
    try:
        from ada.vault import VaultError
        from ada.vault_secrets import resolve_vault_secret
        from ada.vault_unlock import bootstrap_vault_session
    except ImportError:
        return None
    try:
        session = bootstrap_vault_session()
        return resolve_vault_secret(EXA_VAULT_KEY, session)
    except VaultError:
        return None


def resolve_exa_api_key() -> str | None:
    return exa_key_from_env() or exa_key_from_vault()


def search_exa(api_key: str, query: str, count: int = 5) -> list[dict[str, str]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "query": query,
        "numResults": count or 5,
        "contents": {"text": True, "highlights": True},
        "type": "auto",
    }
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        resp = client.post(f"{EXA_API_BASE}/search", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results: list[dict[str, str]] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        link = str(item.get("url") or "").strip()
        body = str(item.get("text") or "").strip()
        if title or body or link:
            results.append({"title": title, "url": link, "text": body})
    return results


def format_search_results(query: str, hits: list[dict[str, str]]) -> str:
    if not hits:
        return f'(no results for query: "{query}")'
    lines = [f'Search results for: "{query}"', ""]
    for index, hit in enumerate(hits, start=1):
        title = hit.get("title") or "(no title)"
        url = hit.get("url") or ""
        text = hit.get("text") or ""
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   {url}")
        if text:
            snippet = text if len(text) <= 500 else text[:500] + "..."
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def run_web_search(api_key: str | None, query: str, max_results: int = 5) -> str:
    query = query.strip()
    if not query:
        return "[ERROR] web_search: query is required"
    if not api_key:
        return (
            "[ERROR] web_search: no API key. Set EXA_API_KEY or store exa.api_key in ada vault."
        )
    try:
        hits = search_exa(api_key, query, max_results)
    except httpx.HTTPError as exc:
        return f"[ERROR] web_search failed: {exc}"
    return format_search_results(query, hits)


def run_web_search_json(api_key: str | None, arguments: dict[str, Any]) -> str:
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results") or 5)
    return run_web_search(api_key, query, max_results)
