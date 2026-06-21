from unittest.mock import patch

import context7_search


def test_run_context7_search_without_key():
    assert "no API key" in context7_search.run_context7_search(None, "routing")


def test_run_context7_search_with_mock():
    libs = [{"libraryId": "fastapi", "name": "FastAPI"}]
    with (
        patch.object(context7_search, "search_library", return_value=libs),
        patch.object(context7_search, "fetch_context", return_value="Use APIRouter for routes."),
    ):
        text = context7_search.run_context7_search("key", "routing", "fastapi")
    assert "FastAPI" in text
    assert "APIRouter" in text


def test_run_context7_search_no_libraries():
    with patch.object(context7_search, "search_library", return_value=[]):
        text = context7_search.run_context7_search("key", "routing", "unknown-lib")
    assert "no libraries matched" in text


def test_resolve_context7_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("CONTEXT7_API_KEY", "env-key")
    assert context7_search.resolve_context7_api_key() == "env-key"
