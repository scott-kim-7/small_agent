from unittest.mock import patch

import exa_search


def test_run_web_search_without_key():
    assert "no API key" in exa_search.run_web_search(None, "news today")


def test_format_empty_results():
    text = exa_search.format_search_results("q", [])
    assert "no results" in text


def test_run_web_search_with_mock():
    hits = [{"title": "News", "url": "https://ex.example", "text": "Body"}]
    with patch.object(exa_search, "search_exa", return_value=hits):
        text = exa_search.run_web_search("key", "news", 3)
    assert "News" in text
    assert "https://ex.example" in text


def test_resolve_exa_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "env-key")
    assert exa_search.resolve_exa_api_key() == "env-key"
