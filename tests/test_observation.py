from observation import ObsKind, Observation


def test_success_is_quiet():
    obs = Observation(ObsKind.CMD, "echo hi", stdout="hi\n", exit_code=0)
    assert obs.to_llm_string() == "hi"
    assert obs.success


def test_failure_highlights_exit_code():
    obs = Observation(ObsKind.CMD, "false", exit_code=1)
    assert "exit code: 1" in obs.to_llm_string()
    assert not obs.success


def test_execution_error():
    obs = Observation(
        ObsKind.CMD,
        "sleep 10",
        timed_out=True,
        error="command exceeded 30s timeout",
    )
    text = obs.to_llm_string()
    assert "[EXECUTION ERROR]" in text
    assert "timed out" in text


def test_truncation():
    obs = Observation(ObsKind.CMD, "big", stdout="x" * 9000, exit_code=0)
    text = obs.to_llm_string()
    assert "truncated" in text
    assert obs.truncated
