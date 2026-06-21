import pytest

from bash_session import BashSession


@pytest.fixture
def bash():
    session = BashSession(timeout=2.0)
    yield session
    session.close()


def test_cd_persists(bash):
    assert "/tmp" in bash.run("cd /tmp && pwd").to_llm_string()
    assert "/tmp" in bash.run("pwd").to_llm_string()


def test_env_var_persists(bash):
    bash.run("export FOO=bar")
    assert "bar" in bash.run("echo $FOO").to_llm_string()


def test_nonzero_exit_code(bash):
    text = bash.run("ls /nonexistent_xyz").to_llm_string()
    assert "exit code:" in text
    assert "No such file" in text


def test_timeout_and_cwd_recovery(bash):
    bash.run("cd /tmp && pwd")
    assert bash.run("sleep 10").timed_out
    assert "/tmp" in bash.run("pwd").to_llm_string()


def test_infinite_loop_recovery(bash):
    assert bash.run("while true; do :; done").timed_out
    assert "alive" in bash.run("echo alive").to_llm_string()
