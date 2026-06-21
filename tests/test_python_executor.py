from python_executor import PythonExecutor


def test_print_output():
    executor = PythonExecutor(timeout=3.0)
    assert "1024" in executor.run("print(2**10)").to_llm_string()


def test_stderr_on_exception():
    executor = PythonExecutor(timeout=3.0)
    assert "ZeroDivisionError" in executor.run("x = 1/0").to_llm_string()


def test_timeout():
    executor = PythonExecutor(timeout=1.0)
    obs = executor.run("import time; time.sleep(10)")
    assert obs.timed_out
