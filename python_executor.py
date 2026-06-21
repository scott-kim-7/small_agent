"""Stateless Python executor.

매 호출마다 임시 .py 파일에 코드를 쓰고 별도 subprocess(python3)로 실행한다.
세션 상태(변수)는 유지되지 않지만, 격리가 깔끔하고 타임아웃/크래시가
메인 에이전트 프로세스에 영향을 주지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

from observation import ObsKind, Observation

DEFAULT_TIMEOUT = 30.0


class PythonExecutor:
    def __init__(self, work_dir: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.work_dir = work_dir or os.getcwd()

    def run(self, code: str) -> Observation:
        start = time.time()
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=self.work_dir,
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            try:
                proc = subprocess.run(
                    [sys.executable, "-u", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.work_dir,
                )
            except subprocess.TimeoutExpired as e:
                return Observation(
                    kind=ObsKind.PYTHON,
                    command=code,
                    stdout=e.stdout or "",
                    stderr=e.stderr or "",
                    timed_out=True,
                    duration_s=time.time() - start,
                    error=f"python execution exceeded {self.timeout}s timeout",
                )

            return Observation(
                kind=ObsKind.PYTHON,
                command=code,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                cwd=self.work_dir,
                duration_s=time.time() - start,
            )
        except Exception as e:
            return Observation(
                kind=ObsKind.PYTHON,
                command=code,
                error=f"failed to execute python: {type(e).__name__}: {e}",
                duration_s=time.time() - start,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
