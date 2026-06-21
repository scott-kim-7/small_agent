"""Stateful bash 세션 (hard-reset 복구 전략).

하나의 bash 프로세스를 띄워 명령을 stdin 으로 주입하고, 고유 센티넬로
출력 경계 + exit code + cwd 를 회수한다. cd/export/venv 상태가 명령 간 유지된다.

타임아웃 처리:
  실행 중 명령이 시간 초과하면, 프로세스 그룹을 통째로 SIGKILL 하고
  마지막으로 알려진 cwd 에서 새 셸을 재기동한다 (hard reset).
  - 장점: hang/무한루프/시그널 무시 자식에도 100% 확실히 복구된다.
  - 한계: 그 시점의 셸-로컬 상태(export 등)는 초기화된다. cwd 는 보존.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
import uuid

from observation import ObsKind, Observation

DEFAULT_TIMEOUT = 30.0


class BashSession:
    def __init__(self, work_dir: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._cwd = work_dir or os.getcwd()
        self._sentinel = f"__CMD_DONE_{uuid.uuid4().hex}__"
        self._proc: subprocess.Popen | None = None
        self._spawn(self._cwd)

    def _spawn(self, cwd: str) -> None:
        env = os.environ.copy()
        env["PS1"] = ""
        env["TERM"] = "dumb"
        self._proc = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        marker = f"__READY_{uuid.uuid4().hex}__"
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(f"echo {marker}\n")
        self._proc.stdin.flush()
        deadline = time.time() + 5
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if line == "" or marker in line:
                break

    def _hard_reset(self) -> None:
        """프로세스 그룹 전체를 죽이고 마지막 cwd 에서 새 셸 기동."""
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._spawn(self._cwd)

    def run(self, command: str) -> Observation:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn(self._cwd)

        start = time.time()
        full = (
            f"{command}\n"
            f'echo "{self._sentinel}:$?:$(pwd)"\n'
        )
        try:
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(full)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return Observation(
                kind=ObsKind.CMD,
                command=command,
                error="failed to write to bash (broken pipe)",
            )

        collected: list[str] = []
        exit_code: int | None = None
        cwd: str | None = None
        timed_out = False
        deadline = start + self.timeout

        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                timed_out = True
                break
            line = self._proc.stdout.readline()
            if line == "":
                break
            if self._sentinel in line:
                try:
                    _, ec_str, cwd_str = line.strip().split(":", 2)
                    exit_code = int(ec_str)
                    cwd = cwd_str
                    self._cwd = cwd_str
                except ValueError:
                    pass
                break
            collected.append(line)

        if timed_out:
            self._hard_reset()
            return Observation(
                kind=ObsKind.CMD,
                command=command,
                stdout="".join(collected),
                cwd=self._cwd,
                timed_out=True,
                duration_s=time.time() - start,
                error=(
                    f"command exceeded {self.timeout}s timeout. "
                    f"Shell was reset (cwd preserved: {self._cwd}; "
                    f"shell-local env/vars cleared)."
                ),
            )

        return Observation(
            kind=ObsKind.CMD,
            command=command,
            stdout="".join(collected),
            exit_code=exit_code,
            cwd=cwd,
            duration_s=time.time() - start,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write("exit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except Exception:
                self._proc.kill()
