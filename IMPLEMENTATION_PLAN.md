# LangGraph 코딩 에이전트 구현 계획서
### OpenHands 스타일 Observation을 갖춘 bash/python 도구 에이전트

> **목적**: 이 문서 하나만 보고 빈 디렉터리에서 시작해 동일한 에이전트를 완성할 수 있도록
> 모든 코드, 설계 의도, 시행착오, 검증 방법을 빠짐없이 기술한다.
>
> **독자**: Python·LLM 에이전트에 익숙하지만 이 코드베이스는 처음 보는 개발자.

---

## 목차

1. [무엇을 만드는가](#1-무엇을-만드는가)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [사전 요구사항](#3-사전-요구사항)
4. [디렉터리 구조](#4-디렉터리-구조)
5. [모듈 1: Observation (관찰 결과 구조화)](#5-모듈-1-observation)
6. [모듈 2: BashSession (stateful 셸)](#6-모듈-2-bashsession)
7. [모듈 3: PythonExecutor (stateless 실행기)](#7-모듈-3-pythonexecutor)
8. [모듈 4: LangGraph 본체](#8-모듈-4-langgraph-본체)
9. [구현 순서와 단계별 검증](#9-구현-순서와-단계별-검증)
10. [내가 실제로 겪은 함정들](#10-내가-실제로-겪은-함정들)
11. [확장 가이드](#11-확장-가이드)
12. [부록: 전체 소스](#12-부록-전체-소스)

---

## 1. 무엇을 만드는가

LLM이 `bash`와 `python` 두 도구를 호출하며 코딩 작업을 수행하는 최소 ReAct 에이전트다.
핵심 차별점은 **도구 실행 결과(Observation)를 OpenHands 수준으로 구조화**한다는 점이다.

단순한 에이전트는 명령의 stdout 문자열만 LLM에게 돌려준다. 그러면 LLM은 명령이
성공했는지 실패했는지, 왜 실패했는지, 출력이 잘렸는지를 알 수 없다. 이 구현은
exit code, stderr, timeout 여부, 출력 truncation, 현재 작업 디렉터리(cwd)를 모두
캡처해서 LLM이 다음 행동을 정확히 판단할 수 있게 한다.

### 설계 결정 (이 문서의 전제)

| 항목 | 선택 | 이유 |
|------|------|------|
| bash 실행 | **Stateful** (한 셸 세션 유지) | `cd`, `export`, venv activate가 명령 간 지속되어야 실제 코딩 작업이 자연스럽다 |
| python 실행 | **Stateless** (매번 임시 .py를 subprocess로) | 격리가 깔끔하고, 크래시·무한루프가 에이전트 본체에 전파되지 않는다 |
| LLM 백엔드 | **OpenAI 호환 로컬 서버** (litellm/MLX, 포트 8089) | 로컬 모델로 완결. `ChatOpenAI`의 `base_url`만 바꾸면 됨 |
| 도구 실행 위치 | **로컬 (격리 컨테이너 아님)** | 개발 머신에서 직접 실행. 보안 경계는 확장 과제로 분리 |

> **주의**: 이 구현은 도구를 **로컬에서 직접** 실행한다. 신뢰할 수 없는 입력이나
> 프로덕션 환경에서는 반드시 컨테이너/샌드박스 격리를 추가해야 한다(11장 참고).

---

## 2. 전체 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │            LangGraph                 │
                    │                                      │
   User input ──────▶   ┌──────────┐   tool_calls 있음?   │
                    │   │  agent   │──────────┐            │
                    │   │  (LLM)   │◀───┐     │ 예         │
                    │   └──────────┘    │     ▼            │
                    │        │          │  ┌────────┐      │
                    │        │ 없음     │  │ tools  │      │
                    │        ▼          └──│ (로컬  │      │
                    │      END             │ 실행)  │      │
                    │                      └────────┘      │
                    └──────────────────────────│──────────┘
                                               │
                          ┌────────────────────┴────────────────┐
                          ▼                                      ▼
                ┌──────────────────┐                ┌──────────────────┐
                │   BashSession    │                │  PythonExecutor  │
                │  (stateful 셸)   │                │ (stateless .py)  │
                └────────┬─────────┘                └────────┬─────────┘
                         │                                   │
                         └──────────────┬────────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │   Observation    │
                              │ exit/stderr/cwd/ │
                              │ timeout/truncate │
                              └──────────────────┘
```

**실행 흐름 (한 턴)**:
1. 사용자 입력 → `agent` 노드가 LLM 호출
2. LLM이 `tool_calls`를 반환하면 → `tools` 노드로 분기
3. `tools` 노드가 `bash`/`python`을 **로컬에서 직접 실행** → `Observation` 생성
4. `Observation.to_llm_string()`을 `ToolMessage`에 실어 다시 `agent`로
5. LLM이 결과를 보고 다음 도구를 부르거나, 최종 답변을 내면 → `END`

---

## 3. 사전 요구사항

### 런타임
- **Python 3.10+** (타입 힌트 `X | None` 문법, `start_new_session` 등 사용)
- **OS**: Linux 또는 macOS. `os.killpg`, `select.select`, `signal`을 쓰므로 **Windows는 미지원** (Windows는 WSL 사용 권장)
- **/bin/bash** 존재

### 패키지
```bash
pip install langgraph langchain-openai langchain-core
```

검증된 버전대 (2026년 6월 기준): `langchain-core 1.x`, `langgraph 1.x`, `langchain-openai 0.3.x`.

> **버전 주의**: LangChain/LangGraph 1.0에서 `langgraph.prebuilt`의 일부가
> `langchain.agents`로 이동했고 `AgentExecutor`는 deprecated다. 이 구현은
> prebuilt에 의존하지 않고 그래프를 직접 조립하므로 영향받지 않지만, 옛날
> 튜토리얼 코드를 섞지 말 것.

### LLM 백엔드
OpenAI 호환 엔드포인트가 떠 있어야 한다. 예: litellm proxy 또는 MLX 서버가
`http://127.0.0.1:8089/v1`에서 `/chat/completions`를 제공.

**모델은 native tool calling(function calling)을 지원해야 한다.** 지원하지 않으면
`tool_calls`가 비어 오므로 도구가 호출되지 않는다. 우회책은 10장과 11장 참고.

---

## 4. 디렉터리 구조

```
agent/
├── observation.py       # Observation 데이터 구조 (의존성 없음, 가장 먼저)
├── bash_session.py      # stateful bash (observation에 의존)
├── python_executor.py   # stateless python (observation에 의존)
├── graph.py             # LangGraph 본체 (위 3개 + langchain에 의존)
└── README.md
```

의존 방향: `observation ← bash_session, python_executor ← graph`.
**반드시 observation.py부터 만들고 검증한다.** 나머지가 모두 여기에 의존한다.

---

## 5. 모듈 1: Observation

### 역할
도구 실행 결과를 구조화하는 데이터 클래스. raw 데이터(stdout/stderr/exit_code/…)는
객체에 보존하고, LLM에게는 `to_llm_string()`으로 **신호가 되는 것만** 골라 포맷한다.

### 설계 포인트 (반드시 이해할 것)

1. **성공은 조용히, 실패는 강조**: exit code가 0이면 상태줄을 출력하지 않는다.
   0이 아닐 때만 `[exit code: N]`을 붙여 LLM의 주의를 끈다. 매 명령마다 "exit code: 0"을
   붙이면 컨텍스트만 낭비하고 신호가 묻힌다.

2. **truncation은 head+tail 보존**: 출력이 길면 앞 5000자 + 뒤 2000자만 남기고
   가운데를 잘라낸다. 에러는 보통 끝(traceback)에 있고 맥락은 앞에 있으므로 양쪽을 보존한다.
   앞만 자르면 정작 중요한 에러 메시지가 사라진다.

3. **error vs exit_code 구분**: `error` 필드는 **실행 인프라 자체의 실패**
   (타임아웃, 셸 죽음, 파일 쓰기 실패)를 의미한다. 명령이 정상 실행됐지만 0이 아닌
   코드로 끝난 것(`exit_code != 0`)과는 다르다. LLM에게 "명령이 실패함"과
   "명령을 실행조차 못 함"은 다른 신호다.

### 전체 코드: `observation.py`

```python
"""구조화된 Observation — OpenHands CmdOutputObservation 설계 참고.

핵심 아이디어:
- 단순 stdout 문자열이 아니라 exit_code / stderr / truncation / timeout /
  cwd / 에러 종류를 모두 보존한다.
- LLM에게 넘길 때는 `to_llm_string()`으로 '필요한 것만' 포맷한다.
  (전체 raw 데이터는 로깅/디버깅용으로 객체에 남겨둔다)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObsKind(str, Enum):
    CMD = "command"        # bash 실행 결과
    PYTHON = "python"      # python 실행 결과
    ERROR = "error"        # 실행 자체가 실패 (타임아웃, 세션 죽음 등)


# LLM 컨텍스트 보호: 출력이 너무 길면 가운데를 잘라낸다 (head/tail 보존).
MAX_OUTPUT_CHARS = 8000
HEAD_CHARS = 5000
TAIL_CHARS = 2000


def _truncate(text: str) -> tuple[str, bool]:
    """가운데를 잘라 head+tail만 남긴다. (잘렸으면 True)"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    marker = f"\n\n[... {omitted} chars truncated ...]\n\n"
    return head + marker + tail, True


@dataclass
class Observation:
    kind: ObsKind
    command: str                       # 실행한 명령/코드 (에코백용)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None       # None = 정상 종료 코드를 못 얻음
    cwd: str | None = None             # 명령 실행 후 현재 작업 디렉터리
    timed_out: bool = False
    truncated: bool = False            # 출력이 잘렸는지
    duration_s: float | None = None
    error: str | None = None           # 실행 인프라 자체의 에러 메시지
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_llm_string(self) -> str:
        """LLM 컨텍스트에 넣을 포맷. 신호가 되는 것만 담는다."""
        lines: list[str] = []

        if self.error is not None:
            lines.append(f"[EXECUTION ERROR] {self.error}")
            if self.timed_out:
                lines.append("(command timed out)")
            return "\n".join(lines)

        out, out_trunc = _truncate(self.stdout)
        err, err_trunc = _truncate(self.stderr)
        self.truncated = out_trunc or err_trunc

        if out.strip():
            lines.append(out.rstrip("\n"))
        if err.strip():
            lines.append(f"[stderr]\n{err.rstrip(chr(10))}")
        if not out.strip() and not err.strip():
            lines.append("(no output)")

        # 상태 줄: exit code가 0이 아닐 때만 강조 (성공은 조용히)
        status_bits: list[str] = []
        if self.exit_code is not None and self.exit_code != 0:
            status_bits.append(f"exit code: {self.exit_code}")
        if self.timed_out:
            status_bits.append("TIMED OUT")
        if self.truncated:
            status_bits.append("output truncated")
        if self.cwd:
            status_bits.append(f"cwd: {self.cwd}")
        if status_bits:
            lines.append(f"\n[{' | '.join(status_bits)}]")

        return "\n".join(lines).strip()

    def short_summary(self) -> str:
        """로그용 한 줄 요약."""
        st = "ok" if self.success else "FAIL"
        ec = self.exit_code if self.exit_code is not None else "?"
        return f"<{self.kind.value} {st} exit={ec} out={len(self.stdout)}b err={len(self.stderr)}b>"
```

### 단위 검증
```python
from observation import Observation, ObsKind
# 성공: 상태줄 없이 출력만
o = Observation(ObsKind.CMD, "echo hi", stdout="hi\n", exit_code=0)
assert o.to_llm_string() == "hi"
assert o.success
# 실패: exit code 강조됨
o = Observation(ObsKind.CMD, "false", exit_code=1)
assert "exit code: 1" in o.to_llm_string()
assert not o.success
print("observation OK")
```

---

## 6. 모듈 2: BashSession

### 역할
하나의 살아있는 bash 프로세스를 유지하며 명령을 주입한다. `cd`/`export`가
명령 간 지속되는 **stateful** 셸.

### 핵심 메커니즘 3가지

#### (1) 센티넬 마커로 출력 경계 + exit code + cwd 회수

bash는 명령이 언제 끝났는지 stdout만 봐서는 알 수 없다(다음 프롬프트가 안 오니까).
그래서 명령 뒤에 **고유 마커를 echo**하는 명령을 붙여 보낸다:

```
<사용자 명령>
echo "__CMD_DONE_<uuid>:$?:$(pwd)"
```

출력을 한 줄씩 읽다가 그 마커 라인이 나오면 명령이 끝난 것이다. 그 라인에서
`$?`(직전 명령의 exit code)와 `$(pwd)`(현재 디렉터리)를 함께 회수한다.
마커에 uuid를 넣는 이유: 사용자 명령의 출력에 우연히 같은 문자열이 섞여
오탐하는 것을 방지.

#### (2) PS1/TERM 정리로 출력 오염 방지

- `PS1=""`: 프롬프트 문자열이 출력에 섞이지 않도록 제거.
- `TERM=dumb`: ANSI 컬러/이스케이프 시퀀스 억제.
- `--norc --noprofile`: 사용자 rc 파일이 출력을 더럽히지 않도록.
- **`-i`(interactive)를 쓰지 말 것**: interactive 셸은 입력 명령을 그대로
  에코백해서 출력에 섞인다. (10장 함정 1 참고)

#### (3) 타임아웃 = hard reset (가장 중요)

명령이 타임아웃되면 **프로세스 그룹 전체를 SIGKILL하고 새 셸을 띄운다.**
이때 마지막으로 알려진 cwd를 복원한다.

왜 graceful interrupt(SIGINT)가 아니라 SIGKILL + 재기동인가? → 10장 함정 2~4에서
상세히 다룬다. 요약하면: SIGINT는 시그널을 무시하는 자식(예: 일부 `sleep` 구현,
trap 건 스크립트)에 무력하고, 셸 자신을 죽이거나 잔여 출력이 다음 명령을 오염시키는
부작용이 크다. OpenHands도 hang 상태에는 결국 프로세스 그룹 강제 종료를 쓴다.

`start_new_session=True`로 bash를 **독립 프로세스 그룹의 리더**로 만들어야
`os.killpg`로 그 그룹(bash + 모든 자식)을 한 번에 죽일 수 있다.

**트레이드오프**: 재기동하면 그 시점의 셸-로컬 상태(export 변수 등)는 사라진다.
cwd만 보존된다. 이 사실을 Observation의 `error` 메시지에 명시해서 LLM이 알게 한다.

### blocking 회피: select 사용

`readline()`은 블로킹이다. 타임아웃을 걸려면 `select.select([stdout], [], [], remaining)`로
**남은 시간 동안만** 읽기 가능 여부를 기다린 뒤, 가능할 때만 `readline()`을 호출한다.
select가 빈 리스트를 반환하면(타임아웃) 루프를 빠져나온다.

### 전체 코드: `bash_session.py`

```python
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

    # ── 셸 생애주기 ──────────────────────────────────────────────────
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
            start_new_session=True,   # 독립 프로세스 그룹 → killpg 가능
        )
        # 초기 출력 비우기
        marker = f"__READY_{uuid.uuid4().hex}__"
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

    # ── 명령 실행 ────────────────────────────────────────────────────
    def run(self, command: str) -> Observation:
        if self._proc is None or self._proc.poll() is not None:
            # 죽어있으면 자동 복구
            self._spawn(self._cwd)

        start = time.time()
        full = (
            f"{command}\n"
            f'echo "{self._sentinel}:$?:$(pwd)"\n'
        )
        try:
            self._proc.stdin.write(full)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return Observation(
                kind=ObsKind.CMD, command=command,
                error="failed to write to bash (broken pipe)",
            )

        collected: list[str] = []
        exit_code: int | None = None
        cwd: str | None = None
        timed_out = False
        deadline = start + self.timeout

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
                    self._cwd = cwd_str   # 다음 hard-reset 시 복원할 cwd 기억
                except ValueError:
                    pass
                break
            collected.append(line)

        if timed_out:
            self._hard_reset()
            return Observation(
                kind=ObsKind.CMD, command=command,
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
            kind=ObsKind.CMD, command=command,
            stdout="".join(collected),
            exit_code=exit_code,
            cwd=cwd,
            duration_s=time.time() - start,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write("exit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except Exception:
                self._proc.kill()
```

### 단위 검증 (반드시 모두 통과해야 다음 모듈로)
```python
from bash_session import BashSession
b = BashSession(timeout=2.0)

# 1. cd 지속성
assert "/tmp" in b.run("cd /tmp && pwd").to_llm_string()
assert "/tmp" in b.run("pwd").to_llm_string()        # 다음 명령에서도 유지!

# 2. env var 지속성
b.run("export FOO=bar")
assert "bar" in b.run("echo $FOO").to_llm_string()

# 3. 실패 exit code
assert "exit code: 2" in b.run("ls /nonexistent_xyz").to_llm_string()

# 4. 타임아웃 + hard reset
assert b.run("sleep 10").timed_out

# 5. 복구 후 cwd 보존 (여전히 /tmp)
assert "/tmp" in b.run("pwd").to_llm_string()

# 6. 무한루프도 복구
assert b.run("while true; do :; done").timed_out
assert "alive" in b.run("echo alive").to_llm_string()

b.close()
print("bash_session OK")
```

---

## 7. 모듈 3: PythonExecutor

### 역할
매 호출마다 임시 `.py` 파일에 코드를 쓰고 별도 `subprocess`로 실행한다.
**stateless** — 변수가 호출 간 유지되지 않는다.

### 설계 포인트
- `sys.executable -u <tmp>`: 현재 파이썬 인터프리터로, `-u`(unbuffered)를 줘서
  타임아웃으로 죽어도 그때까지의 출력을 잃지 않는다.
- stdout/stderr를 **분리** 캡처(`capture_output=True`). bash와 달리 python은
  traceback이 stderr로 가므로 분리하는 게 LLM에게 더 명확하다.
- `subprocess.TimeoutExpired`는 부분 출력(`e.stdout`, `e.stderr`)을 들고 있으니
  버리지 말고 Observation에 담는다.
- `finally`에서 임시 파일을 반드시 삭제(`os.unlink`). `delete=False`로 만든 뒤
  수동 삭제하는 이유: subprocess가 파일을 읽는 동안 열려 있어야 하는데, Windows
  호환을 위해서도 닫은 뒤 실행하는 패턴이 안전하다.

### 전체 코드: `python_executor.py`

```python
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
                mode="w", suffix=".py", dir=self.work_dir,
                delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            try:
                proc = subprocess.run(
                    [sys.executable, "-u", tmp_path],   # -u: 출력 버퍼링 끄기
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.work_dir,
                )
            except subprocess.TimeoutExpired as e:
                return Observation(
                    kind=ObsKind.PYTHON, command=code,
                    stdout=e.stdout or "",
                    stderr=e.stderr or "",
                    timed_out=True,
                    duration_s=time.time() - start,
                    error=f"python execution exceeded {self.timeout}s timeout",
                )

            return Observation(
                kind=ObsKind.PYTHON, command=code,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                cwd=self.work_dir,
                duration_s=time.time() - start,
            )
        except Exception as e:
            return Observation(
                kind=ObsKind.PYTHON, command=code,
                error=f"failed to execute python: {type(e).__name__}: {e}",
                duration_s=time.time() - start,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
```

### 단위 검증
```python
from python_executor import PythonExecutor
p = PythonExecutor(timeout=3.0)
assert "1024" in p.run("print(2**10)").to_llm_string()
assert "ZeroDivisionError" in p.run("x = 1/0").to_llm_string()   # stderr 캡처
assert p.run("import time; time.sleep(10)").timed_out
print("python_executor OK")
```

---

## 8. 모듈 4: LangGraph 본체

### 역할
`agent`(LLM) ↔ `tools`(로컬 실행) 순환 그래프를 조립한다.

### 핵심 포인트

#### (1) 전역 실행기 인스턴스
`BashSession`은 **프로세스 전역에서 1개**를 유지해야 한다(상태 지속이 핵심이므로).
노드 함수 안에서 매번 새로 만들면 cd/export가 매번 초기화된다.

#### (2) State 정의와 `add_messages` reducer
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```
`add_messages`는 LangGraph가 제공하는 reducer로, 노드가 반환한 메시지를 기존
리스트에 **append**한다(덮어쓰지 않음). 이것이 대화 누적의 핵심.

#### (3) LLM 바인딩
```python
ChatOpenAI(base_url=..., api_key="local", model=...).bind_tools([bash, python])
```
`bind_tools`가 도구 스키마를 LLM 요청의 `tools` 필드로 변환한다. 로컬 모델이
native function calling을 지원해야 `tool_calls`가 채워져 온다.

#### (4) 방어적 args 파싱 (로컬 모델 필수)
일부 로컬 모델은 `tool_calls`의 `args`를 dict가 아니라 **JSON 문자열**로 보낸다.
또는 아예 깨진 JSON을 보낸다. 그대로 `invoke`하면 터지므로 `tools_node`에서 보정:
```python
if isinstance(args, str):
    try: args = json.loads(args)
    except: args = {"command": args} if name == "bash" else {"code": args}
```

#### (5) recursion_limit
무한 도구 호출 루프를 막기 위해 `graph.invoke(..., config={"recursion_limit": 25})`.
이 횟수를 넘으면 LangGraph가 예외를 던진다.

### 전체 코드: `graph.py`

```python
"""아주 간단한 ReAct 스타일 LangGraph.

구조:
    agent (LLM, litellm/MLX@8089) ──tool_calls?──> tools (로컬 실행) ──> agent
                                  └──없으면────────> END
"""

from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from bash_session import BashSession
from python_executor import PythonExecutor

# ── 로컬 실행기 (프로세스 전역에서 1개씩 유지: bash 는 상태 지속이 핵심) ──
_BASH = BashSession(work_dir=os.getcwd())
_PYTHON = PythonExecutor(work_dir=os.getcwd())


# ── Tools ─────────────────────────────────────────────────────────────
@tool
def bash(command: str) -> str:
    """Run a bash command in a persistent shell session.
    State (cd, environment variables, activated venvs) persists across calls.
    Returns combined stdout/stderr plus exit code and current working dir."""
    obs = _BASH.run(command)
    return obs.to_llm_string()


@tool
def python(code: str) -> str:
    """Execute a Python code snippet in an isolated subprocess.
    Each call runs fresh (no variable persistence between calls).
    Returns stdout, stderr, and exit code."""
    obs = _PYTHON.run(code)
    return obs.to_llm_string()


TOOLS = {"bash": bash, "python": python}


# ── State ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ── LLM (litellm/MLX @ 8089) ──────────────────────────────────────────
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ.get("LITELLM_URL", "http://127.0.0.1:8089/v1"),
        api_key=os.environ.get("LITELLM_KEY", "local"),
        model=os.environ.get("AGENT_MODEL", "qwen3-vl"),
        temperature=0.2,
        timeout=300,
    ).bind_tools(list(TOOLS.values()))


_LLM = build_llm()


# ── Nodes ─────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    response = _LLM.invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """마지막 AIMessage 의 tool_calls 를 로컬 실행 → ToolMessage 로 회신."""
    last = state["messages"][-1]
    out: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        if name not in TOOLS:
            content = f"[ERROR] unknown tool: {name}"
        else:
            # args 가 문자열로 올 때(일부 로컬 모델) 방어적 파싱
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"command": args} if name == "bash" else {"code": args}
            content = TOOLS[name].invoke(args)
        out.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
    return {"messages": out}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ── Graph ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


SYSTEM_PROMPT = (
    "You are a coding agent with two tools: `bash` (persistent shell — cd and "
    "env vars persist) and `python` (isolated, no state between calls). "
    "Use them to accomplish the user's task. Inspect tool output (exit codes, "
    "stderr) before deciding the next step. When done, give a concise final answer."
)


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    graph = build_graph()
    print("agent ready (/quit to exit)")
    history: list[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    try:
        while True:
            try:
                user = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user in ("/quit", "/exit"):
                break
            history.append(HumanMessage(content=user))
            result = graph.invoke(
                {"messages": history},
                config={"recursion_limit": 25},
            )
            history = result["messages"]
            print(f"\nBot> {history[-1].content}")
    finally:
        _BASH.close()


if __name__ == "__main__":
    main()
```

---

## 9. 구현 순서와 단계별 검증

**반드시 이 순서로.** 각 단계의 검증이 통과해야 다음으로 넘어간다. LLM 백엔드 없이도
1~3단계는 완전히 검증 가능하다(순수 표준 라이브러리).

| 단계 | 만들 것 | 검증 | LLM 필요? |
|------|---------|------|-----------|
| 1 | `observation.py` | 5장 단위 검증 | ✗ |
| 2 | `bash_session.py` | 6장 단위 검증 (6개 assert 전부) | ✗ |
| 3 | `python_executor.py` | 7장 단위 검증 | ✗ |
| 4 | `graph.py` | 아래 통합 검증 | ✓ |

### 4단계 통합 검증
```bash
pip install langgraph langchain-openai langchain-core
export LITELLM_URL=http://127.0.0.1:8089/v1
export LITELLM_KEY=local
export AGENT_MODEL=<litellm에 등록된 모델명>
python graph.py
# You> create a file test.txt with "hello", then read it back
# (에이전트가 bash로 echo > / cat 을 호출하고 결과를 보고하면 성공)
```

### LLM 없이 그래프 구조만 검증 (선택)
백엔드가 아직 없으면, `_LLM`을 가짜 객체로 바꿔 tools_node가 Observation을
제대로 도는지만 확인할 수 있다. tool_calls를 수동으로 만든 AIMessage를 넣어
`tools_node`를 직접 호출해 보면 된다.

---

## 10. 내가 실제로 겪은 함정들

> 이 장이 이 문서의 핵심이다. 같은 시행착오를 반복하지 않도록 실패 과정을 그대로 적는다.
> bash 타임아웃 복구는 **네 번** 갈아엎고 나서야 동작했다.

### 함정 1: interactive 셸(`-i`)의 명령 에코백

**증상**: `cd /tmp && pwd`를 실행했더니 출력에 명령 텍스트 `cd /tmp && pwd`가
그대로 섞이고, 정작 다음 명령의 실제 출력은 누락됐다.

**원인**: `bash -i`(interactive)는 입력받은 명령을 화면에 에코백한다. 이게 stdout에
섞여 들어온다.

**해결**: `-i`를 빼라. `["/bin/bash", "--norc", "--noprofile"]`로 충분하다.
프롬프트는 `PS1=""`로 제거.

### 함정 2: stdin에 `\x03`를 써도 SIGINT가 안 간다

**첫 시도**: 타임아웃 시 `proc.stdin.write("\x03\n")`로 Ctrl-C를 보내려 했다.

**증상**: `sleep 10`이 안 죽고 타임아웃이 계속 났다.

**원인**: non-interactive 셸에서는 stdin의 `\x03`이 **시그널로 해석되지 않고
그냥 문자**로 들어간다. 터미널(tty)이 있어야 `\x03`이 SIGINT로 변환되는데,
파이프로 연결된 stdin엔 tty가 없다.

**교훈**: 진짜 시그널을 보내려면 `os.kill`/`os.killpg`를 써야 한다.

### 함정 3: `killpg`가 bash까지 죽인다

**둘째 시도**: `start_new_session=True`로 프로세스 그룹을 만들고
`os.killpg(pgid, SIGINT)`를 보냈다.

**증상**: `sleep`은 죽었지만 **bash 세션 자체도 죽어서** 그 다음 명령이
"bash session is dead"로 실패했다.

**원인**: `killpg`는 그룹의 **모든** 프로세스에 시그널을 보낸다. bash가 그룹
리더이므로 bash도 SIGINT를 받아 종료됐다.

**셋째 시도**: bash가 `trap '' INT TERM`으로 시그널을 무시하게 했다.

**증상**: bash는 살아남았지만, 이번엔 `sleep`이 trap을 **상속**받아 SIGTERM도
무시해서 안 죽었다. 게다가 타임아웃된 명령의 뒤늦은 센티넬 출력이 **다음 명령의
출력에 섞여** 한 번 더 타임아웃으로 오보됐다.

### 함정 4: 결국 hard reset이 정답

**넷째 시도(최종)**: graceful interrupt를 포기하고, 타임아웃 시
**프로세스 그룹 전체를 SIGKILL → 새 bash 재기동 → cwd 복원**.

```python
os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)  # 그룹 통째로 강제 종료
self._spawn(self._cwd)                                  # 마지막 cwd로 재기동
```

**왜 이게 맞나**:
- SIGKILL은 trap/무시가 불가능하다. 무한루프든 시그널 무시 자식이든 100% 죽는다.
- bash까지 죽지만, 어차피 새로 띄우므로 상관없다.
- 잔여 출력 오염 문제도 프로세스가 통째로 사라지니 자동 해결.
- OpenHands도 hang 상태에는 동일하게 프로세스 그룹 강제 종료를 쓴다.

**트레이드오프**: 그 시점의 export 변수 등 셸-로컬 상태가 사라진다. cwd만 보존된다.
이 한계를 Observation의 `error`에 명시해 LLM이 알게 했다. 대부분의 코딩 작업은
이 트레이드오프가 문제되지 않는다(env가 정말 중요하면 영속 env 파일을 source하면 됨).

### 함정 5: `readline()` 블로킹

`readline()`을 그냥 호출하면 출력이 안 올 때 영원히 막힌다. 타임아웃을 걸려면
반드시 `select.select([stdout], [], [], remaining)`로 먼저 읽기 가능 여부를
확인하고, 남은 시간이 0이 되면 빠져나와야 한다.

### 함정 6: 로컬 모델의 tool_calls args 타입

OpenAI API는 `args`를 dict로 주지만, 일부 로컬 모델(litellm 경유)은 JSON
**문자열**로 주거나 깨진 JSON을 준다. `tools_node`에서 `isinstance(args, str)`
체크 후 `json.loads` 시도, 실패 시 단일 인자로 폴백하는 방어 코드가 필수다.

---

## 11. 확장 가이드

이 구현은 의도적으로 최소화했다. 프로덕션에는 아래를 추가한다.

### 보안 격리 (가장 중요)
현재는 로컬에서 도구를 직접 실행한다. 신뢰 못 할 입력에는 위험하다.
- `tools_node` 앞에 명령 화이트리스트/블랙리스트 필터 추가
- bash/python을 Docker 컨테이너 안에서 실행 (OpenHands의 `DockerRuntime` 방식)
- 위험 명령(`rm -rf`, 네트워크 접근 등) 실행 전 사람 승인 요청

### python을 stateful로 (변수 지속)
`jupyter_client`로 IPython 커널을 띄우면 변수가 호출 간 유지된다
(OpenHands `IPythonRunCellObservation` 방식). 단, 커널 격리·재시작 관리가 추가된다.

### 대화/상태 영속화
LangGraph `checkpointer`(예: `SqliteSaver`)를 `compile(checkpointer=...)`에 주면
대화와 그래프 상태가 저장되어, 프로세스 재시작 후에도 이어갈 수 있다.

### Human-in-the-loop
LangGraph 1.0의 `interrupt`로 위험한 도구 호출 전 그래프를 멈추고 사람 승인을 받는다.

### 스트리밍
`graph.stream(..., stream_mode="messages")`로 토큰을 실시간 출력. 단,
스트리밍 모드에서 tool_call id가 누락되는 알려진 이슈가 있으니 누적 로직을 견고히.

### native tool calling이 안 될 때
로컬 모델이 function calling을 제대로 못 하면:
- litellm의 prompt-based function calling fallback 사용
- 또는 `mlx-openai-server`/`vllm-mlx`처럼 모델별 tool-call 파서를 가진 백엔드로 교체
  (Qwen3 계열은 hermes/nous 파서 사용)

### Observation 강화
- 파일 변경 감지(작업 전후 git status diff)를 Observation에 추가
- 이미지/바이너리 출력 처리
- 컬러 ANSI 코드 stripping (현재는 TERM=dumb로 억제하지만 일부 도구는 강제 출력)

---

## 12. 부록: 전체 소스

위 5~8장의 코드 블록이 완전한 소스다. 4개 파일을 그대로 복사하면 동작한다.
파일별 의존성과 라인 수:

| 파일 | 의존 | 대략 라인 수 |
|------|------|-------------|
| `observation.py` | (없음, stdlib만) | ~97 |
| `bash_session.py` | observation | ~159 |
| `python_executor.py` | observation | ~76 |
| `graph.py` | observation, bash_session, python_executor, langchain, langgraph | ~153 |

### 빠른 시작 요약
```bash
# 1. 4개 파일을 agent/ 디렉터리에 생성 (5~8장 코드)
# 2. 패키지 설치
pip install langgraph langchain-openai langchain-core
# 3. LLM 백엔드 환경변수
export LITELLM_URL=http://127.0.0.1:8089/v1
export LITELLM_KEY=local
export AGENT_MODEL=<your-model>
# 4. 실행
cd agent && python graph.py
```

---

**작성 근거**: 이 문서의 모든 코드는 실제로 작성·실행·디버깅하여 6장/7장의 단위
검증을 통과시킨 결과물이다. 특히 10장의 함정들은 bash 타임아웃 복구를 네 번
재작성하며 직접 겪은 실패 기록이다.
