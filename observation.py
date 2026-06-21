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
    CMD = "command"
    PYTHON = "python"


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
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    cwd: str | None = None
    timed_out: bool = False
    truncated: bool = False
    duration_s: float | None = None
    error: str | None = None
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
