# small_agent — LangGraph 코딩 에이전트

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

OpenHands 스타일 **Observation**을 갖춘 bash/python ReAct 에이전트.

## 구조

```
small_agent/
├── observation.py       # 도구 실행 결과 구조화
├── bash_session.py      # stateful bash (cd/export 지속)
├── python_executor.py   # stateless python subprocess
├── exa_search.py        # Exa web_search
├── context7_search.py   # Context7 library docs search
├── forced_choice.py     # tool_choice=required + respond_directly
├── llm_debug_log.py     # optional raw LLM HTTP request/response JSONL logging
├── graph.py             # LangGraph agent ↔ tools 루프 + CLI
└── tests/
```

## 사전 요구

- Python 3.10+
- macOS 또는 Linux (`/bin/bash`, `select`, `killpg`)
- OpenAI 호환 LLM (`http://127.0.0.1:8089/v1`, native tool calling 필요)
- MLX/LiteLLM은 **tool 결과 follow-up**(`ToolMessage` 히스토리)에서 500 버그가 있어, `graph.py`가 LLM 호출 전 plain-text로 변환함

## 설치·실행

```bash
cd small_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

export LITELLM_URL=http://127.0.0.1:8089/v1
export LITELLM_KEY=local
export AGENT_MODEL=<모델명>   # 생략 시 /health 또는 /v1/models 에서 자동 선택

python graph.py              # stream (기본)
python graph.py --no-stream  # buffered
```

- LLM 응답은 **SSE 스트림** — 최종 답변 토큰 + tool 호출 시 `[web_search] query` 형태 표시 (실행 결과 본문은 숨김)

## 아키텍처

```
User → agent (LLM) → [tool_calls?] → tools (bash/python/web_search/context7_search) → agent → END
```

- **bash**: stateful — `cd`, `export`, venv가 명령 간 유지
- **python**: stateless — 매번 임시 `.py` subprocess
- **web_search**: Exa API — 최신 뉴스·일반 웹 사실 검색
- **context7_search**: Context7 API — 라이브러리/프레임워크 공식 문서 검색
- 타임아웃 시 bash는 **hard reset** (SIGKILL + cwd 복원)

## 환경변수

| 변수 | 기본 |
|------|------|
| `LITELLM_URL` | `http://127.0.0.1:8089/v1` |
| `LITELLM_KEY` | `local` |
| `AGENT_MODEL` | (자동) `/health` → `/v1/models` 첫 항목 |
| `EXA_API_KEY` | (선택) Exa API 키. 없으면 ada vault `exa.api_key` 시도 |
| `CONTEXT7_API_KEY` | (선택) Context7 API 키. 없으면 ada vault `context7.api_key` 시도 |
| `SMALL_AGENT_STREAM` | `1` (0/false = `--no-stream`) |
| `SMALL_AGENT_MAX_TOKENS` | `2048` |
| `SMALL_AGENT_LLM_LOG` | (선택) `1`이면 LLM HTTP 요청/응답 본문 JSONL 로깅 ON (`tools`, `tool_choice`, SSE 포함) |
| `SMALL_AGENT_LLM_LOG_DIR` | (선택) 로그 디렉터리 (기본 `small_agent/.local/llm/`) |
| `SMALL_AGENT_FORCE_CHOICE` | `1` (기본). `0`이면 `tool_choice=auto`, `respond_directly` 미사용 |

시작 시 `forced_choice ON|OFF` / `stream` / 검색 도구 상태가 표시됩니다. LLM 로깅 ON이면 `llm log → …jsonl` 경로도 출력됩니다.

첫 LLM 호출은 `tool_choice=required`로 도구를 하나 고르게 합니다. `respond_directly`만 선택되면 tools 루프 없이 바로 텍스트 답변합니다.

## 테스트

```bash
pytest -q
```

상세 설계·함정 기록: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## 주의

도구는 **로컬에서 직접** 실행됩니다. 신뢰할 수 없는 입력에는 샌드박스/컨테이너 격리가 필요합니다.

## License

MIT — see [LICENSE](LICENSE).
