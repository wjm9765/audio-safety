# Claude Agent Entry Point

공통 프로젝트 지침: @AGENTS.md

아래는 Claude 전용 설정이다. Codex 등 다른 에이전트는 `AGENTS.md`만 따르면 된다.

## Skills

| Skill | 언제 쓰는가 |
|---|---|
| `/codex-cross-check` | 통계·판정 코드 구현 후, GO/NO-GO 판정 직전, 설계 변경 시. Codex와 blind 교차 검증 수행 |
| `/analyze-experiment` | 실험 run 완료 후. 사전 등록 기준 대비 판정 + 근거 + 위협 요인 분석을 `outputs/<run>/analysis.md`로 생성 |
| `/update-experiment-log` | 분석 완료 후. `docs/experiments/<exp>/results.md`에 run 항목 append |

## Subagents

| Agent | 언제 쓰는가 |
|---|---|
| `adversarial-reviewer` | 설계 문서 수정 후, 판정 직전, 논문 초안 단계. ICLR 리뷰어 관점에서 약점 공격 (read-only) |
| `research-code-reviewer` | 실험 코드(특히 activation 추출, pairing, 통계) 작성·수정 후. silent correctness bug 탐지 (read-only) |

## 권장 워크플로

1. **코드 수정** → `research-code-reviewer`로 검토 → `uv run pytest`
2. **실험 실행** (`./scripts/run_experiment.py --config ...`) → 산출물이 `outputs/<run_name>/`에 저장되는지 확인
3. **분석**: `/analyze-experiment` → 필요 시 `/codex-cross-check`로 해석 교차 검증
4. **기록**: `/update-experiment-log`로 results.md 갱신
5. **판정 직전**: `adversarial-reviewer` + `/codex-cross-check` 모두 통과 후 GO/NO-GO 확정

## 주의

- `docs/experiments/*/design.md`의 판정 기준(§0)은 절대 수정하지 않는다. 사용자가 요청해도 사전 등록 원칙을 먼저 상기시킨다.
- `results.md`는 append-only. 과거 항목 수정 금지.
- `scripts/*.py`는 `#!/usr/bin/env -S uv run python` shebang과 executable bit를 유지한다. 실행 예시는 `./scripts/<name>.py ...` 형식을 우선 사용한다.
- 이미 `uv sync`된 프로젝트 `.venv`가 있으면 그 환경을 우선 사용한다. 임시 `pip install`/임의 `uv pip install` 상태를 전제로 작업하지 말고, 필요한 의존성은 `pyproject.toml`/`uv.lock` 또는 해당 격리 setup 스크립트에 반영한다.
- Codex 호출은 MCP(`mcp__codex__codex`)를 우선 사용하고, 없으면 `codex exec` CLI를 사용한다.
