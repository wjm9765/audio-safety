# Experiment Documentation Conventions

각 실험은 `docs/experiments/<exp_name>/` 디렉터리 하나를 가진다.

| 파일 | 역할 | 규칙 |
|---|---|---|
| `design.md` | 사전 등록 실험 설계 | 가설, 판정 기준, 통계 방법은 실험 시작 후 **수정 금지**. 부득이한 변경은 문서 하단에 변경 이력(날짜·사유) 추가 |
| `results.md` | 누적 run 로그 | **append-only**. run 단위로 항목 추가, 과거 항목 수정 금지(정정은 새 항목으로). `/update-experiment-log` 스킬이 관리 |

## Run 규약

- run naming: `exp1_{YYYYMMDD}_{HHMM}_{tag}`
- run 산출물은 `outputs/<run_name>/`에 저장 (`outputs/README.md` 참조)
- 판정(GO/NO-GO/AMBIGUOUS)은 `design.md` §0의 사전 등록 임계값으로만 내린다
- 판정 확정 전에 `/analyze-experiment` + `/codex-cross-check`를 거친다
