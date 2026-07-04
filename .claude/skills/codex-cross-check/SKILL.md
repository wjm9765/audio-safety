---
name: codex-cross-check
description: Claude↔Codex blind 교차 검증. 통계·판정 코드 구현 후, GO/NO-GO 판정 직전, 설계 변경, 중요한 수치 해석 시 사용. 결론을 공유하지 않고 원자료만 Codex에 전달해 독립 검증을 받고, 불일치는 재계산으로 판정한다.
---

# Codex Cross-Check

단일 모델의 판단 오류(그럴듯하지만 틀린 통계 구현, 편향된 해석)를 잡기 위한 blind 교차 검증 절차.

## 핵심 원칙

1. **Blind:** Codex에게 나(Claude)의 결론·판정·의심점을 먼저 알려주지 않는다. 원자료(코드, 수치, 사전 등록 기준)만 준다. 유도성 질문("이 구현이 맞지 않나요?") 금지 — 중립 과제("이 구현을 독립적으로 검증하고 버그를 찾아라")로 준다.
2. **어느 쪽도 정답이 아님:** 불일치 시 Claude도 Codex도 authoritative하지 않다. 재계산·소스 확인·최소 재현 스크립트로 판정한다.
3. **결론이 아니라 근거를 요구:** Codex에게 verdict만이 아니라 단계별 reasoning과 구체적 지적(파일:라인)을 요구한다.

## 절차

### 1. 검증 대상 확정

세 유형 중 하나를 명시한다:

- **(a) 코드 검증** — 예: `src/audio_safety/evaluation/stats.py`의 permutation test 구현
- **(b) 수치/판정 검증** — 예: `outputs/<run>/metrics.json`의 수치가 design.md §0 기준으로 어떤 판정인지
- **(c) 설계/해석 검증** — 예: 결과 해석, confound 논증, 설계 변경안

### 2. Blind 프롬프트 작성

포함할 것:
- 원자료 전문 (코드 파일 내용, metrics.json, design.md의 관련 섹션)
- 과제 정의 (검증할 것, 기대 출력 형식)
- 사전 등록 기준 (판정 검증인 경우 design.md §0 표)

포함하면 안 되는 것:
- 나의 결론, 판정, 의심 지점, "아마 맞을 것" 류의 힌트

유형별 프롬프트 골격:

```text
(a) 코드: "다음은 paired drift 실험의 permutation test 구현이다. 설계 문서의
    통계 명세(첨부)와 대조해 구현 오류, 통계적 오류(exchangeability, p-value
    추정), numerical 문제를 찾아라. 각 지적에 파일:라인과 근거를 달아라.
    문제없으면 '검증 통과'와 확인한 속성 목록을 출력하라."

(b) 판정: "다음 metrics.json과 사전 등록 판정표(§0)가 있다. 이 수치가 어느
    판정(GO/NO-GO/AMBIGUOUS)에 해당하는지 독립적으로 도출하고, 각 조건의
    통과/실패를 표로 보여라. 수치에서 이상 징후(비정상적 CI 폭, n 부족,
    필터 통과율)를 발견하면 함께 보고하라."

(c) 해석: "다음 실험 결과와 설계 문서가 있다. 이 결과에 대한 가장 그럴듯한
    해석과, 그 해석을 위협하는 대안 설명 3개 이상을 도출하라."
```

### 3. Codex 호출

- **1순위:** MCP 도구 `mcp__codex__codex` (ToolSearch로 `select:mcp__codex__codex` 로드 후 호출). 후속 질의는 `mcp__codex__codex-reply`로 같은 세션을 이어간다.
- **2순위 (MCP 부재 시):** CLI — 검증은 읽기 전용이므로 sandbox를 제한한다:
  ```bash
  codex exec --sandbox read-only "<prompt>"
  ```
- 코드 검증이면 Codex가 직접 파일을 읽을 수 있게 repo 경로 기준으로 과제를 준다.

### 4. 비교 및 불일치 해소

항목별 비교표를 만든다:

| # | 항목 | Claude | Codex | 일치? | 판정 근거 |
|---|---|---|---|---|---|

불일치 항목마다:
1. 쟁점을 검증 가능한 명제로 좁힌다 (예: "add-one p-value 추정이 필요한가")
2. ground truth를 확보한다: 최소 재현 스크립트 실행(`uv run python -c ...`), 원 논문/문서 확인, 기존 테스트 확장
3. 판정 후 필요하면 `mcp__codex__codex-reply`로 Codex에 반론 기회를 준다 (이 단계에서는 근거 공유 가능 — blind는 최초 1회만)
4. 해소 불가 항목은 **UNRESOLVED**로 남기고 사용자에게 명시적으로 올린다 — 절대 조용히 한쪽을 채택하지 않는다

### 5. 리포트 저장

`outputs/cross_checks/YYYYMMDD_<topic>.md` (날짜는 `date +%Y%m%d`로):

```markdown
# Cross-Check: <topic>
- Date / Target / Claude·Codex 버전(모델)
- 방법: blind 여부, 프롬프트 요약
## 비교표
## 불일치 해소 로그 (명제 → 검증 방법 → 판정)
## 최종 결론
## UNRESOLVED (있으면)
```

### 6. 후속 조치

- 코드 결함이 확인되면: 수정 → 회귀 테스트 추가 → 필요 시 재검증
- 판정 검증이었다면: 리포트 링크를 `/update-experiment-log` 항목의 Cross-check 필드에 기록
