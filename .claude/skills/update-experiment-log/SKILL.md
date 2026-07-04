---
name: update-experiment-log
description: 실험 run 결과를 docs/experiments/<exp>/results.md에 append-only로 기록. run 디렉터리의 config snapshot·metrics·analysis를 읽어 표준 항목을 추가하고 Current Status와 Run Summary 표를 갱신한다. 분석 완료 후 사용.
---

# Update Experiment Log

`outputs/<run_name>/`의 산출물을 읽어 `docs/experiments/<exp>/results.md`에 표준 형식으로 기록한다.

## 불변 규칙

1. **append-only:** 과거 항목은 절대 수정하지 않는다. 정정이 필요하면 새 항목을 추가하고 원 항목을 링크로 참조한다 (`### Correction to <run_name>`)
2. **design.md는 이 스킬의 수정 대상이 아니다** — 어떤 경우에도 건드리지 않는다
3. **수치는 산출물에서만:** `metrics.json`, `config_snapshot.yaml`, `analysis.md`에 없는 수치를 기억이나 대화 내용으로 채우지 않는다. 없는 값은 `n/a`로 남긴다
4. 갱신 가능한 곳은 상단 **Current Status** 블록과 **Run Summary** 표(행 추가)뿐이다

## 절차

### 1. Run 확정

- 사용자가 run을 지정하지 않았으면 `ls -t outputs/`로 최근 run을 찾고 사용자에게 확인
- 필수 파일 확인: `config_snapshot.yaml`, `metrics.json` — 없으면 중단하고 무엇이 없는지 보고
- `analysis.md`가 없으면: 기록은 가능하지만 `/analyze-experiment`를 먼저 돌릴 것을 권고

### 2. 항목 데이터 수집

- `config_snapshot.yaml`: git commit, config 전체 → base config(`configs/experiments/*.yaml`)와 diff하여 "Config delta" 산출
- `metrics.json`: 판정 metric 전부 (mpc, permutation p, bootstrap CI, dominant axes, family별 n·통과율, classifier F1, decision status+reasons)
- `analysis.md`: 판정 요약과 주요 위협 요인 → Notes에 1–2줄로
- cross-check 리포트가 `outputs/cross_checks/`에 있으면 링크

### 3. results.md 갱신

results.md 하단 주석의 ENTRY TEMPLATE을 복사해 **Entries 섹션 끝에** 추가하고:

- Run Summary 표에 한 행 추가 (최신이 마지막 행)
- Current Status 블록 갱신: Decision(최신 판정), Latest run, Blocking items
- 모든 경로는 repo-relative 상대 링크로

### 4. 검증

- 추가한 항목의 수치가 `metrics.json`과 1:1로 일치하는지 재대조
- 마크다운 표가 깨지지 않았는지 확인
- git diff를 보여주고 이전 항목이 변경되지 않았음을 확인 (`git diff --stat docs/`에서 삭제 라인이 기존 항목을 건드리면 안 됨 — Current Status/Summary 표 갱신분 제외)

## 판정 표기

- Decision은 `metrics.json`의 status를 그대로 쓴다 (GO / NO-GO / AMBIGUOUS / PARTIAL — stage 일부만 실행된 run은 PARTIAL)
- 판정 근거 요약은 decision reasons 필드에서 가져온다 — 재해석 금지
