---
name: analyze-experiment
description: 실험 run 결과를 딥러닝 연구 기준으로 분석. 데이터 무결성 → 사전 등록 기준 대비 판정 → 통계적 건전성 → 메커니즘 해석 → 위협 요인 순으로 검토하고, 근거가 달린 분석 리포트를 outputs/<run>/analysis.md로 생성한다. 실험 run 완료 직후 사용.
---

# Analyze Experiment

실험 결과를 "숫자 나열"이 아니라 **판정 + 근거 + 위협 요인**으로 분석한다. 모든 주장에는 수치 근거를 달고, 판정은 사전 등록 기준으로만 내린다.

## 입력

- run 디렉터리: `outputs/<run_name>/` (사용자 지정, 없으면 가장 최근 run — `ls -t`로 확인 후 사용자에게 확인)
- 필수 파일: `config_snapshot.yaml`, `metrics.json`
- 기준 문서: 해당 실험의 `docs/experiments/<exp>/design.md`

## 절차 — 순서를 지킬 것

**중요: 결과 수치를 보기 전에 1단계에서 사전 등록 기준을 먼저 옮겨 적는다.** 수치를 본 뒤 기준을 해석하면 사후 합리화가 스며든다.

### 1. 사전 등록 기준 복기 (수치 확인 전)

- `design.md` §0의 판정표(임계값, 조건)를 리포트 상단에 그대로 옮겨 적는다
- config snapshot의 `decision:` 값이 design.md와 일치하는지 대조 — 불일치하면 **분석 중단, 프로토콜 위반 보고**

### 2. 데이터 무결성 (숫자를 믿을 수 있는가)

- family별 유효 n ≥ `drift.min_valid_per_family`인가? 미달 family는 어떤 판정도 뒷받침 못 함
- comprehension filter 통과율: family별로 비정상적으로 낮은 곳(< ~50%)이 있으면 selection bias 의심 — 통과 샘플이 원 분포를 대표하는지 논증 필요
- 필터 탈락 샘플이 **모든 family에서 pairwise로 제거**되었는가 (pairing 무결성)
- NaN/inf, seed·git commit 기록 여부, config override 유무

### 3. 판정 (기계적으로)

- 판정은 `audio_safety.evaluation.decision.decide()`의 출력(`metrics.json`의 status + reasons)을 그대로 쓴다 — 직접 재도출하지 않는다
- 단, decide() 입력값(mpc, p, dominant axes, causal validation 여부)이 metrics의 원수치와 일치하는지는 검산한다

### 4. 통계적 건전성 (p-value 너머)

- **효과 크기 우선:** mpc 값 자체와 bootstrap CI. CI가 판정 경계(0.60/0.85)에 걸치면 그 사실을 명시 — "GO이지만 CI 상단이 0.58~0.64에 걸침" 같은 서술
- permutation 횟수 대비 p의 해상도 (p ≈ 1/(n_perm+1) 하한 근처면 "p < X"로만 서술)
- classifier macro-F1은 chance(1/4)와 **CI를 포함해** 비교
- family별 profile norm: 한 family의 drift가 압도적으로 크면 cosine 해석이 왜곡될 수 있음 — 보고

### 5. 메커니즘 해석 (딥러닝 연구 관점)

- L*의 깊이 비율(전체 layer 대비 %)이 문헌 기대(중간 깊이 ~50–70%)와 부합하는가? 극단적 위치면 red flag
- Method A(diff-in-means) vs B(PCA)의 principal angle — subspace가 갈리면 어느 쪽 축으로 판정했는지, 왜 그쪽인지
- causal ablation 결과: 각 축의 refusal-rate 하락폭. 살아남은 축 수 k가 너무 작으면(k=1) H1 검정 자체가 무의미해짐을 지적
- dominant axis 패턴이 **의미적으로 말이 되는가** (예: style family가 hate/harassment 축을 흔든다면 그럴듯한 이유가 있는가) — 단 사후 스토리텔링임을 명시

### 6. 위협 요인 / 대안 설명 (최소 3개)

design.md §10 체크리스트를 전부 순회하고, 추가로 표준 confound를 점검:

- readout 위치가 text/audio에서 진짜 동일했는가 (tokenizer 차이, audio token 뒤 template)
- prompt/chat template이 조건 간 동일한가
- style family의 낮은 cosine이 refusal drift가 아니라 **acoustic feature 자체**를 반영할 가능성 (voice seed 분산이 충분했는지)
- perturbed family의 drift가 단순 입력 노이즈 → representation 노이즈일 가능성 (benign control의 perturbed 버전과 비교 가능한가)
- 각 위협에 대해: 현재 데이터로 반박 가능한지 / 추가 분석이 필요한지 / 다음 실험이 필요한지 분류

### 7. 다음 행동 권고

- design.md §0 표의 "다음 행동" 열을 따른다 (GO → baseline 재현·steering, NO-GO → 피벗, AMBIGUOUS → 카테고리 축 추가·n 증가·필터 강화)
- AMBIGUOUS면: 판정을 가를 수 있는 **최소 실험**을 구체적으로 제안 (무엇을 몇 개 더, 예상 소요)

## 출력: `outputs/<run_name>/analysis.md`

```markdown
# Analysis: <run_name>
- Date / Commit / Config / Analyst: Claude (<model>)

## 0. Pre-registered criteria (수치 확인 전 복기)
## 1. Data integrity        — 표 + PASS/FAIL
## 2. Decision              — status + 조건별 표
## 3. Statistical soundness — 효과 크기·CI 중심
## 4. Mechanistic interpretation
## 5. Threats to validity   — 위협 | 심각도 | 반박 가능? | 필요한 추가 작업
## 6. Recommended next steps
## Appendix: raw metrics
```

## 마무리

- 판정이 GO/NO-GO 확정 국면이면 `/codex-cross-check` (유형 b)를 권고한다
- 분석 완료 후 `/update-experiment-log`로 results.md에 기록할지 사용자에게 확인한다
- **금지:** 사전 등록 기준의 재해석, 임계값 "반올림", 판정에 유리한 샘플 재선별
