# Exp 1: Audio-Induced Refusal-Cone Drift Probe — Results Log

> 사전 등록 설계: [design.md](./design.md) (판정 기준 §0 — 사후 수정 금지)
> 이 파일은 **append-only run 로그**다. 항목은 `/update-experiment-log` 스킬로 추가하며, 과거 항목은 수정하지 않는다 (정정은 새 항목으로).

## Current Status

- **Decision:** `NOT RUN`
- **Latest run:** —
- **Blocking items:** 데이터 준비 (design.md §2), 파이프라인 구현 (§7)

## Run Summary

| Date | Run | Commit | Valid n/family | MPC | Perm p | 95% CI | Dominant axes | Decision | Notes |
|---|---|---|---|---|---|---|---|---|---|

<!-- /update-experiment-log가 여기에 행을 추가한다. 최신 run이 마지막 행. -->

---

## Entries

<!-- 아래에 run 항목을 append한다. 최신 항목이 마지막. -->

<!-- ENTRY TEMPLATE (copy for each run):

### <run_name> — YYYY-MM-DD

- **Git commit:** `<hash>`
- **Config:** `configs/experiments/exp1_refusal_cone_drift.yaml` + overrides: `<none | list>`
- **Config delta vs base:** <none | what changed>
- **Stage(s) run:** <data | cone | drift | stats | all>

**Data integrity**

| Family | Rendered | Comprehension pass | Valid n |
|---|---|---|---|

**Cone**

- L* = <layer> (<pct>% depth), separation sweep: `figures/fig1_layer_sweep.png`
- Axes surviving causal ablation: <k>/<candidates> (Table 1: `<path>`)
- Method A vs B principal angles: <values>

**Decision metrics (vs pre-registered §0)**

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| Mean pairwise cosine | | GO < 0.60 / NO-GO >= 0.85 | |
| Permutation p | | < 0.05 | |
| Dominant-axis disagreement | | >=2 families differ | |
| Bootstrap 95% CI | | — | |
| Family classifier macro-F1 | | chance = 0.25 | |

- **Decision:** <GO | NO-GO | AMBIGUOUS> — <one-line reason>
- **Analysis:** `outputs/<run_name>/analysis.md`
- **Cross-check:** <link to outputs/cross_checks/... | not performed>
- **Figures:** `outputs/<run_name>/figures/`
- **Notes / anomalies:** <anything unexpected, failed samples, deviations>

-->
