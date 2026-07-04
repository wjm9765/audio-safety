# Exp 1: Audio-RDO Refusal Axis Existence Gate — Results Log

> 사전 등록 설계: [design.md](./design.md) (§0 판정 기준)
> 이 파일은 **append-only run 로그**다. 과거 항목은 수정하지 않고, 정정은 새 항목으로 추가한다.

## Current Status

- **Decision:** `NOT RUN`
- **Latest run:** —
- **Blocking items:** curated harmful-benign pair manifest, CosyVoice2 render manifest, GPU pipeline wiring

## Run Summary

| Date | Run | Commit | Site `(ell,p*)` | Add RR | Benign ORR | Abl ASR | Baseline win | Escape | Restore | Decision |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|

<!-- 최신 run이 마지막 행. -->

---

## Entries

<!-- 아래에 run 항목을 append한다. 최신 항목이 마지막. -->

<!-- ENTRY TEMPLATE:

### <run_name> — YYYY-MM-DD

- **Git commit:** `<hash>`
- **Config:** `configs/experiments/exp1_refusal_cone_drift.yaml` + overrides: `<none | list>`
- **Stage(s) run:** <data | behavior | rdo | baselines | style_escape | restoration | stats | all>
- **Selected site:** layer `<ell>`, position `<assistant_start_pre | first_generation_prelogit>`

**Data integrity**

| Split | Harmful-benign pairs | Rendered clips | Transcript pass | Style pass | Geometry-valid |
|---|---:|---:|---:|---:|---:|

**Behavior decomposition**

| Condition | policy_refusal | harmful_compliance | benign_answer | decoding_failure |
|---|---:|---:|---:|---:|

**Axis validation**

| Vector | Harmful add RR delta | Benign ORR delta | Ablation ASR delta | Matched ORR result |
|---|---:|---:|---:|---|
| MDSteer-c2r | | | | |
| SARSteer-text | | | | |
| Random | | | | |
| RDO-A | | | | |

**Style escape / restoration**

| Metric | Value | Threshold | Verdict |
|---|---:|---:|---|
| Genuine style gap | | >= 8pp | |
| Escape Spearman | | >= 0.30 | |
| Escape AUROC | | >= 0.65 | |
| Restoration RR delta | | >= 20pp | |
| Restored fraction | | >= 25% | |
| Restoration benign ORR delta | | <= 3pp | |

- **Decision:** <GO | WEAK-GO | NO-GO | AMBIGUOUS> — <one-line reason>
- **Analysis:** `outputs/<run_name>/analysis.md`
- **Cross-check:** <link to outputs/cross_checks/... | not performed>
- **Figures:** `outputs/<run_name>/figures/`
- **Notes / anomalies:** <failed samples, deviations, judge disagreements>

-->
