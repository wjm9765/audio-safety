# outputs/

Run 산출물 디렉터리. git 추적 제외 (이 README만 커밋).
실제 output 루트는 `$AUDIO_SAFETY_OUTPUT_DIR` (기본: `/workspace/audio_safety/outputs`)이며,
로컬 실행 시 이 디렉터리를 쓸 수도 있다.

## Run 디렉터리 규약

```
outputs/
  <run_name>/                   # exp1_{YYYYMMDD}_{HHMM}_{tag}
    config_snapshot.yaml        # 실행 시점 config 전체 + git commit hash
    metrics.json                # 판정 metric 포함 모든 수치
    behavior_table.json         # 4-way behavior decomposition
    selected_site.json          # validation-selected (layer, position)
    rdo_axis.npz                # learned r_A and metadata
    baseline_vectors.npz        # MDSteer-c2r, SARSteer-text, random controls
    activations/*.npz           # condition/site hidden states
    figures/                    # behavior, axis validation, escape/restoration
    analysis.md                 # /analyze-experiment 산출물
  cross_checks/
    YYYYMMDD_<topic>.md         # /codex-cross-check 리포트
```

`metrics.json`에는 최소한 다음을 포함한다: transcript/style control 통과율,
decoding_failure 비율, addition RR delta, benign ORR delta, ablation ASR delta,
matched-ORR baseline 비교, Escape Spearman/AUROC, restoration RR delta,
restored fraction, bootstrap CI, 판정 결과(status + reasons).
