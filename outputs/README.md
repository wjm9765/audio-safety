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
    activations/*.npz           # layer별 hidden states (조건별)
    projections.npz             # (n_contents, n_families, k) drift projections
    figures/                    # fig1_layer_sweep, fig2_profiles, fig3_separation ...
    analysis.md                 # /analyze-experiment 산출물
  cross_checks/
    YYYYMMDD_<topic>.md         # /codex-cross-check 리포트
```

`metrics.json`에는 최소한 다음을 포함한다: family별 유효 n·comprehension 통과율,
mean pairwise cosine, permutation p, bootstrap CI, dominant axes, 축별 ablation
refusal-rate 하락, classifier macro-F1, 판정 결과(status + reasons).
