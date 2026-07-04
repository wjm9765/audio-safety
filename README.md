# audio-safety

Probing **audio-induced refusal-cone drift** in audio language models.

We test whether different audio attack families (plain TTS, non-speech mixing, speaking-style variation, signal perturbation) push an audio LLM's internal representations along **different axes** of its refusal subspace (H1), or all **collapse onto a single refusal direction** (H0). No training, no defense method — a single pre-registered geometric probe on `Qwen2-Audio-7B-Instruct`.

- **Design (pre-registered):** [`docs/experiments/exp1_refusal_cone_drift/design.md`](docs/experiments/exp1_refusal_cone_drift/design.md)
- **Results log:** [`docs/experiments/exp1_refusal_cone_drift/results.md`](docs/experiments/exp1_refusal_cone_drift/results.md)

## Method sketch

1. Build a **refusal cone** `[b1..bk]` from clean-text harmful/benign contrasts (diff-in-means per harm category + PCA cross-check), at the layer `L*` with maximal separation. Keep only axes that pass **causal ablation**.
2. For each harmful content `c` and attack family `f`, measure the **paired drift** `d_f(c) = h_f(c) − h_text(c)` — content confounds cancel by pairing.
3. Project drifts onto the cone and compare family profiles: **mean pairwise cosine + permutation test + bootstrap CI**, judged against pre-registered GO / NO-GO / AMBIGUOUS thresholds.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # base + dev (CPU-only: stats, tests)
uv sync --group gpu   # + torch/transformers for model inference
```

On a GPU server, point caches at a persistent volume before running:

```bash
export AUDIO_SAFETY_WORKSPACE=/workspace/audio_safety
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TORCH_HOME=/workspace/cache/torch
export XDG_CACHE_HOME=/workspace/cache
```

## Run

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --run-name exp1_$(date +%Y%m%d_%H%M)_probe
```

Everything (model ID, layer sweep, family list, statistical thresholds) is configured in `configs/`; override any key from the CLI:

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --override stats.n_permutations=1000 --override seed=1
```

## Layout

```
configs/         # model / experiment / path configs (YAML)
docs/experiments # pre-registered designs + append-only results logs
src/audio_safety # library code: config, data, models, pipelines, evaluation, utils
scripts/         # thin CLI entry points
tests/           # unit tests (pass without a GPU)
outputs/         # run artifacts (git-ignored)
data/            # dataset notes only; raw data is never committed
```

## Tests

```bash
uv run pytest
```

Tests cover config loading, path/cache resolution, and the statistical decision machinery (mean pairwise cosine, permutation test, bootstrap CI, GO/NO-GO rules) on synthetic H0/H1 data — no GPU needed.
