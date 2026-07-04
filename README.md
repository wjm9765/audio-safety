# audio-safety

Mechanistic safety experiments for audio language models.

The current first experiment is the **Audio-RDO Refusal Axis Existence Gate**:

> Does `Qwen2-Audio-7B-Instruct` contain an audio-conditioned refusal axis in the LLM residual stream when the axis is constructed with RDO-style gradient optimization instead of difference-in-means steering?

This replaces the earlier "multi-cone drift first" plan. Cone geometry, token-aware defense, and style-aware multi-cone interventions are explicitly downstream of this gate.

- **Design:** [`docs/experiments/exp1_refusal_cone_drift/design.md`](docs/experiments/exp1_refusal_cone_drift/design.md)
- **Results log:** [`docs/experiments/exp1_refusal_cone_drift/results.md`](docs/experiments/exp1_refusal_cone_drift/results.md)

## Method Sketch

1. Build a controlled harmful-benign audio dataset from FigStep/SafeBench seeds, CosyVoice2 style renders, and transcript controls.
2. Label outputs as `policy_refusal`, `harmful_compliance`, `benign_answer`, or `decoding_failure`.
3. Train an audio-native RDO refusal axis `r_A` at candidate Qwen2-Audio residual stream sites.
4. Validate `r_A` with addition, ablation, and paired benign retention.
5. Compare against `MDSteer-c2r`, SARSteer-style text-derived refusal vector, and random controls at matched ORR.
6. Test whether transcript-fixed style shifts reduce `r_A` occupancy and whether coordinate restoration recovers refusal.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # base + dev (CPU-only: config/stats/tests)
uv sync --group gpu   # + torch/transformers/librosa for Qwen2-Audio
```

On a GPU server, point caches at a persistent workspace:

```bash
export AUDIO_SAFETY_WORKSPACE=/workspace/audio_safety
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TORCH_HOME=/workspace/cache/torch
export XDG_CACHE_HOME=/workspace/cache
```

## Qwen2-Audio

The project follows the official Qwen/Hugging Face inference path:
`AutoProcessor` + `Qwen2AudioForConditionalGeneration` + `processor.apply_chat_template`.

```bash
./scripts/download_qwen2_audio.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/qwen2_audio_infer.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --audio /workspace/audio_safety/data/audio/demo.wav \
  --instruction "Please answer the question in the audio."
```

## Run Skeleton

The top-level experiment CLI snapshots config and creates run directories. Stage implementations are intentionally explicit and GPU-server oriented.

```bash
./scripts/run_experiment.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --run-name exp1_$(date +%Y%m%d_%H%M)_audio_rdo_gate
```

Stages:

```text
data, behavior, rdo, baselines, style_escape, restoration, stats, all
```

## Layout

```text
configs/         # model / dataset / experiment / path configs
docs/experiments # pre-registered designs + append-only results logs
src/audio_safety # library code: config, data, models, pipelines, evaluation, utils
scripts/         # thin CLI entry points
tests/           # unit tests (GPU-free where possible)
outputs/         # run artifacts (git-ignored)
data/            # dataset notes only; raw data is never committed
```

## Tests

```bash
uv run pytest
```

Tests are intended to pass without GPU dependencies for config loading, path handling, numpy geometry/statistics, and Audio-RDO gate math.
