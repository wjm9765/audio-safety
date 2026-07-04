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
export AUDIO_SAFETY_DATA_DIR=/workspace/audio_safety/data
export AUDIO_SAFETY_OUTPUT_DIR=/workspace/audio_safety/outputs
export AUDIO_SAFETY_CACHE_DIR=/workspace/audio_safety/cache
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TORCH_HOME=/workspace/cache/torch
export XDG_CACHE_HOME=/workspace/cache
export OPENROUTER_API_KEY=<your_key>
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

## Stage Order

After cloning on an A40 GPU cloud instance:

```bash
uv sync
uv sync --group gpu

mkdir -p "$AUDIO_SAFETY_DATA_DIR/text/figstep"
git clone https://github.com/CryptoAILab/FigStep /tmp/FigStep
cp /tmp/FigStep/data/question/safebench.csv \
  "$AUDIO_SAFETY_DATA_DIR/text/figstep/safebench.csv"

./scripts/prepare_audio_rdo_pairs.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --limit 150

# Review/edit $AUDIO_SAFETY_DATA_DIR/text/figstep/audio_rdo_pairs.jsonl.

./scripts/render_audio_rdo.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --dry-run

# Then set dataset.tts.command_template for your CosyVoice2 install and run
# without --dry-run.

./scripts/score_transcripts.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/download_qwen2_audio.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/generate_behavior.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

RUN_NAME=exp1_$(date +%Y%m%d_%H%M)_audio_rdo_gate

./scripts/train_rdo_axis.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --run-name "$RUN_NAME"

./scripts/extract_rdo_activations.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --run-name "$RUN_NAME"

./scripts/evaluate_rdo_gate.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --run-name "$RUN_NAME"
```

The top-level runner supports the same stages, but long GPU jobs are easier to
resume with the dedicated scripts above:

```text
pairs, render_audio, score_transcripts, behavior, rdo, baselines,
extract_activations, style_escape, restoration, stats
```

`all` is intentionally not wired; run stages explicitly so failures are resumable.

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
