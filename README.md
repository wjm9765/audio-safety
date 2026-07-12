# audio-safety

Mechanistic safety experiments for audio language models.

The current first experiment is the **Audio-RDO Refusal Axis Existence Gate**:

> Does `Qwen2-Audio-7B-Instruct` contain an audio-conditioned refusal axis in the LLM residual stream when the axis is constructed with RDO-style gradient optimization instead of difference-in-means steering?

This replaces the earlier "multi-cone drift first" plan. Cone geometry, token-aware defense, and style-aware multi-cone interventions are explicitly downstream of this gate.

- **Design:** [`docs/experiments/exp1_refusal_cone_drift/design.md`](docs/experiments/exp1_refusal_cone_drift/design.md)
- **Current implementation context:** [`docs/experiments/exp1_refusal_cone_drift/context.md`](docs/experiments/exp1_refusal_cone_drift/context.md)
- **Results log:** [`docs/experiments/exp1_refusal_cone_drift/results.md`](docs/experiments/exp1_refusal_cone_drift/results.md)

## Method Sketch

1. Build a controlled harmful-benign audio dataset from FigStep/SafeBench seeds, CosyVoice2 style renders, and transcript controls.
2. Label outputs as `policy_refusal`, `harmful_compliance`, `benign_answer`, or `decoding_failure`.
3. Train an audio-native RDO refusal axis `r_A` at candidate Qwen2-Audio residual stream sites. The current implementation freezes Qwen2-Audio, optimizes only one hidden-size vector per candidate site, uses add/ablate/benign-retain losses, and accumulates gradients one microbatch at a time to fit on an A40.
4. Validate `r_A` with addition, ablation, and paired benign retention.
5. Compare against `MDSteer-c2r`, SARSteer-style text-derived refusal vector, and random controls at matched ORR.
6. Test whether transcript-fixed style shifts reduce `r_A` occupancy and whether coordinate restoration recovers refusal.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The default `uv sync` installs the dev and GPU dependency groups. The `gpu`
dependency group is pinned for the current RunPod A40 target: `torch==2.9.1` and
`torchaudio==2.9.1` are resolved from the official PyTorch CUDA 12.8 wheel index
(`cu128`). Do not loosen this to CUDA 13 wheels unless the server NVIDIA driver
is upgraded accordingly.

On a GPU server, point caches at a persistent workspace:

```bash
export AUDIO_SAFETY_WORKSPACE=/workspace/audio_safety_data
export AUDIO_SAFETY_DATA_DIR=/workspace/audio_safety_data/data
export AUDIO_SAFETY_OUTPUT_DIR=/workspace/audio_safety_data/outputs
export AUDIO_SAFETY_CACHE_DIR=/workspace/audio_safety_data/cache
export HF_HOME=/workspace/audio_safety_data/cache/huggingface
export HF_HUB_CACHE=/workspace/audio_safety_data/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/audio_safety_data/cache/huggingface/datasets
export TORCH_HOME=/workspace/audio_safety_data/cache/torch
export XDG_CACHE_HOME=/workspace/audio_safety_data/cache
export OPENROUTER_API_KEY=<your_key>
export PYTORCH_ALLOC_CONF=expandable_segments:True  # recommended for RDO training on A40
```

## Qwen2-Audio

The project follows the official Qwen/Hugging Face inference path:
`AutoProcessor` + `Qwen2AudioForConditionalGeneration` + `processor.apply_chat_template`.

```bash
./scripts/download_qwen2_audio.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/qwen2_audio_infer.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --audio /workspace/audio_safety_data/data/audio/demo.wav \
  --instruction "Please answer the question in the audio."
```

## Stage Order

After cloning on an A40 GPU cloud instance:

```bash
uv sync

export OPENROUTER_API_KEY=<your_key>

./scripts/prepare_audio_rdo_pairs.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --limit 150

# OpenRouter has no native Chat Batch endpoint. The project therefore runs a
# bounded client-side batch with 8 concurrent requests by default. Tune it if
# the selected provider returns 429/503 responses:
./scripts/prepare_audio_rdo_pairs.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --limit 150 \
  --override dataset.pair_generation.max_concurrency=4

# Review/edit $AUDIO_SAFETY_DATA_DIR/text/figstep/audio_rdo_pairs.jsonl.

./scripts/render_audio_rdo.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --dry-run

# The default renderer uses scripts/cosyvoice2_tts.py in batch mode. It loads
# CosyVoice2 once, then renders all pending jobs from a JSONL queue under
# AUDIO_SAFETY_DATA_DIR/manifests/. The current fast gate uses three styles
# (neutral, sad, angry): 150 pairs x harmful/benign x 3 styles = 900 wav files.
./scripts/render_audio_rdo.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

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

## Fast All-Position Rebuttal Check

Use the permanent fast config for the current all-position intervention rebuttal
check. Preconditions: behavior outputs already cover the 3-style train and
validation splits. The smoke run trains only and writes validation metrics; the
overnight run trains, extracts activations/baselines, evaluates heldout
interventions, and writes the final gate metrics.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RUN_NAME=exp1_$(date +%Y%m%d_%H%M)_allpos_rebuttal

# Smoke: cheap direction and over-steering check, train only.
./scripts/train_rdo_axis.py \
  --config configs/experiments/exp1_refusal_cone_drift_fast.yaml \
  --run-name "${RUN_NAME}_smoke"

# Overnight: layer-12/16 neighborhood, train -> extract -> evaluate.
nohup bash -c '
./scripts/train_rdo_axis.py \
  --config configs/experiments/exp1_refusal_cone_drift_fast.yaml \
  --override "hidden.layers=[12,14,16,18,20]" \
  --override "rdo.train_steps=100" \
  --override "rdo.limit_per_site=20" \
  --run-name "$RUN_NAME" \
&& ./scripts/extract_rdo_activations.py \
  --config configs/experiments/exp1_refusal_cone_drift_fast.yaml \
  --run-name "$RUN_NAME" \
&& ./scripts/evaluate_rdo_gate.py \
  --config configs/experiments/exp1_refusal_cone_drift_fast.yaml \
  --run-name "$RUN_NAME"
' > "outputs/${RUN_NAME}.log" 2>&1 &

tail -f "outputs/${RUN_NAME}.log"
cat "/workspace/audio_safety_data/outputs/${RUN_NAME}/metrics.json"
```

Always quote list overrides such as `"hidden.layers=[12,14,16,18,20]"`, especially
under zsh. With all-position intervention, `alpha=2.0` may over-steer; if benign
ORR or decoding failures spike, lower alpha with a matching train/eval override
such as `--override rdo.alpha=1.0` or `--override rdo.alpha=0.5`.

Use `configs/experiments/exp1_refusal_cone_drift.yaml` for the full paper-facing
run after the fast config gives a promising signal.

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
