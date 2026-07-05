# Audio-RDO Gate Context

Last updated: 2026-07-05

This file preserves the working context behind the current experiment rewrite. The
folder name remains `exp1_refusal_cone_drift` for repository continuity, but the
active first experiment is now the **Audio-RDO Refusal Axis Existence Gate**.

## Current Thesis

The first paper claim should not start from a multi-cone defense. The first gate
is narrower:

> Does Qwen2-Audio contain a manipulable audio-conditioned refusal axis in the
> LLM residual stream when the axis is optimized with RDO-style gradients rather
> than constructed with difference-in-means?

If this gate fails, style-aware defenses, multi-cone geometry, and token-local
interventions should be treated as downstream pivots rather than current claims.

The intended paper spine is:

> Large Audio-Language Models contain audio-conditioned refusal coordinates in
> the LLM residual stream, but speech style can move harmful audio away from
> these coordinates while preserving transcript semantics. This style-induced
> escape is missed by text-derived steering and can be causally restored by
> patching the validated audio-native refusal coordinate.

## Research Discipline

- Use Qwen2-Audio-7B-Instruct as the first model.
- Keep the claim in the LLM residual stream, not the raw audio encoder space.
- Do not start with `cone` language. Validate a 1D axis first.
- Use 4-way behavior labels:
  - `policy_refusal`
  - `harmful_compliance`
  - `benign_answer`
  - `decoding_failure`
- Exclude `decoding_failure` from geometry analysis and report it separately.
- Keep token-aware interventions for a later phase. The first gate is axis-level
  causal mediation only.

## Source Mapping

Implement each algorithm with the corresponding paper as the conceptual source:

- **Audio-RDO axis training and retain loss:** based on the gradient-optimized
  refusal direction / concept geometry approach in *Geometry of Refusal*.
- **DIM baselines and SAR-style text vector:** based on SARSteer's audio
  activation mean-difference baselines and text-derived refusal steering.
- **Style set and transcript-preserving acoustic variation:** based on
  StyleBreak's paralinguistic/extralinguistic style axis framing.
- **Audio harmfulness motivation and Qwen2-Audio relevance:** based on AIAH and
  related LALM safety results.

Primary references checked during implementation:

- SARSteer: <https://arxiv.org/abs/2510.17633>
- Geometry of Refusal / concept cones: <https://arxiv.org/abs/2502.17420>
- StyleBreak: <https://arxiv.org/abs/2511.10692>
- AIAH: <https://arxiv.org/abs/2410.23861>
- OpenRouter API: <https://openrouter.ai/docs/api-reference/overview>

## User Decisions

The current implementation reflects these decisions:

- Adopt the proposed staged pipeline.
- Use OpenRouter for cheap benign-pair generation.
- The OpenRouter key is supplied by the user through `OPENROUTER_API_KEY`.
- Pair generation must avoid producing unsafe operational details; it rewrites
  harmful prompts into benign control questions.
- Keep the first implementation simple.
- Do not require a strict style-classifier pass in the first gate.
- Do not use an LLM judge initially; use heuristic labeling plus manual review
  fields where needed.
- Limit model generations to short outputs.
- Split the run into resumable stages.
- Target an A40 GPU first.
- Do not run the real GPU/model/TTS pipeline on the local MacBook Air.
- Simple CPU tests are allowed.
- Remove test cache and temporary files after local tests.

## Implemented Pipeline Shape

The stage scripts are executable and use `uv` in their shebangs, so they can be
called as `./scripts/<name>.py` on a prepared machine. The current experiment
config is:

```text
configs/experiments/exp1_refusal_cone_drift.yaml
```

Current stage order on the A40 server:

```bash
uv sync
uv sync --group gpu

export OPENROUTER_API_KEY=<your_key>

./scripts/prepare_audio_rdo_pairs.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --limit 150

./scripts/cosyvoice2_tts.py --setup-only

./scripts/render_audio_rdo.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/score_transcripts.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/download_qwen2_audio.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

./scripts/generate_behavior.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml

export PYTORCH_ALLOC_CONF=expandable_segments:True
export RUN_NAME=exp1_$(date +%Y%m%d_%H%M)_audio_rdo_gate

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

`scripts/run_experiment.py` also exposes named stages, but `all` is intentionally
not wired as a single monolithic GPU run. The expensive path should remain
resumable stage by stage.

### Fast RDO config

For initial direction checks, use:

```text
configs/experiments/exp1_refusal_cone_drift_fast.yaml
```

It differs from the full config only in runtime-heavy RDO/stat settings:

```text
hidden.layers = [12, 16, 20]
hidden.positions = [first_generation_prelogit]
rdo.train_steps = 50
rdo.limit_per_site = 10
baselines.random_vectors = 4
stats.n_permutations = 1000
stats.n_bootstrap = 500
```

This gives 3 RDO candidate sites. Each site uses 50 x 10 = 500 training
microbatches, plus limited validation intervention generations. Expected A40
wall time after model load is roughly 1-2 hours. It is a direction-check config,
not the paper-facing final run.

## Current Implementation Status

Current server-oriented implementation snapshot, 2026-07-05:

- Raw data, cache, and run outputs default to `/workspace/audio_safety_data`.
  The git checkout remains `/workspace/audio-safety`.
- Base project dependencies are managed with `uv sync`; GPU dependencies use
  `uv sync --group gpu`. The only intentional isolated virtualenv is the
  CosyVoice2 adapter under the data/cache workspace, because its dependency set
  conflicts with the main Qwen2-Audio environment.
- OpenRouter pair generation is resumable. Per-row OpenRouter failures are
  written to a sidecar `.errors.jsonl` instead of aborting the whole run, and a
  later successful retry clears that stale sidecar entry.
- Current fast gate uses two styles, `neutral` and `sad`: 150 pairs x
  harmful/benign x 2 styles = 600 wav files. Broaden the style list only after
  the RDO direction is promising.
- CosyVoice2 rendering uses `scripts/cosyvoice2_tts.py --batch-jsonl`; the model
  is loaded once and pending wav files are generated from a JSONL job queue.
- ASR transcript control is currently `dataset.asr.mode: skip`. The
  `score_transcripts.py` stage remains in the pipeline only to produce the
  downstream scored manifest with `transcript_control_skipped=true` and
  `transcript_control_passed=true`.
- Qwen2-Audio processor calls use `audio=` plus `sampling_rate=`. The previous
  `audios=` warning indicated the processor was ignoring audio input and is not
  acceptable for real behavior generation.
- `generate_behavior.py` is resumable row by row, has tqdm progress, and supports
  `--overwrite` when behavior outputs must be regenerated after an inference fix.
- Qwen decoder layers are resolved through `model.language_model.layers`; current
  Qwen2-Audio exposes 32 decoder layers. The configured sweep is layers
  `[8, 12, 16, 20, 24, 28]` x positions `assistant_start_pre` and
  `first_generation_prelogit`.
- RDO training was made A40-safe: gradients are accumulated with one backward per
  training microbatch, retain KL is computed only at the intervention token, and
  residual hooks avoid in-place activation edits. This prevents the previous
  graph-retention OOM on a 44GB A40.
- Residual intervention hooks accept both trainable torch tensors and saved numpy
  axes, so training and validation/evaluation use the same hook path.

Recent committed baseline:

- `7181586 Allow skipping ASR transcript control`
- `bc14bb8 Handle OpenRouter pair generation failures`
- `f4a31c9 Make audio RDO setup reproducible`
- `723c27d Pin GPU dependencies to PyTorch CUDA 12.8`
- `35a440a Add Audio-RDO experiment context`
- `cc8574f Implement staged Audio-RDO data pipeline`
- `95c4ef2 Add Audio-RDO refusal axis gate`

Current local verification after the A40 RDO memory fix:

```bash
uv run pytest
# 51 passed
```

A one-site GPU smoke run with `train_steps=1`, `limit=1`, layer 8, and
`assistant_start_pre` completed and wrote `rdo_axis.npz`. This is a smoke test
only; it is not an experiment result.

## Known Boundaries

- The full train -> activation extraction -> heldout evaluation run is still the
  experiment result path. The documented GPU smoke only proves the RDO code path
  can execute without the previous A40 OOM/autograd failures.
- The TTS engine itself is not vendored into this repo. The repo provides a
  reproducible CosyVoice2 setup/render adapter, and the GPU environment downloads
  the external repo/checkpoint into `/workspace/audio_safety_data/cache`.
- OpenRouter model availability and pricing can change. The config contains the
  current default and fallback, but the cloud run should confirm availability.
- Heuristic labeling is intentionally lightweight. Ambiguous harmful-compliance
  rows are marked for manual review instead of being treated as final judge
  labels.
- ASR transcript control and style-classifier enforcement are disabled for the
  current fast gate. Decoding-failure filtering still applies; transcript/style
  validation should be restored before making a stronger final paper claim.

## Go / No-Go Reminder

Strong GO requires all of:

- Style effects remain after excluding decoding failures.
- RDO audio axis passes addition, ablation, and benign-retention checks.
- RDO audio axis beats MDSteer-c2r and SAR-style text vectors at matched ORR.
- Benign-controlled style escape predicts compliance and coordinate restoration
  causally restores refusal without materially raising benign ORR.

No-Go or pivot if:

- RDO behaves like noisy DIM steering.
- Harmful refusal rises only by raising benign over-refusal.
- Style effects are mostly decoding failures.
- Occupancy correlation exists but coordinate restoration fails.
- SAR-style text vector dominates at matched ORR.
