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
called as `./scripts/<name>.py` on a prepared machine.

Current stage order:

```bash
./scripts/prepare_audio_rdo_pairs.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/render_audio_rdo.py --config configs/experiments/audio_rdo_gate.yaml --dry-run
./scripts/render_audio_rdo.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/score_transcripts.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/download_qwen2_audio.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/generate_behavior.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/train_rdo_axis.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/extract_rdo_activations.py --config configs/experiments/audio_rdo_gate.yaml
./scripts/evaluate_rdo_gate.py --config configs/experiments/audio_rdo_gate.yaml
```

`scripts/run_experiment.py` also exposes named stages, but `all` is intentionally
not wired as a single monolithic GPU run. The expensive path should remain
resumable stage by stage.

## Current Implementation Status

Pushed commits:

- `95c4ef2 Add Audio-RDO refusal axis gate`
- `cc8574f Implement staged Audio-RDO data pipeline`

Implemented pieces:

- OpenRouter benign-pair generation helper.
- Command-template TTS adapter with dry-run support.
- Simple ASR/transcript scoring path with WER and harmful-token checks.
- Heuristic 4-way behavior labeler.
- Qwen2-Audio loading, prompt formatting, generation, hidden-state capture, and
  residual-stream intervention helpers.
- Audio-RDO training batches with per-input token positions.
- Candidate site validation, baseline vector construction, activation
  extraction, style escape metrics, restoration metrics, and final gate summary.
- CPU tests for config loading, RDO utility behavior, text scoring, transcript
  scoring, and label heuristics.

Verified locally:

```bash
uv run pytest tests/test_config.py tests/test_audio_rdo.py tests/test_stage_helpers.py
```

Result at implementation time: `18 passed`.

## Known Boundaries

- The actual GPU model run has not been executed locally by design.
- The TTS engine itself is not vendored into this repo. The repo provides a
  command-template adapter; the GPU/cloud environment must provide the concrete
  TTS command.
- OpenRouter model availability and pricing can change. The config contains the
  current default and fallback, but the cloud run should confirm availability.
- Heuristic labeling is intentionally lightweight. Ambiguous harmful-compliance
  rows are marked for manual review instead of being treated as final judge
  labels.
- Style classifier enforcement is off for the first gate. Transcript quality and
  decoding-failure filtering carry the early-stage quality control.

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
