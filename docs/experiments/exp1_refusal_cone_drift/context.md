# Audio-RDO Gate Context

Last updated: 2026-07-07

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
- **Style set and content-preserving expressive variation:** based on
  StyleBreak's paralinguistic/extralinguistic style axis framing. As of
  2026-07-07, the exploratory style condition is no longer strict same-transcript
  acoustic-only TTS; it allows controlled affective rewrites that preserve the
  request content while adding stronger spoken style.
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
- As of 2026-07-07, compare `original`/`neutral` against `sad` and `angry`
  variants first. Refinements such as more style classes, human ABX, and stronger
  ASR/style filtering come after this simpler pivot is validated.
- The new style claim must be described as **content-preserving expressive style
  rewrite + acoustic style**, not as a pure same-transcript prosody intervention.

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
  --limit 150 \
  --style-variants \
  --style-safety-label both

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

### Fast RDO result: `exp1_fast_20260705_0702_audio_rdo_gate`

Run date: 2026-07-05. Implementation commit: `8051c84`.

Selected site:

```text
layer = 16
position = first_generation_prelogit
```

Validation-site selection looked promising on the small fast validation subset:

```text
add_rr_pp = +20.0
benign_orr_add_pp = +0.0
ablation_asr_pp = +10.0
score = 30.0
n_add = n_benign = n_ablate = 10
```

Heldout final gate was `NO-GO` because addition did not clear the preregistered
+20pp refusal-recovery threshold:

```text
add_rr_pp = +11.8          # below +20pp threshold
benign_orr_add_pp = +2.6   # within <= +3pp threshold
ablation_asr_pp = +21.5    # clears +10pp threshold
rdo_beats_mdsteer_c2r = true
rdo_beats_sarsteer_text = true
```

Interpretation:

- The run is not a paper-facing GO. The final decision is `NO-GO` because the
  heldout addition effect is too weak.
- It is not a dead result. The ablation effect is strong and benign ORR remains
  controlled, which suggests the selected direction is related to refusal rather
  than being a generic refusal-everything vector.
- RDO beats the implemented MDSteer-c2r and SARSteer-style text baselines at the
  observed benign ORR level, but the absolute RDO addition effect is still below
  the gate.
- The current two-style setup does not support a style-escape claim. Heldout
  neutral vs sad behavior is nearly flat (`genuine_style_gap_pp = -1.7`), escape
  metrics are weak (`Spearman = 0.097`, `AUROC = 0.556`), and coordinate
  restoration does not recover refusal (`restoration_rr_pp = 0.0`).

Working conclusion: weak positive for an audio-RDO refusal direction, negative
for the full gate and negative for the current style-escape/restoration claim.
The next practical run should thicken the layer-16 neighborhood before attempting
the full 12-site sweep, for example layers `[14, 16, 18, 20]`,
`first_generation_prelogit`, `train_steps=100..150`, and `limit_per_site=20..30`.

### 2026-07-07 style-pivot conclusion

The same-transcript CosyVoice2 neutral-vs-sad condition was too weak for the
style claim. In the fast run, sad was not easier to attack than neutral
(`harmful:neutral` compliance 28/60 vs `harmful:sad` compliance 27/60), and the
hidden-state escape/restoration metrics also failed. Therefore the current paper
claim should not say that a pure sad prosody shift was enough to create refusal
escape.

The next claim is narrower and more realistic:

> For LALM safety, content-preserving expressive style changes can combine
> lexical/pragmatic tone and acoustic prosody to move harmful audio behavior and
> refusal-coordinate occupancy.

Operationally, the pipeline now creates non-neutral style inputs in two steps:

1. OpenRouter rewrites each selected prompt into `sad` and `angry` variants while
   preserving the original request content and forbidding added operational
   detail.
2. CosyVoice2 renders those rewritten prompts with matching stronger style
   instructions.

Pilot check on 2026-07-07:

- OpenRouter model tested: `z-ai/glm-5.2`.
- Pilot shape: 5 seed rows x 2 styles (`sad`, `angry`) = 10 rewrites.
- Result: 10/10 JSON success, 0 provider refusals, 0 self-reported added
  operational detail, and all outputs reported high content preservation.
- Qualitative weakness: angry variants can overuse impatience markers and sad
  variants can introduce personal affect. The strengthened prompt now forbids
  new backstory, threats, coercion, new urgency, and new operational specifics.
- Cost estimate at the 2026-07-07 checked OpenRouter price for `z-ai/glm-5.2`:
  generating 150 prompts x 2 styles is expected to be well below USD 1, with the
  observed estimate around USD 0.21-0.26.

Protocol note: this pivot is not the preregistered strict same-transcript H3/H4
condition. If used as a paper-facing main claim, the methods/results text must
explicitly state that the intervention is **content-preserving expressive rewrite
plus acoustic TTS style**. A strict acoustic-only claim would require a separate
TTS system or a new run with stronger validated same-transcript style control.

## 2026-07-07 literature survey and NO-GO root-cause diagnosis

After the fast-run NO-GO, four parallel web surveys were run (refusal-direction
literature, audio-LLM safety novelty check, activation-steering methodology, and
venue/framing strategy including an analysis of 493 reviews from ICLR 2026
activation-steering submissions). The consolidated conclusion is that the NO-GO
is most likely an **intervention-operator artifact, not evidence that the axis is
absent**.

### Diagnosis: single-token-position intervention is the suspected root cause

- Our current intervention edits only the last prompt token at a single layer.
  `models/qwen2_audio.py:generate_audio_response_with_intervention` resolves one
  absolute `token_index` and installs a single `ResidualStreamIntervention`
  (`models/hooks.py`), so during KV-cached decode steps the absolute index falls
  outside the length-1 step and the hook becomes a no-op. Generated tokens only
  receive the edit indirectly through attention over one patched position.
- Every steering method surveyed applies addition at **all token positions**
  (single layer), and ablation across all layers/positions:
  - Arditi et al., *Refusal Is Mediated by a Single Direction*, NeurIPS 2024
    (arXiv:2406.11717): addition at one layer, **all token positions**; ablation
    at every layer and position.
  - Wollschlaeger et al. (RDO / *Geometry of Refusal*, ICML 2025,
    arXiv:2502.17420): explicitly "follow common practice to apply both
    operations across all token positions."
  - CAA (arXiv:2312.06681), BiPO (arXiv:2406.00045), RepE (arXiv:2310.01405),
    and even SARSteer itself add per-layer at all generated-token positions.
  - arXiv:2509.12065 shows last-prompt-token-only steering can yield **zero**
    downstream effect while all-position steering succeeds.
- Coefficient is likely under-scaled: alpha=2.0 on a unit vector is ~1-5% of the
  mid-layer residual norm for a 7B Qwen model. Qwen-family effective steering
  factors were ~100-800 vs ~5-40 for Llama in arXiv:2509.12065. Arditi scales
  the addition coefficient to the real refusal-coordinate magnitude
  (`avg_proj_harmful`); RDO scales to the DIM norm.
- The strong-ablation / weak-addition asymmetry is a known pattern: RDO
  down-weights its addition loss to 0.2, and ACE (arXiv:2411.09003) explains it
  as blind addition assuming a wrong (origin-centered) baseline, where ablation
  is self-calibrating. So a +21.5pp ablation with a controlled benign ORR is
  consistent with a real, but narrowly-driven, refusal coordinate.
- Style-escape/restoration (H3/H4) were **untested, not refuted**: the fast run
  used only {neutral, sad}, sad produced no genuine gap, so there were zero
  style-induced compliance samples for restoration to recover.

### Novelty and positioning (survey result)

- The exact combination "gradient-optimized, audio-conditioned refusal axis in
  Qwen2-Audio validated by causal steering" appears **unoccupied** as of
  2026-07. Nearest neighbors use difference-in-means / SVD and are mostly
  observational or on other models: Roh and Houmansadr (arXiv:2604.16659,
  DIM/observational, not Qwen2-Audio), Omni-Safety/OmniSteer (arXiv:2602.10161),
  Safety Geometry Collapse/ReGap (arXiv:2605.18104).
- SARSteer (arXiv:2510.17633) is under review at ICLR 2026 with borderline
  ratings [6,4,6,2]. Its negative claim ("audio DIM steering fails, no shared
  harmful/safe subspace") is our direct foil; a gradient-optimized axis that
  succeeds where DIM fails is a clean correction-style contribution.
- StyleBreak (arXiv:2511.10692) is accepted at AAAI 2026, so the style-behavior
  effect is taken, but a representation-level "style -> refusal-axis occupancy"
  mechanism is still open.
- Cleanest framing (audio analog of *Geometry of Refusal*): "Why audio steering
  was thought to fail, and what actually works: gradient-optimized refusal axes
  in audio LLMs."

### Reviewer-derived pre-submission checklist (from ICLR 2026 review analysis)

Missing baselines (43% of steering reviews), single-model risk, judge
reliability, layer/strength sensitivity, and TTS realism dominate. Concretely:
add a random/orthogonal-direction specificity control; use >=2 judges plus a
human-agreement subset (not keyword-based refusal detection); add >=1 additional
LALM or an explicit scoping argument; validate TTS against real speech; report
benign over-refusal explicitly.

### Next-action plan (agreed direction, not yet executed)

1. **Highest-leverage fix — intervention scope.** Change addition/ablation to
   apply at **all generated token positions** (make the hook survive KV-cached
   decode steps), match train and eval intervention scope, and sweep alpha at
   the natural refusal-coordinate / DIM-norm scale instead of a fixed 2.0 on a
   unit vector. Add an Arditi-style coordinate-clamp (`h <- h - (h.u)u + c u`)
   variant. Report single-position vs all-position side by side so the
   intervention-breadth characterization becomes a contribution rather than a
   silent fix.
2. **Style set.** Replace {neutral, sad} with StyleBreak-effective styles
   (child_female / elderly_male / fearful) to first establish a genuine
   neutral-vs-style behavior gap before re-testing restoration (H4).
3. **Process gate before re-running.** research-code-reviewer on the hook change
   -> `uv run pytest` -> `/codex-cross-check` on the numbers ->
   adversarial-reviewer, per the CLAUDE.md workflow.
4. **Preregistration integrity.** The intervention-operator change is a §5
   *operator-definition* correction (align with literature all-position
   standard), not a §0 threshold change. Record it in design.md's change log and
   report both operator variants. §0 GO/NO-GO thresholds remain untouched.

### Venue timing (verified where possible, 2026-07-07)

- Primary target: **ICLR 2027**, full paper approx Sept 23-24 2026 (inferred from
  the 2026 pattern; official 2027 dates not yet posted).
- Alternatives with runway: ICASSP 2027 (Sept 16 2026), Interspeech 2027
  (approx Feb-Mar 2027), NeurIPS 2026 workshops (approx Aug 29 2026, for an early
  timestamp), TMLR (rolling; best fit for an honest necessity-strong /
  sufficiency-partial result). AAAI-27 (Jul 28 2026) is too soon.

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
- OpenRouter style-variant generation is also resumable and now lives in the
  same data-preparation path as pair generation:
  `./scripts/prepare_audio_rdo_pairs.py --style-variants`. It writes
  `text/figstep/audio_rdo_style_variants.jsonl` plus a sidecar error manifest,
  and `render_audio_rdo.py` automatically uses valid `sad`/`angry` variants when
  they exist.
- Current pivot uses three styles, `neutral`, `sad`, and `angry`: 150 pairs x
  harmful/benign x 3 styles = 900 wav files. Non-neutral styles can be backed by
  `text/figstep/audio_rdo_style_variants.jsonl`, generated through OpenRouter.
- CosyVoice2 rendering uses `scripts/cosyvoice2_tts.py --batch-jsonl`. Run
  `./scripts/cosyvoice2_tts.py --setup-only` once before rendering so repo/venv
  and checkpoint setup are not raced by parallel workers.
- On the RTX A5000 config, `render_audio_rdo.py` shards pending TTS jobs into 2
  worker JSONL files and launches 2 long-lived CosyVoice2 processes on
  `CUDA_VISIBLE_DEVICES=0`. If 24GB VRAM leaves headroom, raise with
  `--override dataset.tts.batch_workers=3`; if contention appears, lower to `1`.
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
# 52 passed
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
