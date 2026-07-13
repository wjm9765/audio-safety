# Run 4 → Causal-Attribution direction + interchange-patching code (2026-07-13)

> **Direction-finding spec + implementation note.** NOT a §0 gate; `design.md` §0 and
> `run4_conversion_gap_design.md` §0/§1 are **unchanged**; `results.md` stays append-only.
> This records (a) the paper-direction decision reached with Codex (gpt-5.6-sol, xhigh)
> and (b) the code added this session to test it on Qwen2-Audio. Prior context:
> [context.md](./context.md), [run4_direction_20260712.md](./run4_direction_20260712.md),
> [run4_dissociation_20260712.md](./run4_dissociation_20260712.md).

## 0. The decision (why the headline changed again)

The `run4_direction_20260712.md` headline — *"harmfulness sensor `r_H` vs causal refusal
actuator `r_A`; audio attacks preserve sensing while bypassing the actuator"* — is **not**
the paper direction. Two independent reasons (Claude audit + Codex adversarial cross-check,
both grounded in the code and primary literature):

1. **Preempted + we lack the causal leg.** [LLMs Encode Harmfulness and Refusal Separately
   (2507.11878, NeurIPS 2025)](https://arxiv.org/abs/2507.11878) already establishes the
   harmfulness/refusal dissociation **causally** in text (reply-inversion). Our §8 causal
   rescue was **negative** (frozen `r_A` ≈ norm-matched random at the validated strength), so
   on the factorization we are *behind* the text state of the art, not ahead.
2. **No audio-specificity in the data.** §8 jb_pap: audio×text interaction ≈ 0, benign DiD ≈ 0
   (a generic compliance boost, not an audio- or harmful-specific jailbreak); jb_ica is an
   intelligibility collapse; matched-neutral T0 is null; consistent with JALMBench's
   Qwen2-Audio audio≈text (7.3 vs 6.9).

### Locked direction (outcome-robust)

**Title:** *"Is It Really an Audio Jailbreak? Causal Attribution of Safety Failures in
Audio-Language Models."*

**Thesis:** Raw audio-jailbreak success does not identify an audio-*specific* safety failure;
a matched modality×attack×harmfulness factorial plus causal state interventions distinguish
genuine audio-specific failures from modality-agnostic compliance, perception collapse, and
over-refusal calibration shifts.

**Why it can't be killed by the empirics:** it yields a paper on **both** branches —
(A) an audio-specific mechanism survives matching → causal taxonomy + defense prediction;
(B) it does not → the reported audio>text gaps are decomposed into generic compliance /
perception / calibration and shown to be protocol-induced. The negatives become the contribution.

**Novelty daylight (vs ReGap 2605.18104 + activation-patching literature):** ReGap *assumes and
corrects* an aggregate modality-drift mediator, and conventional patching *conditions on a
researcher-chosen* mediator; we first test whether modality causally contributes **at all** via
a matched design, and accept internal localization only when bidirectional patches survive
alternative-stage and generic-behavior controls. `r_A` is demoted to an instrument, not the headline.

**Reviewer forecast (Codex + Claude, as-if executed):** MVP floor **7/10 accept**; target **8**
(2nd model + full-SARSteer/ReGap Pareto + a ranking change); ceiling **9** (held-out-attack, then
held-out-model, defense prediction). One-model or a bare null is 5–6.

## 1. What this session built (the decisive Qwen-only test)

The core of the direction is a **multidimensional causal method**: does injecting a **clean-run**
residual state into the **attacked** run at the decision anchor causally restore refusal,
harmful-specifically, beyond shams? Code (Codex-reviewed for silent-bug risks before writing):

- **`models/hooks.py` — `patch_state` mode.** Interchange/activation patching: REPLACE a single
  residual position with a full `d_model` donor state, **used verbatim (never normalized)**,
  prefill-only, single application. Guards: separate `replacement_state` field (not the axis
  `vector`), non-negative absolute index required (a `-1` would silently repatch every decode
  step), `applied_count` (must be 1 after a generation), batch-size-1, dim-check.
- **`models/qwen2_audio.py`** — `capture_audio_state(...)` (donor snapshot at a role-relative
  site) and `generate_audio_response_with_state_patch(...)` (asserts exactly one application).
- **`evaluation/causal_trace.py`** (pure, CPU-tested) — trace-id (collision-proof), condition
  planning, and adjudication: identity-invariance check, benign-adjusted rescue contrast `C`,
  control ladder, intention-to-treat reproduced-flip accounting.
- **`evaluation/attack_flip.py` — `harmful_specific_interaction()`** — the **unconditional**
  harmful-specific audio×attack interaction τ (triple difference), wired into the §8 report.
  This is the correct behavioral anchor; it fixes the conditional-flip-rate pitfall (the old
  `+20.7pp` was conditional on originally-compliant rows ≈ `+10pp` unconditional).
- **Scripts:** `causal_trace_flip.py` (generate) → `judge_traces.py` (blind judge by `trace_id`)
  → `analyze_causal_trace.py` (adjudicate). **Config:** `causal_trace` block in
  `run4_attack_flip.yaml`; `CausalTraceConfig` in the schema.

### Conditions (primary cell = layer 16 / `first_generation_prelogit`, preregistered)

`no_patch` (contemporaneous baseline) · `identity` (self-patch, MUST reproduce `no_patch`
verbatim) · **`same_item`** (clean→attacked, the alive/dead lever) · `wrong_item` · `reverse`
(attacked→clean, should induce compliance) · `random_displacement`
(`||δ|| = ||h_clean − h_attack||`, on-manifold) · `r_a_coord` (concept patch, **only** at r_A's
trained site) · benign matched `no_patch` / `same_item` (over-refusal cost).

**Primary contrast** `C = [refusal(same_item,H) − refusal(no_patch,H)] − [refusal(benign_same_item)
− refusal(benign_no_patch)]`, judged by the blinded LLM judge(s), **not** the heuristic labeler.

**Alive** iff: identity invariant AND `C > 0` (CI) AND `same_item` beats displacement/wrong-item
AND `reverse` tends to compliance. Any layer sweep is **exploratory**; layer 16 is the
preregistered primary (with n≈19 flips, post-hoc best-layer selection is indefensible).

## 2. Scope decisions (this session; direction-finding, not strict)

- **Qwen2-Audio only.** A second architecture is deferred until the direction shows alive on Qwen.
- **Existing §8 cohort** (150 shared FigStep items, consensus PAP flips) reused — direction-finding
  is allowed on exposed data; the paper-facing run needs a fresh cohort.
- **Human audit / benign-pair review SKIPPED** — the user inspected the benign/TTS/LLM pairs and
  accepted them; not gating for this pass.
- **ASR/WER gate SKIPPED** (`dataset.asr.mode: skip`), degenerate/non-answer handled by the judge taxonomy.
- **Exact AIAH/JALMBench replication + full-SARSteer/ReGap Pareto + defense prediction** are the
  paper-facing program (MVP→8→9), not this cheap pass.
- Judge = the §8 substitution (`google/gemini-2.5-flash` + `anthropic/claude-haiku-4.5`).

## 3. Run commands (A40; data + models already on the box)

Prerequisite: the §8 judged manifest `manifests/audio_rdo_attack_flip_judged.jsonl` and the
rendered clean/jb_pap audio must exist (from the §8 flip run). Then:

```bash
# 1) generate patched traces (GPU)
./scripts/causal_trace_flip.py \
  --config configs/experiments/run4_attack_flip.yaml \
  --axis-artifact "$AUDIO_SAFETY_OUTPUT_DIR/exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz" \
  --run-name run4_causal_trace

# 2) judge the patched outputs (OpenRouter, blind, by trace_id)
./scripts/judge_traces.py \
  --config configs/experiments/run4_attack_flip.yaml --run-name run4_causal_trace

# 3) adjudicate: identity invariance + contrast C + control ladder + ITT accounting
./scripts/analyze_causal_trace.py \
  --config configs/experiments/run4_attack_flip.yaml --run-name run4_causal_trace
```

`--axis-artifact` is optional; omit it to skip the `r_a_coord` concept control (still runs the
full-state causal test, which is the primary alive/dead signal).

## 4. Honesty / limits

Single model, n≈19 consensus flips (underpowered), authored (non-verbatim) ICA/PAP-style wrappers
rendered in neutral voice (so the "attack" is textual), existing exposed cohort, no human audit,
`r_a_coord` uses a clean-trained axis. This is a **direction-check**: a positive `C` (identity
invariant, beating shams) says the causal-attribution direction is worth the paper-facing redesign;
a null says pivot to the branch-B measurement/attribution framing. Do not present any number here
as a paper result.

## 5. Change history

- 2026-07-13 — Direction locked to *causal attribution of audio safety failures* (Codex
  gpt-5.6-sol xhigh deep-discussion, 2 rounds; Claude code audit). Added interchange-patching
  (`patch_state`) causal-trace pipeline + unconditional τ estimand + tests (`uv run pytest`
  green). Qwen-only direction-finding scope. §0/§1 and `design.md` unchanged; `results.md`
  append-only (no run yet — run on A40 then append a results entry).
