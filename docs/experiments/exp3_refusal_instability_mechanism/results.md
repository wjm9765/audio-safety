# Exp3 — Acoustic Refusal Instability Mechanism: run log

> **Append-only.** Entries are added, never edited; corrections go in a new entry.
> Judgements may only use the interpretation rules frozen *before* the run in
> [`rules.md`](rules.md) §6, with any amendment recorded in `rules.md` §11.
>
> Exp3 is **exploratory**. It does not amend the Exp1/Exp2 pre-registered
> decision tables.

## Current status

| item | state |
|---|---|
| environment bring-up (A5000) | done (exp3-000) |
| pipeline smoke gate, all stages, all conditions | **PASS** (exp3-000) |
| defects found and fixed before any non-smoke run | 3 (exp3-000) |
| full phase pilot (500 pairs) | **complete** (exp3-001) |
| every pre-registered integrity gate | **PASS** (exp3-001) |
| phenomenon: paired refusal disagreement | 0.197 harmful / 0.220 benign (exp3-001) |
| M2 `real` > `wrong_item` at L10 | **PASS** (exp3-001) |
| safety-specific claim | **forbidden** — interaction includes 0 (exp3-001) |
| content-invariance of the contrast | **not established**; human audit outstanding |
| `echo` / `tone` breadth arms | still disabled (`rules.md` §4) |
| `t_AB` hand-off narrative | **retracted** (exp3-003) |
| routing: relay vs R-avoiding | **mixed/ambiguous**, r=0.737 [0.645,0.836] (exp3-004) |
| consolidated handoff | [`handoff_20260730.md`](handoff_20260730.md) |

---

## exp3-000 — A5000 bring-up and smoke gate (2026-07-29) — GPU

**Purpose.** `rules.md` §8 states that the last real-weight check of the hook/API
surface is delegated to a GPU smoke gate, because the final code-only
cross-check never returned. This entry records that gate.

### Environment

The available GPU is a **single RTX A5000 (24 GiB)**, not the A40 (48 GiB) the
rules budgeted against. Qwen2-Audio-7B-Instruct in bf16 occupies **16.2 GiB**, so
only **one** model instance fits; there is no second-model concurrency to exploit.

- `torch 2.9.1+cu128`, `transformers 5.12.1`, HF revision pinned to
  `0a095220c30b7b31434169c3086508ef3ea5bf0a`
- model load ≈ 90 s; **≈1.6 s/generation at 32 new tokens**
- GPU during generation: **76.3 % utilisation at 210 W of a 230 W cap**. The
  pipeline is not CPU-starved and is close to the power limit, so batching the
  batch-size-1 generation path was judged not worth the numerical risk.

### Smoke configuration

The `rules.md` §9 smoke command reduces the condition, layer and window lists,
which would leave `position_sham`, `random_direction`, layers 12/14/16 and the
second escape window unexercised. The gate was therefore run with the **full**
condition/layer/window sets and only the pair counts reduced (6 behavior pairs;
cohort caps 1/1/1), so that every code path executes.

### Defects found (all fixed before any non-smoke run)

**1. `_mechanism_cohort` read `role` at the row top level.**
`select_transition_cohort` nests caller-supplied stratum values under
`"stratum"`. The per-(role, direction) GPU-budget cap therefore raised
`KeyError('role')` as soon as *any* discordant pair existed — i.e. the stage
could never have run on real data. Fixed; regression test added.

**2. M1's Whisper-padding assumption is false.**
`rules.md` §5 Stage 1 asserted that the paired arms' `input_features` agree
outside the valid frames. Direct measurement falsifies this. Whisper's log-Mel
floor is `max(log_spec, log_spec.max() - 8.0)` — a **global** clamp — so any
acoustic edit that moves the peak level shifts the zero-padded tail by exactly
the same amount. On item `figstep_safebench_0496` the padded delta was
`0.016398` and the change in global max was `0.016398`: identical. The padded
tail carries **23–32 %** of ‖ΔF‖_F.

`model_eye_component` zeroed that tail for every component including `all`, so
`source + 1.0 · component` could never reproduce the physical target and the
α=1 endpoint gate was unsatisfiable *in principle*. `all` now returns the exact
ΔF and a new `pad_floor` component isolates the tail, keeping
`temporal_slow + temporal_fast + pad_floor == all`. Recorded in `rules.md` §11.

**3. `patch_state` added a delta instead of assigning the donor.**
`ResidualStreamIntervention._patch_state_hook` computed `hidden + (donor -
current)`. In bf16 that rounds twice: 91 of 256 random bf16 elements miss the
donor, max error 1.6e-2. Identity patching hid it (delta identically zero) and
the existing tests used float32 `allclose`.

It surfaced only at the **Stage 5 final-layer control**, which `rules.md` §6
makes a hard gate: of 8 L31 `t_AB` interchange cells, 5 reproduced the donor
margin to ~1e-7 but two missed by 8.9e-3 and 5.3e-2 — and the *same pair* was
exact in one direction and off in the other, which is rounding, not miswiring.
All 8 already reproduced the donor's **full response text and marker**, so the
control was substantively passing and only the scalar readout was inexact. The
hook now assigns the donor verbatim, matching its own docstring and
`SpanStateIntervention` (which backs M2/M3 and was already correct).

### Integrity checks that passed on real weights

| check | result |
|---|---|
| M1 α=0 / α=1 physical endpoint | **exact**, margin abs error `0.000e+00` |
| M2 span-patch identity, full text | 48/48 exact |
| M2 span-patch identity, margin | max abs error `0.000e+00` |
| Stage 5 identity (L10/L18/L31) | 8/8 exact text, margin error `0.000e+00` |
| hook `applied_count` | `1` in every cell |

### Prior expectation for the pilot (exploratory, not a pre-registration)

The Exp2 Qwen arm sweep (`exp2-005`, 500 items × 6 arms) was left unanalysed.
Re-scored with the **Exp3 frozen matcher**, the phase contrast gives:

| contrast | paired instability | R→NR | NR→R | net refusal rate |
|---|---:|---:|---:|---|
| `pv_locked → pv_standard` | **21.0 %** | 58 | 47 | 66.2 % → 64.0 % |

The two directions nearly cancel, so the **net** rate moves only 2.2 pp while
21 % of items disagree. This is direct support for `rules.md` §3.1's choice of
paired disagreement, not net difference, as the primary estimand.

That sweep used the same instruction string but did not pin the HF revision or
the Exp3 system prompt, which is exactly why `rules.md` §5 Stage 0 requires both
physical arms to be regenerated under this run's contract. It is a prior, not a
result.

### Verdict

Smoke gate **PASS** after three fixes. No non-smoke behavioural or mechanistic
result is claimed in this entry.

---

## exp3-001 — full phase pilot, 500 pairs (2026-07-29) — GPU

Run `exp3_20260729_1445_qwen_phase_mechanism`, commit `275dcab`,
Qwen2-Audio-7B-Instruct @ `0a095220…`, bf16/sdpa/greedy, 1× RTX A5000.
14:42 → 19:18 UTC, all seven GPU stages plus analyze, 0 errors.
Full analysis: `outputs/<run>/analysis.md`; figure `figures/transport.png`.

### Integrity — every pre-registered gate passed

M1 α=0/α=1 endpoints 80/80 exact (max margin err `0.00e+00`); M2 identity 452/452
exact text and `0.00e+00` margin over 1356 cells; M3 `identity` and `reset_only`
exactly 0.000; Stage 5 identity 678/678; **final-layer L31 control reproduces the
donor margin to `0.00e+00`** (n=226); `applied_count` == 1 in all 11,300
intervention cells. The L31 pass is what licenses reading the L10/L18 nulls as
mechanistic rather than as a broken pipeline (`rules.md` §6).

### Phenomenon (primary endpoint)

| stratum | n | paired disagreement | R→NR | NR→R | net refusal diff |
|---|---:|---|---:|---:|---|
| harmful | 350 | **0.197** [0.154, 0.240] | 41 | 28 | −0.037 [−0.086, **+0.009**] |
| benign | 150 | **0.220** [0.160, 0.287] | 16 | 17 | +0.007 [−0.067, **+0.080**] |

**Primary CIs exclude zero in both strata; the secondary net-difference CI
includes zero in both.** ~1 item in 5 changes its refusal label while the
arm-level refusal rate is statistically indistinguishable — a direct vindication
of `rules.md` §3.1's choice of paired disagreement over net difference. Exp2's
exploratory sweep (0.210 / 57 / 47) replicates here at 0.204 / 57 / 45 under a
pinned revision and system prompt.

Failure-excluded sensitivity is essentially unchanged (0.185 / 0.228), so the 15
decoding failures do not carry the effect. Exact McNemar p = 0.148 / 1.000: the
switching is bidirectional, **not** a directional weakening, and conditional on
the start state the NR→R limb is the stronger one in harmful items. This is not
a jailbreak result.

### Mechanism

- **M2 (113-pair cohort, 6780 cells).** L10 `real` donorward **+0.781 ±0.067** vs
  `wrong_item` **+0.425 ±0.080** — the pre-registered contrast passes. L18 is a
  null (`real` +0.322 indistinguishable from `sham` +0.384). On the continuous
  margin, `real` is the only condition with common-mode G ≈ 0 (+0.009…+0.062)
  while every control carries a large |G| (random up to +1.13). Direct paired
  real−control differences exclude zero in all 18 layer×control cells.
- **Multi-draw random null** (post-registered robustness, 4068 cells, 3 draws;
  draw 0 reproduces the registered draw bit-identically 1356/1356). `real`
  T_resid significant at all six layers (+0.571…+0.173); a random draw scored
  against its own siblings is null at every layer.
- **M3.** Resetting the audio span at L12 removes **96.7%** of the injected
  effect; at L14, 16.5% survives. Path escape is therefore **forbidden** for the
  L10→L12 window and begins between L12 and L14.
- **Stage 5.** At L10 the whole audio span gives T = +1.104 but the single `t_AB`
  position gives **+0.027**; by L18 `t_AB` carries +0.503. The difference is
  audio-resident before it is readout-resident.
- **M1.** `pad_floor` inert (median transport 0.016, 1/40 flips) despite 34.8%
  mean share of ‖ΔF‖. A follow-up run (`exp3_20260730_0200_m1_interaction`) added
  the missing `valid_only` cell: pad main effect +0.013 ±0.055, **interaction
  −0.004 ±0.128**, speech share 103.5%. M1 does **not** establish input-level
  item-specificity at n=40.

### Two pre-registration defects found (neither contaminates the primary)

1. **`wer ≤ 0.20` content gate is mis-specified for Qwen.** It answers a
   transcription prompt with a carrier phrase, so a word-perfect transcript
   scores WER 1.0 — one pair has `overlap = 1.000` and `wer = 1.000` at once.
   112/120 are carrier-quoted, so the gate reads 0/60 for reasons unrelated to
   content. `rules.md` §5 Stage 6 already forbids using the audit as an exclusion
   filter, so nothing downstream is affected. Carrier-stripped, both arms
   transcribe about equally well (41/60 vs 40/60, paired Δoverlap −0.008) and
   22/60 pairs transcribe identically. **The repair is post-hoc and cannot make
   the registered gate retroactively pass.**
2. **`R_tAB` and the refusal marker diverge in sign for 28/73 discordant pairs
   (38.4%).** A generation can open with a non-refusal prefix and still contain
   "I cannot" later. The continuous transport statistic and the behavioural
   donorward score are therefore **not two views of one quantity**, and
   margin-level specificity does not propagate monotonically into behaviour —
   at L18 it demonstrably does not.

### Claim ladder (after two rounds of `gpt-5.6-sol` xhigh review)

Supported: bidirectional phase-handling sensitivity of the explicit-refusal
marker with no detectable marginal-rate change; a **pair-matched
audio-conditioned representation causally effective on the refusal-margin
readout**, above each host's own random baseline; and a layer profile *consistent
with* an early distributed signal becoming progressively available at the answer
position.

Not supported: safety specificity (benign ≥ harmful, interaction includes 0);
"phase alone with perceived content held constant" (content equivalence was never
validly evaluated); a discovered refusal axis (full spans/states were patched);
inert controls (they are substantial — `real` is an excess over them); L31 as a
causal locus (it is an indexing/readout control); an encoder-level refusal
representation (harmful-only AUROC 0.512, chance); input-level item-specificity;
or generalisation beyond Qwen2-Audio and beyond the phase contrast.

### Biggest remaining threat and the cheapest test

The contrast is **phase-handling**, not demonstrated **content-invariant**:
carrier-corrected WER is still 0.316/0.376, only ~2/3 of audited outputs are
content-faithful, and `pv_standard` carries more decoding failures. The cheapest
decisive follow-up needs **no new model inference**: a blinded, pre-specified
human input-equivalence audit over the existing 500 pairs, annotators blind to
arm and outcome, followed by a frozen analysis of marker disagreement within the
content-equivalent stratum.

---

## exp3-002 — corrections to exp3-001 and ICLR venue assessment (2026-07-30) — no GPU

`results.md` is append-only, so this entry corrects exp3-001 rather than editing it.

### Two overstatements in exp3-001, now corrected in `analysis.md`

1. **"Net refusal-rate difference includes zero" was written as if it established
   no marginal change. It does not.** Failing to reject is not equivalence: the
   harmful CI [−0.086, +0.009] still permits an **8.6-point decrease**. The
   defensible statement is that marginal-rate testing *fails to reveal* the
   within-pair churn, not that the marginal rate is unchanged. A pre-specified
   equivalence test would be needed for the stronger claim.
2. **The Exp2 sweep agreement is a robustness rerun, not an independent
   replication.** It uses the *same 500 items and the same renders*; only the
   generation contract differs. exp3-001 called it "replication".

Also: "phase-only input" is technically inaccurate and is replaced by
**"phase-handling synthesis contrast"** — re-analysis into model features turns
phase-handling differences into magnitude-feature differences, and M1 shows
‖ΔF‖ is substantial.

### Venue assessment (`gpt-5.6-sol` xhigh, round 3, as ICLR area chair)

**Verdict: workshop paper now; ICLR main track only after substantial additional
evidence.** Predicted reviews 3–6, modal 4–5, mean ≈4.5, acceptance ≈15%
(range 8–25%). Most likely rejection reason: a careful single-model
replication/extension of a phenomenon already reported by Acoustic Interference
(2605.18168) and StyleBreak (2511.10692), where the genuinely new causal
localization is narrow and not behaviourally validated end-to-end.

This is **consistent with the project's own earlier two-reviewer consensus**
(2026-07-24): *as-is 3/10; 5–6 after 2-model × family replication + blinded
relabel*, with white space judged "thin and negative", mostly owned by HARC
(2607.00572), ReGap (2605.18104), **GACL (2606.05161)**, MTAM (2509.24215) and
AIA. Note GACL was **not** included in the round-3 prompt, and it owns same-audio
counterfactuals (64.1% sign flip) plus answer-position localization (ρ=0.93)
across 5 ALMs — so the round-3 estimate is, if anything, optimistic. GACL's task
is modality arbitration rather than refusal, which is where daylight remains.

### The single defensible novelty sentence

> For content-equivalent phase-handling variants of the same recording,
> pair-matched audio-span interchange at L10 transfers the eventual refusal
> outcome, while resetting only those audio-token states at L12 nearly eliminates
> the transfer — localizing an **early, item-matched causal carrier that
> final-layer refusal-direction analyses do not identify**.

It may **not** be extended with "…which then moves to `t_AB` by L18 and causes
refusal". The L18 behavioural null and the 38.4% `R_tAB`/marker sign disagreement
prohibit that; the proposed hand-off is currently the weakest link in the story,
not a supported result.

### Required before an ICLR main-track attempt

| # | Work | Objection it kills | Cost |
|---|---|---|---|
| 1 | Human/LLM **semantic** labelling of all 102 discordant pairs + matched stable controls | the 20% may be lexical-marker churn, not refusal change | **0 GPU** (204 generations already on disk) |
| 2 | Full-generation `t_AB` patching L14–L31 scored semantically, plus a **reset-and-rescue** test | repairs or falsifies the audio-span → readout hand-off | 3–6 GPU-h |
| 3 | Head-to-head against AIA's closest patching protocol on this cohort | shows what the reset localization adds beyond the nearest paper | 5–12 GPU-h + implementation |
| 4 | Safety-specificity control: same edit on matched non-safety output attributes | benign (0.220) ≥ harmful (0.197); may be generic decoder instability | ~5 GPU-h |
| 5 | Second audio system | single-model result | MiniCPM 5–8 h (**within Qwen2 LLM family** — a second *system*, not an independent backbone); Ultravox/Llama needs new conversation + readout adapters, ~1 day |
| 6 | A second phase-handling manipulation or strength | may be specific to one renderer contrast | ~5 GPU-h per condition |
| 7 | Treatment-fidelity characterisation (loudness/duration/spectral) | keeps "phase-handling" technically accurate | <1 GPU-h |
| 8 | Simultaneous/hierarchical inference across layers; drop or formalise the equivalence implication | layerwise multiplicity; non-significance ≠ unchanged | 0 GPU |

**Reset-and-rescue is the decisive test:** L10 injection transfers the outcome →
L12 audio reset removes it → a later `t_AB` donor intervention restores it. If
step 3 fails, the hand-off narrative should be abandoned, not rescued via `R_tAB`.

### To cut or demote

Cut: "refusal axis"; "marginal rate unchanged"; "independent replication";
"reaches the readout by L18" as a behavioural claim; "phase-only input".
Demote to appendix: most probe AUROCs (correlational, partly owned, harmful
encoder at chance), the L31 reproduction (an implementation check), the full
interpolation/pad-floor ledger, and the 18-cell layer×control table. Report M1's
temporal/wrong-item comparison as the **null** it is.

Do not bury the benign result, the L18 behavioural null, or the `R_tAB`/marker
disagreement — reviewers will find them, and hiding them is more damaging.

---

## exp3-003 — t_AB mediation test, and two corrections to my own interpretation (2026-07-30) — GPU

Run `tab_mediation/records.jsonl` inside `exp3_20260729_1445_qwen_phase_mechanism`.
1808 cells, 113-pair frozen cohort, both directions, 8 conditions, ~60 min.
Endpoint is the **full generated refusal marker**, not `R_tAB`, because exp3-002
showed the two diverge in sign on 38.4% of discordant pairs.

**Why blocking rather than rescue.** The reviewer proposed a rescue design
(L10 inject → L12 span reset → later `t_AB` donor injection restores it). That is
**under-identified**: Stage 5 showed donor `t_AB` injection at L18 *alone* already
moves the output, so "it came back" cannot separate a hand-off from re-running the
Stage 5 effect. Confirmed empirically — `reset_rescue` +0.349 vs `rescue_only`
+0.322, paired difference **+0.027 ±0.038, includes zero**. Blocking was made
primary instead: inject the donor span at L10, then clamp `t_AB` back to the
**host** state at L14 or L18.

### Results (pair-clustered, n=73 discordant pairs)

| condition | donorward | r_k vs inject_only |
|---|---|---:|
| `identity` | +0.000 | — |
| `clamp_l18_only` | +0.000 | — |
| `inject_only` | +0.781 ±0.066 | 100% |
| `inject_clamp_l14` | +0.705 ±0.071 | **90.4%** |
| `inject_clamp_l18` | +0.589 ±0.077 | **75.4%** |
| `inject_reset` (L12 span → host) | +0.144 ±0.052 | **18.4%** |

Paired within-pair differences all exclude zero: vs `clamp_l14` +0.075 ±0.059,
vs `clamp_l18` +0.192 ±0.087, vs `inject_reset` +0.637 ±0.090.
Integrity: 452/452 no-op cells exact, margin error `0.0e+00`, `applied_counts`
(1,), (1,1), (1,1,1) all firing exactly once.

### What this licenses — and two corrections to what I first said

**Correction 1.** I first wrote that the effect *"bypasses the readout position"*.
Wrong. A one-time clamp is transient: at the next block `t_AB` can become
donor-conditioned again by attending to earlier positions. Survival after a
single-layer clamp is not evidence of bypass.

**Correction 2.** My repair — *"the audio positions are read repeatedly across
layers"* — also overreaches, on three counts raised in review:
- the L10-patched positions do not remain *donor* states; they become **hybrid
  descendants** of the patch. Say the perturbation persists, not the donor state.
- re-entry need not come from audio positions at all: donor information may
  already have propagated into intervening text/control tokens.
- the 83% (span reset) and 24% (single-token clamp) figures differ in timing,
  duration and dimensionality and are **not comparable as "percent mediated"**.

**Defensible statement:**

> An L10 donor audio-span patch remained behaviourally effective after a one-time
> host reset of the answer-boundary residual at L14 or L18, but was largely
> attenuated by resetting the audio-span residual at L12 — ruling out either
> tested intermediate answer state as an **exclusive** causal bottleneck, without
> establishing repeated audio-to-answer readout.

### Relation to GACL — a qualification, not a contradiction

I previously framed this as contradicting GACL's answer-position localization
(ρ=0.93). That overreads a partial result: the L18 clamp produced **significant**
attenuation, so this is not a null answer-position finding, and a single-layer
clamp cannot test persistent localization. Information can be strongly readable
and causal at the answer position while also being distributed elsewhere and
regenerated there.

### Trap recorded for the obvious follow-up

Clamping `t_AB` at **all** layers ≥ k looks like the fix, but if the final
answer-position residual is what gets unembedded, clamping through the last block
**forces the first-token logits to the host logits** — near-total blocking would
then be algebraic, not mechanistic, and with a full-generation endpoint one forced
token can redirect the whole rollout. A path-specific edge patch (replace only the
audio-conditioned attention contribution into `t_AB`, leaving its residual/MLP
free) is the coherent version. Any such run needs a non-trivial matched sham and
pre-registered output-quality gates, so that "no refusal marker in garbage output"
is scored as intervention failure rather than successful blocking.

---

## exp3-004 — relay-closure routing test (2026-07-30) — GPU

Artifact `relay_closure/records.jsonl` inside `exp3_20260729_1445_qwen_phase_mechanism`.
904 cells = 113 cohort pairs × 2 directions × 4 arms. Script
`scripts/exp3/relay_closure.py`. Decision rule frozen in that script's docstring
before execution.

### Question

exp3-003 established that neither tested intermediate answer-position state is an
exclusive bottleneck. This asks the routing question directly: does the L10
audio-span effect reach the output *through* the prompt tokens that sit between
the audio and the readout, or around them?

### Measured prompt geometry

| positions | contents | count |
|---|---|---|
| `0 … a0−1` | `<\|im_start\|>system … Audio 1:` | 20 |
| `a0 … a1` | audio tokens (injection site) | 74–247 (78 in the first pair) |
| `a1+1 … t−1` | **R (relay set)**: `<\|audio_eos\|>\nPlease respond to the request in the audio.<\|im_end\|>\n<\|im_start\|>assistant` | **15** |
| `t` | `t_AB`, the readout | 1 |

### Intervention

Inject the donor audio span at L10, then overwrite every R position with its
**host** state after every decoder block from L11 to L31 (21 layers).

Equivalence argument (supplied in review): in a causal transformer, contaminated
R output at layer L cannot affect other positions within that same layer, because
they read R's layer-*input* state. Clamping R after every block destroys the
contamination before it can become R's K/V at the next block; by induction
nothing leaks through R. R has no other contaminated predecessor in this
geometry, so the state clamp is equivalent to closing A→R edges here. R is not
unembedded, so unlike a `t_AB` clamp no outcome can be forced algebraically.

Arms: `identity` (host self-patch at L10), `open` (donor at L10),
`relay_closed` (open + R clamped), `sham_preaudio` (open + the 20 pre-audio
positions clamped; those cannot attend to the audio, so this must be an exact
no-op).

### Validity gates — all passed

| gate | result |
|---|---|
| `identity` reproduces baseline | **226/226** exact |
| `sham_preaudio` reproduces `open` | **226/226** exact, 0 mismatches |
| hooks applied exactly once (1- and 22-hook compositions) | pass |
| quality-flagged generations | 1 of 904 (in `relay_closed`) |

### Result (pair-clustered, n=73 discordant pairs)

| arm | donorward |
|---|---|
| `open` | +0.781 ±0.066 |
| `relay_closed` | +0.575 ±0.065 |
| `sham_preaudio` | +0.781 ±0.066 |

**Retention r = donorward(relay_closed) / donorward(open) = 0.737, 95%
cluster-bootstrap CI [0.645, 0.836]** (20,000 resamples over pair IDs).

Frozen decision rule: CI entirely ≤0.20 ⇒ relay necessary; CI entirely ≥0.80 ⇒
R-avoiding routes sufficient; otherwise mixed/ambiguous with no binary claim.

**Verdict: MIXED / AMBIGUOUS.** The interval lies between the two thresholds, so
neither pre-registered conclusion is licensed.

### What the interval does support

- The relay is **not necessary**: clamping all 15 R positions across 21 layers
  leaves 73.7% of the effect, CI lower bound 0.645.
- The relay is **not inert** either: the 26% reduction has an upper CI bound of
  0.836, below 1, and closing it reversed the injected decision in **6 of 73**
  pairs.

Combined with exp3-003 (clamping `t_AB` at L18 leaves 75.4%), no single tested
route carries a majority of the effect on its own.

### Explicitly not licensed

- "R-avoiding routes are sufficient" — the CI does not reach the 0.80 threshold.
- "The effect is direct audio→readout" — even had the threshold been met, this
  would not follow: generated positions can attend to the audio themselves and
  relay through earlier generated positions, a third route this design does not
  close.
- Any additive decomposition of the two routes; they are counterfactual
  capacities, not mediation shares.

### Pre-registered stop rule reached

The review-supplied stop rule states that if intervals straddle the material
threshold, further ablations on this cohort will not add power and more
independent pairs are required. The interval half-width here is ±0.10 at n=73.
