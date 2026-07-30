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
