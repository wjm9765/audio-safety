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
| full phase pilot (500 pairs) | **not started** |

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
