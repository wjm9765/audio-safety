# Exp5 — y1 versus prompt cache: pre-registered design

> **Status:** pre-registered and frozen before the first GPU run, 2026-07-31.
> Thresholds and primary estimands must not be edited after a non-smoke run
> begins; any necessary change is a dated amendment in §8.
>
> **Scope:** direct follow-up to Exp4. The endpoint remains the frozen literal
> explicit-refusal marker, not harmful compliance or jailbreak success. Physical
> `H`/`D` endpoints are measured on the executing device, per Exp4's
> `design.md` §9 amendment.

## 0. Frozen decision table

All intervals are 95% pair-cluster bootstrap intervals (20,000 draws) over the
**changed-`y1` directions** of the current-device discordant cohort. Available
directions are averaged within pair before pairs are resampled. No generated row
is excluded because of answer quality.

Notation `M_{a,t,y}`: `a` = audio-token cache source, `t` = `t_AB` cache source,
`y` = first generated token source; each is `I` (injected trajectory) or `H`
(physical host).

| quantity | definition | material conditional contributor | negligible at pilot resolution | otherwise |
|---|---|---|---|---|
| `D_joint_y1H` | `M_IIH − M_HHH` | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |
| `D_y1_at_HH` | `M_HHI − M_HHH` | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |

Secondary, reported with intervals but **no binary decision**: the route
contrasts under a host first token, `D_audio_y1H = M_IIH − M_HIH` and
`D_tAB_y1H = M_IIH − M_IHH`; the `y1` simple effect at every cache state; and
the three-way interaction.

An interval wholly below zero is reported as a reverse effect. These contrasts
are **conditional causal capacities**, not mediation shares, and they need not
sum to anything.

## 1. Question left open by Exp4

Exp4 (`exp4_20260731_2010_audio_kv`, 904 cells, 69 current-device discordant
pairs, every integrity gate passed) established:

| quantity | estimate [95% CI] | frozen decision |
|---|---|---|
| `D_audio` | +0.123 [+0.065, +0.188] | ambiguous |
| `D_tAB` | +0.029 [+0.000, +0.065] | negligible at pilot resolution |
| `D_joint` | +0.167 [+0.101, +0.239] | ambiguous |
| `T_HH` | +0.413 [+0.333, +0.493] | — |

`T_HH` splits exactly along whether the injected prefill changed `y1`: 0/76
donorward when `y1` was unchanged (an identity consequence of Exp4 gate 8a), and
+0.917 across 48 pairs when it changed. Exp4 held `y1` fixed at the **injected**
token in all four of its cells, so it cannot separate the causal contribution of
that token's identity from the contribution of the cache.

A post-hoc stratification of the Exp4 records shows the aggregate is a mixture of
two regimes:

| stratum | n directions | `M_II` | `M_HI` | `M_IH` | `M_HH` |
|---|---|---|---|---|---|
| `y1` unchanged | 76 | 0.263 | 0.079 | 0.211 | 0.000 |
| `y1` changed | 62 | 0.968 | 0.919 | 0.968 | 0.919 |

In the changed-`y1` stratum every cell sits near ceiling, so cache contrasts are
compressed to ≈0.05. This stratification is **exploratory** and is recorded here
only as the motivation for Exp5; it licenses no claim.

Exp5 asks the question Exp4 conditioned away:

> When the first generated token is reset to the physical-host token, does the
> injected prompt cache still drive the donor refusal marker?

## 2. Frozen 2×2×2 design

Construction of the injected prefill is inherited unchanged from Exp4 §3: donor
audio-span replacement after decoder block L10, with all 15 post-audio relay
positions clamped to their physical-host residual states after every block
L11–L31. Cache surgery is unchanged from Exp4 §4: K and V transplanted together
at the selected absolute positions in every decoder layer, from independent
exact clones.

The single addition is that the first decoded token is supplied from either
trajectory:

| factor | levels |
|---|---|
| audio-token K/V | injected, host |
| `t_AB` K/V | injected, host |
| first token `y1` | injected, host |

giving eight cells per pair × direction. Greedy decoding then proceeds manually
from that token through at most 96 total new tokens, under the decoder contract
fixed in Exp4 `design.md` §9 (resolved generation config, repetition penalty
included, float32 processor-then-argmax). The stored sequence includes `y1`.

The `y1` factor only varies where the injection changed the first token. Where
the injected and host `y1` are the same token ID, the two `y1` levels are
required to produce identical output (gate 6 below), which makes those cells a
correctness check rather than a contrast.

## 3. Integrity gates (fail closed)

A non-smoke run is invalid if any gate fails. Gates 1–5 are inherited from Exp4
§5 and are re-checked here.

1. Source Exp4/Exp3 artifacts hashed before GPU inference; a changed source
   artifact requires a new run name.
2. Model revision, prompt, matcher, cohort, L10 site, directions and the
   measured relay count of 15 are inherited and re-asserted.
3. Every residual intervention hook fires exactly once during prefill.
4. Cloning the injected cache is exact, and after transplanting both `A` and
   `t_AB` from the physical host the entire mixed cache equals the physical host
   cache at tolerance 0.
5. Manual decoding of `M_III` reproduces ordinary greedy `model.generate` under
   the same residual hooks exactly, token for token, independently in every
   loaded model replica.
6. **`y1`-degeneracy.** Wherever the injected and host `y1` token IDs are equal,
   the `y=I` and `y=H` cells of the same cache state must be token-identical.
7. **Host closure.** `M_HHH` must reproduce, token for token, an independently
   recomputed ordinary greedy `model.generate` on the physical-host prompt in
   the same process and replica. Its donorward score is therefore 0 by
   construction, and is not an estimated null.
8. Physical `H`/`D` endpoints are measured on the executing device before any
   cell runs; the frozen-versus-current transition table is recorded.
9. The output checkpoint contains exactly one row per pair × direction ×
   condition cell.

## 4. Endpoints and estimands

For a discordant direction with host/donor markers `H ≠ D` and cell marker
`Y_c`, `donorward(c) = (Y_c − H)(D − H)`, which is 1 when the cell matches the
donor and 0 when it matches the host. Directions are averaged within pair.

Primary contrasts are those in §0. `M_HHH` is fixed at 0 by gate 7, so
`D_joint_y1H` reduces to `M_IIH` and `D_y1_at_HH` to `M_HHI`; both are still
reported as contrasts so the reference is explicit.

Interpretation:

- large `D_joint_y1H`: the injected prompt cache retains causal capacity that
  does not require the injected first token;
- large `D_y1_at_HH` with negligible `D_joint_y1H`: `y1`-dominant conditional
  lock-in at this resolution;
- both large: parallel or redundant capacities, described by the interaction
  rather than by any unique path;
- neither large: no detectable capacity in either factor for the changed-`y1`
  population at pilot resolution.

Unchanged-`y1` directions and stable-control pairs are reported descriptively.
Unchanged-`y1` directions carry no `y1` contrast by construction, but their
cache contrasts are reported because they are the subpopulation in which the
cache is the only channel for injected information.

## 5. What this experiment cannot establish

- It does not identify which audio tokens, layers, heads, or attention edges
  carry the effect; K/V is replaced as whole spans.
- It does not decompose mediation shares. Capacities may be redundant,
  synergistic, or downstream copies of a common signal.
- It does not show that `y1` is necessary, nor that `y1` encodes an abstract
  refusal state rather than lexical autoregressive commitment.
- It does not test safety specificity, perceptual content invariance, other
  models, other acoustic transformations, or other hardware.
- Marker movement is not harmful compliance and not semantic equivalence.
- The primary population is the changed-`y1` subgroup of a device-local
  discordant cohort; it is not a general population estimand.

## 6. Artifacts and commands

Run names use `exp5_{YYYYMMDD}_{HHMM}_{tag}`. Each run writes the Exp4 artifact
layout plus `endpoints/physical.jsonl`.

```bash
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage preflight
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage endpoints
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage run --shard-count 2 --shard-index 0
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage run --shard-count 2 --shard-index 1
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage merge --shard-count 2
./scripts/exp5/run_y1_cache_factorial.py --run-name exp5_YYYYMMDD_HHMM_tag --stage analyze
```

## 7. Provenance of this design

The 2×2×2 factorial, the two primary estimands, the gate-6 degeneracy check and
the gate-7 host-closure reference were proposed by an independent Codex review
(`gpt-5.6-sol`, reasoning effort max) of the completed Exp4 numbers, before this
document was written. The 0.20/0.10 thresholds are carried over unchanged from
Exp4 §0.

## 8. Amendment log

No amendments.
