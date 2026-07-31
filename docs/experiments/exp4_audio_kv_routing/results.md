# Exp4 — post-prefill audio/KV routing: run log

> **Append-only.** Add one entry per run. Do not edit prior entries; record a
> correction as a new dated entry. Verdicts must use `design.md` §0.

## Current status

| item | state |
|---|---|
| design and thresholds | **frozen** (2026-08-01); amended 2026-07-31, see `design.md` §9 |
| implementation | complete; config-driven, resumable, shardable |
| CPU tests | **PASS**, full project suite |
| real-weight smoke | two archived **INVALID** (pre-decoder-fix), then PASS |
| non-smoke run | **`exp4_20260731_2010_audio_kv`**, all gates passed |
| result | `D_audio` ambiguous, `D_tAB` negligible, `D_joint` ambiguous |

The Exp3 observations motivating this experiment are prior inputs, not Exp4
results. The run below is an **amended, device-local** execution: it is not a
confirmatory execution of the original pre-registration on the hardware that
produced its source labels. See `design.md` §9.

---

## exp4-000 — design and CPU implementation (2026-08-01) — no GPU

Implemented the frozen 2×2 post-prefill cache surgery, source-Exp3 artifact
hashing, manual fixed-`y1` greedy decoding, pair-cluster bootstrap analysis, and
fail-closed integrity gates. The complete project test suite passes (439 tests),
including optional CPU checks against Transformers `DynamicCache`.

This entry contains no model result. The unresolved real-weight checks are the
exact full-host cache reconstruction and manual-decoder equivalence gates; both
are deliberately executed before an Exp4 analysis can be accepted.

---

## exp4-001 — two archived invalid smokes (2026-07-31) — A40

`exp4_20260731_1810_smoke2x` and `exp4_20260731_1930_smoke2x` are marked
**INVALID** in their run directories and must never be resumed, merged or
analysed. Both are retained only as the record of two real defects they exposed.

The 1810 smoke failed `design.md` §5 gate 7. Diagnosis: Qwen2-Audio ships
`repetition_penalty: 1.1` in `generation_config.json`, and Transformers 5.12.1
casts next-token logits to float32 before applying logits processors and taking
the argmax. Exp3's entire frozen behaviour corpus and every Exp3 intervention
used `model.generate`, i.e. that decoder; Exp4's `manual_greedy_decode` used an
unprocessed bfloat16 argmax. Teacher-forced against `generate`, the two paths
disagreed by 0.7–2.9 logits at every step and flipped one argmax in 30 tokens
where `generate`'s true top1–top2 gap was 0.148. The prefill KV caches were
bitwise identical and an untransplanted host cache diverged identically, so the
cache surgery was not implicated.

The 1930 smoke, with the decoder corrected, passed gate 7 and failed gate 8.
Diagnosis: the source Exp3 run executed on an RTX A5000 and Exp4 on an A40.
Replaying Exp3's exact behaviour path on the A40 for all 113 pairs reproduced
147/226 texts (65.0%) and 220/226 refusal markers (97.3%). Both smokes were
discarded rather than repaired in place.

## exp4-002 — `exp4_20260731_2010_audio_kv` (2026-07-31) — A40, amended

**Verdict by `design.md` §0, as amended in §9.**

| quantity | estimate [95% CI] | frozen decision |
|---|---|---|
| `D_audio` = `T_II − T_HI` | +0.123 [+0.065, +0.188] | ambiguous |
| `D_tAB` = `T_II − T_IH` | +0.029 [+0.000, +0.065] | **negligible at pilot resolution** |
| `D_joint` = `T_II − T_HH` | +0.167 [+0.101, +0.239] | ambiguous |
| factorial interaction | −0.014 [−0.058, +0.022] | descriptive; no interaction detected |

Cells: `T_II` +0.580, `T_IH` +0.551, `T_HI` +0.457, `T_HH` +0.413. 904 cells,
69 current-device discordant pairs, 2 directions, 20,000-draw pair-cluster
bootstrap. `T_II` = 0.580 closely reproduces the inherited Exp3 relay-closure
value of ≈0.575.

**`T_HH` is `y1` lock-in, not a surviving cache route.** It splits exactly on
whether the injection moved the first generated token: 0/76 donorward across
directions where it did not (an identity consequence of gate 8a, not an
estimated null), and +0.917 [+0.833, +0.979] across 48 pairs where it did.
Because gate 6 proves the mixed cache equals the physical host cache at
tolerance 0, the only injected information in that cell is the identity of `y1`.

**Integrity.** All hook counts correct; cache clone error 0.0; full host-cache
closure error 0.0; 4 standard-generate equivalence checks (2 per model replica),
all exact on token IDs; 156 gate-8a host-reproduction checks, all exact.
Stable-control pairs changed the host marker in 0.011–0.034 of cells,
so the intervention is not indiscriminate.

**Robustness.** The endpoint amendment did not manufacture the result:
`D_audio`/`D_tAB`/`D_joint` are +0.123/+0.029/+0.167 under the amended
current-device 69-pair cohort, +0.116/+0.021/+0.151 under the original frozen
73-pair labels, and +0.125/+0.022/+0.162 on the 68 pairs common to both. A
serial re-run of six pairs was token-identical to the concurrent two-shard run
on all 48 comparable cells, so co-resident replicas did not perturb the numbers.

**Threats recorded.** 14.7% of primary cells hit the 96-token limit. The
endpoint is a literal marker: many cells match neither physical endpoint
textually, and marker movement is not harmful compliance or semantic
equivalence. `HI` and `IH` are hybrid caches and may be off-manifold. The
primary 69-pair target was determined by current-device physical endpoints
before any cache cell ran — deterministic and pre-intervention, but still
endpoint-dependent, so "no outcome-dependent selection" overstates it. The
bootstrap covers within-cohort variation only, not hardware, model,
transformation or corpus uncertainty.

**Not claimed.** That audio K/V is a material, dominant, necessary or unique
refusal route; that `t_AB` has no effect; that the joint prompt cache is
irrelevant; that `y1` explains any percentage of anything, or is necessary, or
carries an abstract refusal state; that the near-zero binary interaction shows
independence or additivity; that any tokens, layers, heads or the `A→t_AB` edge
are identified; that the result generalises across GPUs, models, acoustic
transformations or non-discordant inputs.

Interpretation was cross-checked blind against an independent Codex review
(`gpt-5.6-sol`, effort max) that pre-registered its reading of every possible
outcome pattern before these numbers existed. Exp5 follows directly.
