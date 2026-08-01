# Exp5 — y1 versus prompt cache: run log

> **Append-only.** Add one entry per run. Do not edit prior entries; record a
> correction as a new dated entry. Verdicts must use `design.md` §0.

## Current status

| item | state |
|---|---|
| design and thresholds | **frozen** (2026-07-31) |
| implementation | complete; shares the Exp4 pipeline and CLI |
| CPU tests | **PASS**, full project suite |
| real-weight smoke | **PASS** (`exp5_20260731_2110_smoke`, 64 cells, 4 pairs) |
| non-smoke run | **`exp5_20260731_2145_y1_cache`**, all gates passed |
| result | `D_joint_y1H` negligible, `D_y1_at_HH` material |

No claim is made until every gate in `design.md` §3 passes on a non-smoke run.

---

## exp5-000 — design, implementation and smoke (2026-07-31) — A40

Pre-registered the 2×2×2 factorial, both primary estimands and both new gates
before any GPU cell ran, on the basis of the completed Exp4 result and an
independent Codex review of it. Implemented the `y1` factor as an optional
`__y1_host` condition suffix so Exp4's four condition names, two-letter cell
codes and metrics are untouched; re-running Exp4's analyze stage after the
change reproduced its numbers exactly.

The four-pair smoke passed all gates including the two new ones: `y1`-degenerate
cells were token-identical wherever the injection left `y1` unchanged, and every
fully host-closed cell reproduced an independently recomputed same-replica
physical-host generation. This entry contains no model result.

---

## exp5-001 — aborted run (2026-07-31) — A40

`exp5_20260731_2120_y1_cache` is marked **INVALID**. Its preflight snapshot
pinned commit `bdccad3`; a documentation-only commit moved HEAD before the
routing stage, and the reproducibility gate correctly refused to continue. The
endpoints stage had completed but no cache cell was produced. Superseded by
`exp5_20260731_2145_y1_cache`.

## exp5-002 — `exp5_20260731_2145_y1_cache` (2026-07-31) — A40, device-local endpoints

**Verdict by `design.md` §0.** 1808 cells; primary population the 62 changed-`y1`
directions across 48 pairs; 20,000-draw pair-cluster bootstrap.

| quantity | estimate [95% CI] | frozen decision |
|---|---|---|
| `D_joint_y1H` = `M_IIH − M_HHH` | **+0.010** [+0.000, +0.031] | **negligible at pilot resolution** |
| `D_y1_at_HH` = `M_HHI − M_HHH` | **+0.917** [+0.833, +0.979] | **material conditional contributor** |

Cells: `T_III` +0.969, `T_IHI` +0.969, `T_HII` +0.917, `T_HHI` +0.917 against
`T_IIH` +0.010, `T_IHH` +0.031, `T_HIH` +0.000, `T_HHH` +0.000. Secondary:
`D_audio_y1H` +0.010, `D_tAB_y1H` −0.021 [−0.062, +0.000], `D_y1_at_II` +0.958,
`D_audio_y1I` +0.052, `D_tAB_y1I` +0.000, `D_joint_y1I` +0.052, three-way
interaction +0.021 [−0.042, +0.083].

With the entire injected prompt cache present but `y1` reset to the host token,
donorward transport is ~1%; with a bitwise-host cache and only `y1` swapped it is
92%. The effect runs in both endpoint directions (38/40 nonrefusal→refusal,
19/22 refusal→nonrefusal), so it is not a one-sided refusal-prefix artifact.
54 of the 62 injected `y1` tokens equal the physical donor's own first token, so
generic autoregressive lexical commitment remains a live alternative to any
abstract-refusal-state reading.

Descriptive only, the 76 unchanged-`y1` directions: `T_III` 0.200, `T_IHI` 0.155,
`T_HII` 0.055, `T_HHH` 0.000. Every `T_XXH` equals its `T_XXI` counterpart
exactly there, which is the `y1`-degeneracy gate appearing at aggregate level.

**Integrity.** Cache clone error 0.0; full host-cache closure error 0.0; 4
standard-generate checks (2 per replica) exact on token IDs; **382**
host-reproduction checks all exact; endpoints reproduced Exp4's 220/226 markers
and 73→69 role transitions identically.

**Not an independent sample.** All 904 `y1 = I` cells reproduce Exp4
token-for-token; the `y1 = H` counterparts are identical by degeneracy wherever
`y1` was unchanged. Exp5 re-expresses the same cache evidence and adds the `y1`
arm.

**Not claimed.** That `y1` is necessary, that it carries an abstract refusal
state, that `0.010` is "1% of the causal effect" (it is a residual capacity after
fixing a mediator; the prompt cache still shapes which `y1` is naturally
produced), or that the result generalises beyond the changed-`y1` subgroup of
this device-local cohort. Safety specificity is **not** supported: harmful
`D_y1_at_HH` +0.926 versus benign +0.905.

See `docs/experiments/session_record_20260801_exp4_exp5_exp6.md` for the full
session record.
