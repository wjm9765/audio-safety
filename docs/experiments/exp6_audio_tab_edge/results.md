# Exp6 — prefill audio → t_AB attention edge: run log

> **Append-only.** Verdicts must use `design.md` §0.

## Current status

| item | state |
|---|---|
| design and thresholds | **frozen** (2026-08-01), one pre-run amendment in §4 |
| implementation | complete; prefill-only, shardable |
| identity gate | **PASS**, bitwise, 226/226 |
| non-smoke run | **`exp6_20260801_0345_edge`**, all gates passed |
| result | edge is a **material, specific** contributor to `y1` selection |

---

## exp6-000 — invalid first official run (2026-08-01) — A40

`exp6_20260801_0310_edge` is marked **INVALID**. Both shards wrote to a single
`edge/records.jsonl`, and because each rewrites the whole file from its own
state, shard 0's 456 rows were overwritten by shard 1's 448. The script now
writes per-shard checkpoints. Its half-cohort numbers (specificity +0.750
[+0.604, +0.875]) are consistent with the full run below but must not be quoted.

Two earlier smokes exposed setup defects: `wrong_edge` was silently absent
because 85 of the 113 Exp3 wrong-item partners lie outside the mechanism cohort
and were looked up in the wrong file, and the partner spans are not natively
length-matched (see §4 amendment). The run now fails closed if a wrong-item edge
cannot be built.

## exp6-001 — `exp6_20260801_0345_edge` (2026-08-01) — A40

**Verdict by `design.md` §0.** 904 rows; primary population the 62 changed-`y1`
directions across 48 pairs; 20,000-draw pair-cluster bootstrap.

| quantity | estimate [95% CI] | frozen decision |
|---|---|---|
| `R_host_edge` | **+1.000** [+1.000, +1.000] | **material** |
| `R_wrong_edge` | +0.333 [+0.219, +0.448] | — |
| `R_host_edge − R_wrong_edge` | **+0.667** [+0.552, +0.781] | **specific** |

Both §0 criteria are met, so the prefill audio → `t_AB` edge is recorded as a
material contributor to first-token selection under this design.

By role: harmful +0.630 [+0.481, +0.778] (27 pairs), benign +0.714 [+0.548,
+0.881] (21 pairs) — no safety specificity, consistent with Exp3–Exp5.

**Integrity.** `identity` reproduced `no_patch` bitwise on 226/226 rows; every
edge hook fired exactly once per layer; `R_no_patch` = 0.000 by construction.

**Interpretive caveat.** `R_host_edge` = 1.000 with zero variance is close to
structurally forced: Exp4's design already clamps the 15 relay positions to host
values at every layer L11–L31, and `t_AB` cannot causally reach the injection
through pre-audio positions, so patching the audio edge restores essentially all
of `t_AB`'s layer-11+ attention inputs to host values. The informative quantity
is the **0.667 contrast against a wrong-item edge**, which shows the reversion
depends on the edge carrying host-specific content rather than merely being
perturbed.

**Scope.** This localises an edge in a conduit that Exp3's own transcription
control shows is not safety-specific: the same L10 patch under a
*"Transcribe the spoken audio verbatim"* instruction transports the donor's
transcript with pair-clustered donorward 0.684 [0.553, 0.816], overlapping the
safety task's 0.781 [0.712, 0.842]. A positive Exp6 result localises a route for
a response-mode switch, not for a safety judgement.

See `docs/experiments/session_record_20260801_exp4_exp5_exp6.md`.
