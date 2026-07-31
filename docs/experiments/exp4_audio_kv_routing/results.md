# Exp4 — post-prefill audio/KV routing: run log

> **Append-only.** Add one entry per run. Do not edit prior entries; record a
> correction as a new dated entry. Verdicts must use `design.md` §0.

## Current status

| item | state |
|---|---|
| design and thresholds | **frozen** (2026-08-01) |
| implementation | complete; config-driven and resumable |
| CPU tests | **PASS**, full project suite (439 collected) |
| real-weight smoke | not run |
| non-smoke run | not run |
| result | no empirical Exp4 result yet |

The Exp3 observations motivating this experiment are prior inputs, not Exp4
results. No cache-routing conclusion is claimed until every integrity gate in
`design.md` §5 passes on a non-smoke run.

---

## exp4-000 — design and CPU implementation (2026-08-01) — no GPU

Implemented the frozen 2×2 post-prefill cache surgery, source-Exp3 artifact
hashing, manual fixed-`y1` greedy decoding, pair-cluster bootstrap analysis, and
fail-closed integrity gates. The complete project test suite passes (439 tests),
including optional CPU checks against Transformers `DynamicCache`.

This entry contains no model result. The unresolved real-weight checks are the
exact full-host cache reconstruction and manual-decoder equivalence gates; both
are deliberately executed before an Exp4 analysis can be accepted.
