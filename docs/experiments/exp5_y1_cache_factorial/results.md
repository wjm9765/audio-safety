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
| non-smoke run | in progress |
| result | none yet |

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
