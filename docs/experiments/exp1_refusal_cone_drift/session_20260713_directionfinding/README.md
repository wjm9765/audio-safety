# 2026-07-13 Direction-finding session — index

Critical, web-grounded (Claude + Codex `gpt-5.6-sol`) decision on whether to write the ICLR paper on
the locked "causal-attribution" direction. Raw data artifacts (audio, `.jsonl`, per-sample outputs) stay
under `/workspace/audio_safety_data/outputs/` (git-ignored); this folder preserves the TEXT record.

## Outcome (one line)
Locked causal-attribution direction + all empirical pivots = **NO-GO** (dead + preempted, dual-agent).
New direction **GREENLIT (conditional)**: **Certified Acoustic Safety Margin for LALMs** — see
[../run5_acoustic_safety_margin_direction_20260713.md](../run5_acoustic_safety_margin_direction_20260713.md).

## Read in this order
1. [DECISION_brief.md](./DECISION_brief.md) — the full session brief: every direction, evidence, verdicts.
2. [open_problems_map.md](./open_problems_map.md) — literature-admitted open gaps (seeds for new problems).

## Experiment reports (text)
- [pilot_acoustic_margin_report.md](./pilot_acoustic_margin_report.md) — the GREENLIT pilot (18/20 certified, 2 brittle).
- [pivotB_killtest_report.md](./pivotB_killtest_report.md) — clause-localized clarify-vs-guess kill-test (KILL).
- [pivotB_probe_report.md](./pivotB_probe_report.md) — global-degradation clarification base-rate (0%).
- [phenom_instability_report.md](./phenom_instability_report.md) — within-item nuisance instability probe (weak).

## Codex deep-discussion record (blind, web-grounded)
- r1 [codex_r1.md](./codex_r1.md) — causal-attribution direction stress-test (reject 3-4/10).
- r2 [codex_r2.md](./codex_r2.md) — pivot to paralinguistic factorization.
- r3 [codex_r3.md](./codex_r3.md) — DROP paralinguistic; propose uncertainty/clarification.
- r4 [codex_r4.md](./codex_r4.md) — final reviewer sign-off (no greenlight).
- r5 [codex_r5.md](./codex_r5.md) — KILL uncertainty safety framing after kill-test.
- r6 [codex_r6.md](./codex_r6.md) — reject speaker-bias; greenlight privacy-gating (later withdrawn).
- r7 [codex_r7.md](./codex_r7.md) — withdraw privacy; declare neighborhood SATURATED, escalate to user.
- [codex_crosscheck.md](./codex_crosscheck.md) — blind cross-check of the causal-trace NULL (NOT-ALIVE).
- [codex_formal_check_out.md](./codex_formal_check_out.md) — formal-object "acoustic safety margin" (5.5/10 conditional).
- [codex_final_greenlight_out.md](./codex_final_greenlight_out.md) — CONDITIONAL GREENLIGHT, commit to this direction.

## Code changed this session
Greedy decoding threaded through the causal-trace generate functions (`src/audio_safety/models/qwen2_audio.py`,
`scripts/causal_trace_flip.py`) — interchange/activation patching requires deterministic decode; the default
sampling config had voided the first run (identity self-patch not reproducing no_patch).
