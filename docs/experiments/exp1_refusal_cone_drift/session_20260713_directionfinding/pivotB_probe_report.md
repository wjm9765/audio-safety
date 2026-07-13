# Pivot B v0 probe — clarification vs guess under GLOBAL acoustic degradation

GLOBAL degradation (no clause localization); v0 base-rate signal. n per cell below.

| safety | level | n | clarify% | blanket-refuse% | harmful-comply% |
|---|---|---:|---:|---:|---:|
| harmful | clean | 20 | 0 | 80 | 20 |
| harmful | mild | 20 | 0 | 80 | 15 |
| harmful | strong | 20 | 0 | 100 | 0 |
| benign | clean | 20 | 0 | 0 | nan |
| benign | mild | 20 | 0 | 0 | nan |
| benign | strong | 20 | 0 | 15 | nan |

## Honest interpretation (Claude)
- **Clarification rate = 0% in every cell**, even under strong degradation and with a system prompt
  that explicitly permits asking to repeat. Qwen2-Audio never hedges/clarifies — it answers or
  blanket-refuses. Strong degradation shifts harmful→100% refuse and benign over-refusal 0→15%
  (a utility cost paid via crude blanket refusal, not calibrated clarification). This is suggestive
  support for Pivot B's premise ("guess/refuse, don't clarify").
- **BUT the v0 signal is INCONCLUSIVE and confounded:** spot-check shows strong-degraded outputs are
  still ON-TOPIC (harmful refusals name the exact harmful act; benign answers are correct) → the
  GLOBAL lowpass+noise degradation did NOT actually destroy intelligibility of the safety-critical
  content. So "0% clarify" may simply mean "the model still understood, so no clarification was
  warranted." This v0 does not create verified uncertainty about the safety-critical span.
- **This directly validates the full design's load-bearing pieces (Codex round-3):** the test MUST
  (a) mask ONLY the safety-critical clause (localization via forced alignment), and (b) verify with
  ASR that the critical content is actually unintelligible while the rest is intact. Global
  degradation cannot separate "safety-uncertainty handling" from "general robustness."
- **Verdict:** direction-check is NEITHER a clean GO nor a clean KILL. The 0%-clarification base rate
  is a genuine, reusable hook; conclusiveness requires the localized + ASR-verified design. Worth the
  user's full build, with clause-localized masking as the non-negotiable core. n=20/cell, 1 model,
  1 TTS, heuristic clarification detection (0% verified by manual spot-check of strong-degraded outputs).
