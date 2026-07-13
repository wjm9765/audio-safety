# Run 5 direction (GREENLIT) — Certified Acoustic Safety Margin for LALMs (2026-07-13)

> **Status: dual-agent CONDITIONAL GREENLIGHT to start the ICLR build.** First direction of the session
> to receive a start-build approval (all prior: NO-GO — see `outputs/run4_causal_attribution_DECISION.md`,
> [context.md](./context.md)). Reached via the reviewer-endorsed "new formal object" path after the
> Qwen2-Audio/r_A/CosyVoice2 empirical neighborhood was exhaustively shown saturated. `design.md` §0/§1
> and prior run §0 unchanged; this is a NEW direction the user will set up as the full experiment.

## The object
For a harmful spoken request x that the model refuses, define the **acoustic safety margin**: the radius
r_safe(x), in a **content-preserving** acoustic-perturbation metric, of the largest region around x within
which the model's safety verdict stays "refuse". An input is ε-safe-robust iff no content-preserving
perturbation within ε flips refuse→comply. Contribution = the formal object + a certification procedure +
the empirical margin distribution + (optionally) a margin-improving defense.

## Why greenlit (evidence)
Cheap pilot (`run4_acoustic_margin/`; 20 harmful items × ~40 perturbations of pitch U[-3,3]st /
time U[.85,1.2] / gain U[.4,1]; greedy; 2-judge; Clopper-Pearson): **18/20 certified-refusal-robust
(CP-lower>0.5), 2/20 BRITTLE** (a normally-refused harmful request complied with ~70% of the time under
benign perturbation). The certificate discriminates robust from brittle harmful items — deployment-relevant.
Least-preempted direction found (no exact LALM acoustic-refusal certificate exists as of 2026-07-13).

## Make-or-break deliverable (both reviewers) — this is what decides ICLR
A **sound, non-vacuous DETERMINISTIC certificate over a perceptually-calibrated JND transform box** —
prove non-existence of verdict flips within a JND-metered, low-dimensional (3-6 param) content-preserving
transform region via justified bounds (grid + Lipschitz/interpolation over the low-dim manifold), NOT a
dense empirical grid or purely probabilistic RS radii. Must state precisely what is certified relative to
the judge oracle (include a judge error/disagreement bound). Deliver this → **7/10 (weak accept→accept)**;
fall back to sampling-based CP bounds → **5/10 or below** ("generic randomized smoothing applied to audio").

## Precise claim boundary (do not overstate)
CP-lower>0.5 certifies MAJORITY-refusal-probability under the perturbation DISTRIBUTION; it does NOT
certify robustness to EVERY allowed transform (18/20 pilot items still had ≥1 flip). The deterministic
JND-box certificate is what upgrades "probably refuses on average" to "provably no flip within this
perceptual box." Keep these separate in all claims.

## Differentiation vs closest work (verified this session)
- RS-for-VLMs (2509.16088): certifies judge-mapped generative safety, but generic high-dim l2 — ours is
  a perceptual JND, low-dim, deterministic non-existence certificate for AUDIO transforms.
- SSI (2512.12066): seed/temperature reliability metric, no acoustic geometry, no certificate.
- AJailBench (2505.15406): supplies acoustic transforms + ASR, but NO per-input certified radius.
- ASR/speaker-rec audio certificates exist but are not safety-verdict certificates.

## Full-build checklist (user's setup)
- ≥2 models (Qwen2-Audio + a structurally different LALM), ≥100 harmful items + hard-benign controls,
  multiple judges + a validated judge error bound, the deterministic JND-box certificate + its soundness
  argument, the brittle-tail analysis (what makes an item's refusal acoustically fragile), and a
  margin-improving defense (interface-level smoothing/aug) evaluated on the brittle tail.
- Re-run the novelty screen at submission time (space moves monthly).

## Pivot de-risking probe (2026-07-13 PM, after the internal-direction NO-GO)
`outputs/cand4_jnd_probe/jnd_margin_probe.json` — quick feasibility probe of Codex's make-or-break risk
("a certificate over a DISCONTINUOUS black-box judge verdict may be infeasible; certify a model-side
CONTINUOUS refusal margin instead"). 5 harmful items × a 5×3 pitch/gain JND grid; endpoint = the
first-token refusal-logit margin (from the internal-direction work). Result: the continuous margin is
**bounded-variation over the box** (max adjacent-cell jump ~1.2–4.0, NO discontinuities), **sign-stable for
3/5 items** (certifiable-robust; e.g. margin ∈ [+1.87,+5.15] all-refuse or [−4.65,−1.84] all-comply) and
**cleanly sign-flipping for 2/5** (located brittle counterexamples, margin crosses 0 inside the box). This
supports the continuous-margin certificate route and suggests the deterministic JND-box certificate is
**feasible via a valid modulus-of-continuity bound + adaptive refinement near the margin=0 boundary** — the
coarse 1.5st grid is too sparse to exclude a narrow flip by sampling alone (confirming a sound bound, not a
dense grid, is required). **First full-build experiment (Codex r3): certify one robust + one brittle item on
a 2-D JND box via adaptive interval branch-and-bound with a mathematically valid bound; do not scale until
that works.**

## Caveats (Claude)
- The pilot is sampling-based (= generic RS) — the paper's novelty rests ENTIRELY on the deterministic
  JND certificate landing. If that proves infeasible on a black-box judge-defined verdict, the direction
  degrades to a 5/10 measurement paper. This is the single biggest risk; prototype the certificate early.
- n=20/1-model/2-judge pilot; the 2 brittle items should be re-verified with more judges (some judge
  noise possible, though 70% comply is well above noise).
