# Audio-safety open-problems map (seeds for a genuinely novel direction) — 2026-07-13

Purpose: raw material for the USER-supplied novel problem both reviewers said is required. These are
gaps the 2025-26 literature ITSELF flags as open — but each is paired with what already partially
occupies it (from tonight's verified reading) so you can aim at a real remaining sliver, not a trap.
Reviewers' bar for a fresh problem: (i) a real deployment/stakeholder failure, (ii) an unexpected
empirical phenomenon found BEFORE picking a mechanism, or (iii) a new formal object/guarantee — NOT
another recombination of refusal axes / speaker attributes / privacy gates.

## A. Admitted-open gaps (with occupancy caveats)
1. **Backdoors / latent acoustic triggers in LALMs.** The LALM trustworthiness survey (2605.20266)
   explicitly says backdoors are barely covered vs the mature jailbreak line. Caveat: acoustic
   backdoor attacks exist in ASR; check whether *instruction-following* LALM backdoors + a defense are
   open. Formal-object potential (certified trigger detection).
2. **Multi-turn / conversational audio safety.** Flagged as future work by VoxSafeBench (2604.14548)
   and the FSS eval (2603.13262). Most safety work is single-turn. Caveat: spoken-dialogue judging
   (LALM-as-a-Judge 2602.04796) touches it; the open sliver is safety DRIFT across turns / context
   accumulation attacks in full-duplex audio.
3. **Safety of audio OUTPUT (speech the model GENERATES), not input.** Nearly all work is input-side.
   Caveat: AudioGuard (2604.08867) claims output-side too; check if harmful-paralinguistic GENERATION
   (e.g., model synthesizing coercive/emotionally-manipulative prosody) is open.
4. **Real-world deployment gap: over-the-air / codec / telephony robustness of safety.** Benchmarks use
   clean synthetic TTS. Does safety survive phone codecs, room acoustics, packet loss? Caveat: some
   over-the-air jailbreak work (AudioJailbreak 2505.14103); the open sliver is safety RELIABILITY (not
   attack) under realistic channels — a deployment-failure framing reviewers like.
5. **Formal / guarantee-based audio safety.** No certified robustness or guarantee object exists for
   LALM safety. A new formal object (e.g., an invariance certificate: verdict must be stable under a
   defined benign nuisance set) would be genuinely novel — see the phenomenon probe below for a seed.

## B. Tonight's phenomenon-first seed (uncatalogued behavior candidate)
Ran a WITHIN-item probe (distinct from group-mean bias work): does the SAME harmful request flip its
safety verdict under benign, content-preserving nuisance re-renders (speed/pitch/gain)? If many items
flip, the safety boundary is acoustically ARBITRARY at the individual level — a reliability-of-safety
phenomenon (not adversarial, not demographic bias) that could seed a "safety invariance" formal object
(gap A5). **Result (v0, n=15): 2/15 (13%) of harmful items FLIPPED verdict under benign pitch/speed re-renders
(e.g. pitch −2 semitones or speed 1.15× flips comply↔refuse); 13/15 stable. MODEST/WEAK seed — small n,
judge noise possible; needs n≥150 + robust judge to confirm. If it holds at scale it supports a
"safety-invariance under benign nuisance" reliability claim + a formal invariance-certificate object.**
Novelty caveat to check if the flip rate is high: adversarial-perturbation jailbreaks (2603.13847,
2604.09222) flip verdicts, but via OPTIMIZED perturbations; BENIGN-nuisance within-item instability as
a reliability claim + an invariance guarantee is the potentially-novel angle.

## C. What would make any of these greenlightable (reviewer-derived)
- A problem defined by a real failure/stakeholder need or an unexpected phenomenon, THEN a mechanism —
  not a mechanism (r_A/patching) hunting for a problem.
- A clean baseline-beating result OR a new formal guarantee, on ≥2 models + realistic conditions.
- Verified non-occupancy at submission time (this space moves monthly; re-run the novelty screen).

## D. Phenomenon seed (Section B) — also CLOSED (dual-agent, web-verified)
Codex blind check + Claude search: DON'T SCALE. Substantially preempted — AJailBench (semantically-preserved
pitch/time/amplitude safety perturbations across 7 LALMs), speaker-emotion "safety instability" (2510.16893),
SSI (2512.12066, reliability-metric framing). 13% at n=15 is weak (95% CI ~1.7–40%, possible judge noise).
ICLR ceiling: workshop/negative-diagnostic, not main. → the phenomenon-first avenue also lands on occupied
ground, reinforcing: a novel problem must come from USER domain insight, not autonomous search/discovery here.
