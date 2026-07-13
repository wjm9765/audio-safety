1. Rule checks:

- (1) Identity invariance: **PASS, both judges.** Identity equals no_patch (0.105), with 0/19 mismatches.
- (2) Benign-adjusted rescue \(C>0\) with CI: **FAIL / not demonstrated.** No benign-condition rates, adjusted contrast, or CI are provided.
- (3) same_item beats both controls: **FAIL, both judges.** Gemini: same_item 0.579 < wrong_item 0.632. Haiku: 0.316 = 0.316. Both beat random displacement, but neither beats wrong_item.
- (4) Reverse lowers refusal from clean baseline: **FAIL / not demonstrated.** Reverse remains highly refusal-like (0.895/0.842); no numerical clean baseline is supplied. Under the stated characterization that reverse raises refusal, it directly fails.

Overall: **NOT-ALIVE.** The preregistered rule requires all four conditions, and at least condition (3) decisively fails for both judges.

Mechanistically, the patch appears to transfer a **generic refusal-promoting or clean-run state**, not item-specific causal content counteracting the jailbreak. A wrong-item donor working as well as—or better than—the same-item donor rules against item specificity. Its advantage over norm-matched noise shows the transferred structure is meaningful, but not the hypothesized item-specific causal signal. The reverse behavior further undermines the proposed bidirectional causal interpretation.

Gemini estimates a larger rescue than Haiku, indicating judge-sensitive effect magnitude. It does **not** change the verdict: both judges fail the critical same-item-over-wrong-item requirement.