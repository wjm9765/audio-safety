# Exp6 — the prefill audio → t_AB attention edge: pre-registered design

> **Status:** pre-registered and frozen before the first GPU cell, 2026-08-01.
> Thresholds and the primary estimand must not be edited after the run begins.
>
> **Scope:** Exp5 showed that once the first generated token `y1` is reset to the
> host token, the entire injected prompt cache retains +0.010 [0.000, 0.031]
> donorward capacity, while swapping `y1` alone under a bitwise-host cache gives
> +0.917. The decisive event is therefore the selection of `y1` during prefill.
> Exp6 asks how the injected audio span reaches that selection.

## 0. Frozen decision table

95% pair-cluster bootstrap intervals (20,000 draws) over the **changed-`y1`
directions** of the current-device discordant cohort, averaging available
directions within pair.

Primary endpoint: **exact host-`y1` reversion**, i.e. the indicator that the
patched injected prefill selects the same first token the physical host prefill
selects.

| quantity | definition | material | negligible | otherwise |
|---|---|---|---|---|
| `R_host_edge` | reversion under the host-audio edge patch | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |
| `R_host_edge − R_wrong_edge` | specificity over a length-matched wrong-item edge patch | CI low > 0 | CI high < 0.10 | ambiguous |

**Both** must be material for the edge to be called a material contributor. A
material `R_host_edge` with a non-material specificity contrast is reported as a
non-specific perturbation effect, not as a route.

## 1. Intervention

The injected prefill is Exp4's: donor audio-span replacement after decoder block
L10, with all 15 post-audio relay positions clamped to their physical-host
residual states after every block L11–L31.

On top of that, at each layer L11–L31, the attention module is run twice: once
unmodified, and once with the attention-input states at the audio positions
replaced. Only the `t_AB` row of the second output is kept. Because the query at
`t_AB` comes from its own unmodified hidden state, that row is exactly "`t_AB`
attends to replacement audio K/V, and to unmodified K/V everywhere else". Every
other position, the residual stream, the MLPs, and the KV cache written for
decoding are untouched. The second call is made with `past_key_values=None` so
it cannot touch the cache. This is prefill only; the run stops after `y1`.

| condition | audio states seen by `t_AB` |
|---|---|
| `no_patch` | injected (the Exp4 prefill) |
| `identity` | the injected run's own captured states |
| `host_edge` | the physical-host prefill's states |
| `wrong_edge` | a length-matched other cohort item's states |

The wrong-item assignment is inherited verbatim from the Exp3 span-patch
`wrong_item_id`, which is already exact-length matched.

## 2. Integrity gates (fail closed)

1. `identity` must reproduce `no_patch` bitwise: identical final logits, identical
   KV cache, identical `y1`. Validated before pre-registration on three pairs at
   max|Δlogit| = 0.0 and max|ΔK| = 0.0.
2. Every edge hook fires exactly once per layer per forward.
3. The injected prefill must reproduce Exp4's recorded `fixed_y1` for the same
   pair and direction, and the physical host prefill must reproduce Exp4's
   `host_y1`.
4. Physical endpoints are the current-device measurements from Exp4/Exp5.
5. Exactly one row per pair × direction × condition.

## 3. What this cannot establish

- It does not show the edge is necessary in the natural, uninjected forward pass.
- It does not identify heads, layers or audio tokens; the whole span and the
  whole L11–L31 range are patched together.
- Reverting `y1` is not the same as reverting the refusal marker; the final
  marker is reported as a secondary endpoint only.
- Exp3 already records that **safety specificity is not supported** on this
  cohort (benign ≥ harmful, interaction includes zero) and that `real` span
  patching is an excess over substantial non-inert shams. Exp6 inherits both
  limitations: a positive result localises a route for a response-mode switch,
  not for a safety judgement.
- One model, one acoustic transformation, one selected cohort, literal marker.

## 4. Amendment log

No amendments.

### 2026-08-01 — wrong-item length matching (pre-run)

Written before any non-smoke cell. §1 stated that the Exp3 `wrong_item_id`
assignment is "already exact-length matched". That is wrong: the `span_len`
recorded on an Exp3 wrong-item row is the **host** span length, not the
partner's own, and some partners are natively shorter than the host span (e.g.
`figstep_safebench_0384` needs 199 positions, its partner supplies fewer).

The partner assignment is still inherited verbatim. Where the partner's own
audio span is shorter than the host's, its captured attention-input states are
**tiled** to the required length; where it is longer, they are truncated. Each
record stores `wrong_item_native_span` so the tiled cases are auditable and can
be dropped in a sensitivity analysis. No threshold, estimand or population
changes.
