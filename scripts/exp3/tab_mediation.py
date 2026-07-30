#!/usr/bin/env -S uv run python
"""Post-hoc t_AB mediation test for an existing Exp3 run.

Does NOT modify the pre-registered pipeline or its decision rule. It answers the
one question the pilot left open: the L10 audio-span effect reaches the output
somehow, but is that route *through* the `t_AB` readout position?

M2/M3 established that a donor audio span injected at L10 moves the refusal
marker, and that resetting the span at L12 removes ~97% of it. Stage 5 showed
`t_AB` carries almost nothing at L10 (+0.027) and a lot by L18 (+0.503). The
tempting story -- "the signal hands off to t_AB by L18" -- is not yet supported,
because at L18 whole-span patching is behaviourally null and `R_tAB` disagrees in
sign with the eventual marker on 38.4% of discordant pairs.

A pure rescue test is under-powered for this: injecting the donor `t_AB` state
alone already moves the output, so "it came back" cannot distinguish a genuine
hand-off from simply re-running the Stage 5 L18 effect. The decisive test is
therefore **blocking**, with rescue reported alongside:

    inject_only          L10 span <- donor
    inject_reset         L10 span <- donor, L12 span <- host      (reproduces M3)
    inject_clamp_l14     L10 span <- donor, L14 t_AB <- host
    inject_clamp_l18     L10 span <- donor, L18 t_AB <- host      <-- PRIMARY
    clamp_l18_only       L18 t_AB <- host                          (must be a no-op)
    reset_rescue         L10 span <- donor, L12 span <- host, L18 t_AB <- donor
    rescue_only          L18 t_AB <- donor                         (Stage 5 reference)
    identity             L10 span <- host                          (no-op)

If `inject_clamp_l18` collapses toward baseline while `inject_only` does not, the
L10 effect is mediated by the readout position. If it survives, the effect
reaches the output by a route that does not pass through `t_AB` at L18 and the
hand-off narrative should be dropped rather than rescued.

Endpoint is the **full generated marker**, not `R_tAB`, precisely because the two
diverge. `clamp_l18_only` clamps the host's own state into a host run, so it must
reproduce the baseline exactly -- a free hook sanity check.

Usage:
    ./scripts/exp3/tab_mediation.py \
        --config configs/experiments/exp3_qwen_refusal_mechanism.yaml \
        --run-name exp3_YYYYMMDD_HHMM_tag
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_safety.config import load_exp3_config
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _artifact,
    _behavior_index,
    _capture_audio_spans,
    _capture_t_ab_states,
    _checkpoint_mapping,
    _contrast,
    _generate_from_inputs,
    _mechanism_cohort,
    _oriented,
    _prepare_aligned_pair,
    _validated_physical_baselines,
    readout_token_ids,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

ARTIFACT = Path("tab_mediation/records.jsonl")
INJECT_LAYER = 10
RESET_LAYER = 12
CLAMP_LAYERS = (14, 18)
CONDITIONS = (
    "identity",
    "inject_only",
    "inject_reset",
    "inject_clamp_l14",
    "inject_clamp_l18",
    "clamp_l18_only",
    "reset_rescue",
    "rescue_only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml"),
    )
    parser.add_argument("--run-name", required=True, help="an existing Exp3 run directory")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_exp3_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    if not (run_dir / cfg.exp3.artifacts.span_patch_file).is_file():
        raise FileNotFoundError("run has no span-patch records; run the pilot first")

    gate = cfg.exp3.span_patch
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    frozen_behavior = _behavior_index(cfg, run_dir)

    path = _artifact(run_dir, ARTIFACT)
    existing = load_jsonl(path) if path.is_file() else []
    state = {(str(r["pair_id"]), str(r["direction"]), str(r["condition"])): r for r in existing}

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("t_AB mediation requires CUDA")
    from audio_safety.models.hooks import ResidualStreamIntervention, SpanStateIntervention
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)

    span_layers = [INJECT_LAYER, RESET_LAYER]
    completed = 0
    for index, pair in enumerate(cohort):
        pending = [
            (direction, condition)
            for direction in gate.directions
            for condition in CONDITIONS
            if (str(pair["pair_id"]), direction, condition) not in state
        ]
        if not pending:
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        src_span = _capture_audio_spans(model, source, span_layers)
        tgt_span = _capture_audio_spans(model, target, span_layers)
        src_tab = _capture_t_ab_states(model, source, list(CLAMP_LAYERS))
        tgt_tab = _capture_t_ab_states(model, target, list(CLAMP_LAYERS))
        source_baseline, target_baseline = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            frozen_behavior=frozen_behavior,
        )

        for direction, condition in pending:
            host, _donor, host_span, donor_span = _oriented(
                direction, source, target, src_span, tgt_span
            )
            _h2, _d2, host_tab, donor_tab = _oriented(direction, source, target, src_tab, tgt_tab)
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )

            positions = host["audio_positions"]
            t_ab = int(host["t_ab"])

            def span_hook(layer: int, states, _pos=positions) -> object:
                return SpanStateIntervention(
                    model,
                    layer_idx=layer,
                    positions=_pos,
                    replacement=states[layer],
                )

            def tab_hook(layer: int, states, _tab=t_ab) -> object:
                return ResidualStreamIntervention(
                    model,
                    layer_idx=layer,
                    token_index=_tab,
                    mode="patch_state",
                    replacement_state=states[layer],
                )

            contexts: list[object] = []
            if condition == "identity":
                contexts.append(span_hook(INJECT_LAYER, host_span))
            elif condition == "inject_only":
                contexts.append(span_hook(INJECT_LAYER, donor_span))
            elif condition == "inject_reset":
                contexts.append(span_hook(INJECT_LAYER, donor_span))
                contexts.append(span_hook(RESET_LAYER, host_span))
            elif condition == "inject_clamp_l14":
                contexts.append(span_hook(INJECT_LAYER, donor_span))
                contexts.append(tab_hook(14, host_tab))
            elif condition == "inject_clamp_l18":
                contexts.append(span_hook(INJECT_LAYER, donor_span))
                contexts.append(tab_hook(18, host_tab))
            elif condition == "clamp_l18_only":
                contexts.append(tab_hook(18, host_tab))
            elif condition == "reset_rescue":
                contexts.append(span_hook(INJECT_LAYER, donor_span))
                contexts.append(span_hook(RESET_LAYER, host_span))
                contexts.append(tab_hook(18, donor_tab))
            elif condition == "rescue_only":
                contexts.append(tab_hook(18, donor_tab))
            else:
                raise ValueError(f"unknown condition {condition!r}")

            generated = _generate_from_inputs(
                cfg,
                model,
                processor,
                host["inputs"],
                refusal_ids,
                nonrefusal_ids,
                contexts=contexts,
            )
            no_op = condition in {"identity", "clamp_l18_only"}
            state[(str(pair["pair_id"]), direction, condition)] = {
                "schema_version": cfg.exp3.schema_version,
                "stage": "tab_mediation",
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "transition": pair.get("transition"),
                "selection_role": pair.get("selection_role"),
                "direction": direction,
                "condition": condition,
                "inject_layer": INJECT_LAYER,
                "reset_layer": RESET_LAYER,
                "baseline_response": baseline["response"],
                "baseline_explicit_refusal": baseline["explicit_refusal"],
                "baseline_r_tab_margin": baseline["r_tab_margin"],
                "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                "donor_r_tab_margin": donor_baseline["r_tab_margin"],
                "response": generated["response"],
                "explicit_refusal": generated["explicit_refusal"],
                "r_tab_margin": generated["r_tab_margin"],
                "applied_counts": generated["applied_counts"],
                # Both no-op conditions replace a state with itself, so any
                # deviation here is a hook bug rather than a result.
                "no_op_response_exact": (
                    generated["response"] == baseline["response"] if no_op else None
                ),
                "no_op_margin_abs_error": (
                    abs(generated["r_tab_margin"] - baseline["r_tab_margin"]) if no_op else None
                ),
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=25)
        print(f"[exp3-tab] pair {index + 1}/{len(cohort)} cells={completed}", flush=True)

    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    bad = [row for row in state.values() if any(c != 1 for c in row["applied_counts"])]
    if bad:
        raise RuntimeError(f"{len(bad)} cells did not apply every hook exactly once")
    noop_bad = [
        row
        for row in state.values()
        if row["no_op_response_exact"] is False or (row["no_op_margin_abs_error"] or 0) > 0
    ]
    if noop_bad:
        raise RuntimeError(f"{len(noop_bad)} no-op cells did not reproduce the baseline")
    print(f"[exp3-tab] done cells={len(state)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
