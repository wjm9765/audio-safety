#!/usr/bin/env -S uv run python
"""How long must the audio positions keep carrying the perturbation?

Post-hoc sweep over an existing Exp3 run. Does not modify the pre-registered
pipeline. M3 measured only two reset layers (L12 -> 3.3% survives, L14 -> 16.5%),
which is a curve with two points. This fills it in.

Inject the donor audio span at L10, then reset the audio span to the host
trajectory at layer k. The survival curve

    s(k) = donorward(reset at k) / donorward(inject_only)

and the layer k* where s first exceeds 0.5 answer, in one number a non-specialist
can read: *the audio tokens have to keep carrying the difference until about k*;
restoring them after that no longer restores host behaviour.*

Crucially the reset targets **audio** positions, never the answer-boundary
position whose final residual is unembedded, so no outcome here can be produced
algebraically by forcing the first-token logits -- the trap that contaminates a
t_AB clamp through the final block.

The endpoint is the generated refusal marker, and every cell is screened for
degenerate output, so that "no refusal marker in garbage" is scored as
intervention failure rather than as a behavioural change.

Usage:
    ./scripts/exp3/span_persistence_sweep.py --run-name exp3_YYYYMMDD_HHMM_tag
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from audio_safety.config import load_exp3_config
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _artifact,
    _behavior_index,
    _capture_audio_spans,
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

ARTIFACT = Path("span_persistence/records.jsonl")
INJECT_LAYER = 10
RESET_LAYERS = (11, 12, 13, 14, 16, 18, 20, 22)
NON_LATIN = re.compile(r"[一-鿿぀-ヿ가-힯]")


def degenerate(text: str) -> str | None:
    """Pre-specified quality screen; a flagged cell is an intervention failure."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(NON_LATIN.findall(stripped)) > 10:
        return "non_latin"
    tokens = stripped.split()
    for size in (8, 12):
        if len(tokens) >= 2 * size:
            for start in range(len(tokens) - 2 * size + 1):
                head = tokens[start : start + size]
                if head == tokens[start + size : start + 2 * size]:
                    return "verbatim_repeat"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml"),
    )
    parser.add_argument("--run-name", required=True)
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
        raise RuntimeError("span persistence sweep requires CUDA")
    from audio_safety.models.hooks import SpanStateIntervention
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)

    # inject_only does not depend on the reset layer, so it is run once per
    # (pair, direction) rather than once per layer.
    conditions = ["identity", "inject_only"] + [f"reset_l{k}" for k in RESET_LAYERS]
    layers = sorted({INJECT_LAYER, *RESET_LAYERS})
    completed = 0
    for index, pair in enumerate(cohort):
        pending = [
            (direction, condition)
            for direction in gate.directions
            for condition in conditions
            if (str(pair["pair_id"]), direction, condition) not in state
        ]
        if not pending:
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        src_span = _capture_audio_spans(model, source, layers)
        tgt_span = _capture_audio_spans(model, target, layers)
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
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )
            positions = host["audio_positions"]

            def hook(layer: int, states, _pos=positions) -> object:
                return SpanStateIntervention(
                    model, layer_idx=layer, positions=_pos, replacement=states[layer]
                )

            reset_layer = None
            if condition == "identity":
                contexts = [hook(INJECT_LAYER, host_span)]
            elif condition == "inject_only":
                contexts = [hook(INJECT_LAYER, donor_span)]
            else:
                reset_layer = int(condition.removeprefix("reset_l"))
                contexts = [hook(INJECT_LAYER, donor_span), hook(reset_layer, host_span)]

            generated = _generate_from_inputs(
                cfg,
                model,
                processor,
                host["inputs"],
                refusal_ids,
                nonrefusal_ids,
                contexts=contexts,
            )
            state[(str(pair["pair_id"]), direction, condition)] = {
                "schema_version": cfg.exp3.schema_version,
                "stage": "span_persistence",
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "transition": pair.get("transition"),
                "selection_role": pair.get("selection_role"),
                "direction": direction,
                "condition": condition,
                "inject_layer": INJECT_LAYER,
                "reset_layer": reset_layer,
                "baseline_response": baseline["response"],
                "baseline_explicit_refusal": baseline["explicit_refusal"],
                "baseline_r_tab_margin": baseline["r_tab_margin"],
                "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                "donor_r_tab_margin": donor_baseline["r_tab_margin"],
                "response": generated["response"],
                "explicit_refusal": generated["explicit_refusal"],
                "r_tab_margin": generated["r_tab_margin"],
                "applied_counts": generated["applied_counts"],
                "quality_flag": degenerate(generated["response"]),
                "identity_exact": (
                    generated["response"] == baseline["response"]
                    if condition == "identity"
                    else None
                ),
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=25)
        print(f"[exp3-persist] pair {index + 1}/{len(cohort)} cells={completed}", flush=True)

    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    bad = [r for r in state.values() if any(c != 1 for c in r["applied_counts"])]
    if bad:
        raise RuntimeError(f"{len(bad)} cells did not apply every hook exactly once")
    ident_bad = [r for r in state.values() if r["identity_exact"] is False]
    if ident_bad:
        raise RuntimeError(f"{len(ident_bad)} identity cells did not reproduce the baseline")
    flagged = sum(1 for r in state.values() if r["quality_flag"])
    print(
        f"[exp3-persist] done cells={len(state)} quality-flagged={flagged} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
