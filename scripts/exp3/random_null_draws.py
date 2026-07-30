#!/usr/bin/env -S uv run python
"""Post-hoc multi-draw random-direction null for an existing Exp3 span-patch run.

This does NOT modify the pre-registered pipeline or its decision rule. The frozen
`real - wrong_item` contrast in `rules.md` stays exactly as registered; this adds
a host-specific baseline needed to separate two things the single registered
random draw cannot.

Why it is needed. `random_direction` produces a common-mode, refusal-ward push
whose size depends on the host: an already-refusing host is near saturation and
barely moves, while a non-refusing host moves a lot. In a discordant pair one arm
is each type, so that asymmetry aligns with the donor direction and manufactures
a positive donor-aligned score for a control that transports nothing. Averaging
several independent draws per host gives

    mu_rand(i, arm, layer)

so the real effect can be reported net of it:

    T_resid = sign(dR)/2 * [ (d_real_AB - mu_rand_A) - (d_real_BA - mu_rand_B) ]

Reads the frozen cohort/pairs of an existing run and writes a separate artifact;
nothing under the original stage outputs is touched.

Usage:
    ./scripts/exp3/random_null_draws.py \
        --config configs/experiments/exp3_qwen_refusal_mechanism.yaml \
        --run-name exp3_YYYYMMDD_HHMM_tag \
        --draws 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_exp3_config
from audio_safety.evaluation.model_eye import norm_match
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _artifact,
    _capture_audio_spans,
    _checkpoint_mapping,
    _contrast,
    _mechanism_cohort,
    _oriented,
    _prepare_aligned_pair,
    _stable_int,
    readout_token_ids,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

ARTIFACT = Path("random_null/records.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml"),
    )
    parser.add_argument("--run-name", required=True, help="an existing Exp3 run directory")
    parser.add_argument("--draws", type=int, default=3, help="independent random draws per cell")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.draws < 1:
        raise ValueError("--draws must be at least one")
    cfg = load_exp3_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    if not (run_dir / cfg.exp3.artifacts.span_patch_file).is_file():
        raise FileNotFoundError("run has no span-patch records; run the pilot first")

    gate = cfg.exp3.span_patch
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)

    path = _artifact(run_dir, ARTIFACT)
    existing = load_jsonl(path) if path.is_file() else []
    state = {
        (str(r["pair_id"]), str(r["direction"]), int(r["layer"]), int(r["draw_index"])): r
        for r in existing
    }

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("random-null draws require CUDA")
    from audio_safety.models.hooks import SpanStateIntervention
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    from audio_safety.pipelines.exp3_qwen_mechanism import _generate_from_inputs

    completed = 0
    for index, pair in enumerate(cohort):
        needed = [
            (direction, layer, draw)
            for direction in gate.directions
            for layer in gate.layers
            for draw in range(args.draws)
            if (str(pair["pair_id"]), direction, int(layer), draw) not in state
        ]
        if not needed:
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        source_states = _capture_audio_spans(model, source, gate.layers)
        target_states = _capture_audio_spans(model, target, gate.layers)

        for direction, layer, draw in needed:
            host, _donor, host_states, donor_states = _oriented(
                direction, source, target, source_states, target_states
            )
            real_delta = donor_states[layer] - host_states[layer]
            # Draw 0 must reproduce the registered run's draw bit-for-bit, so it
            # uses the pipeline's exact 5-part seed; later draws append the index.
            seed_parts: tuple[object, ...] = (
                gate.seed,
                pair["pair_id"],
                direction,
                layer,
                "random_direction",
            )
            if draw:
                seed_parts = (*seed_parts, draw)
            seed = _stable_int(*seed_parts)
            rng = np.random.default_rng(seed)
            delta = norm_match(rng.standard_normal(real_delta.shape).astype(np.float32), real_delta)
            intervention = SpanStateIntervention(
                model,
                layer_idx=layer,
                positions=host["audio_positions"],
                replacement=host_states[layer] + delta,
            )
            generated = _generate_from_inputs(
                cfg,
                model,
                processor,
                host["inputs"],
                refusal_ids,
                nonrefusal_ids,
                contexts=[intervention],
                max_new_tokens=1,
            )
            key = (str(pair["pair_id"]), direction, int(layer), int(draw))
            state[key] = {
                "schema_version": cfg.exp3.schema_version,
                "stage": "random_direction_null",
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "transition": pair.get("transition"),
                "selection_role": pair.get("selection_role"),
                "direction": direction,
                "layer": int(layer),
                "condition": "random_direction",
                "draw_index": int(draw),
                "host_arm": (
                    pair["source_arm"] if direction == "source_to_target" else pair["target_arm"]
                ),
                "delta_fro": float(np.linalg.norm(delta.reshape(-1))),
                "real_delta_fro": float(np.linalg.norm(real_delta.reshape(-1))),
                "r_tab_margin": generated["r_tab_margin"],
                "applied_count": generated["applied_counts"][0],
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=50)
        print(f"[exp3-null] pair {index + 1}/{len(cohort)} cells={completed}", flush=True)

    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    counts = {row["applied_count"] for row in state.values()}
    if counts != {1}:
        raise RuntimeError(f"hook applied_count must always be one, saw {sorted(counts)}")
    print(f"[exp3-null] done cells={len(state)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
