#!/usr/bin/env -S uv run python
"""Adjudicate the audio>text mechanism from Run 4 Stage B activations.

Reads the captured activations and emits the mechanism call
(CONVERSION / PERCEPTION / DRIFT / READOUT / MIXED / UNRESOLVED) with quantified
evidence, plus the block-level writer profile. Direction-finding only.

Example:
    ./scripts/analyze_conversion.py \
        --config configs/experiments/run4_conversion_gap.yaml \
        --run-name run4_20260712_1200_probe
"""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.conversion_probe import (
    adjudicate_conversion,
    block_writer_gap,
    cross_fit_dim,
    readout_auroc,
)
from audio_safety.pipelines.rdo_gate import load_axis
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--axis-artifact", type=Path, default=None)
    return parser.parse_args()


def _select_primary_ch_layer(ch_stack, meta, c_h_layers, k, seed) -> int:
    """Pick the content-position layer whose cross-fit r_H best separates harmful
    vs benign in the AUDIO arm (the hardest, most decisive readout)."""
    item_ids = [str(m["item_id"]) for m in meta]
    harmful = np.array([str(m["safety_label"]) == "harmful" for m in meta])
    audio = np.array([str(m["modality"]) == "audio" for m in meta])
    best_layer, best_auroc = 0, -1.0
    aurocs = {}
    for j, layer in enumerate(c_h_layers):
        c_h = cross_fit_dim(ch_stack[:, j, :], harmful, item_ids, k=k, seed=seed)
        auroc = readout_auroc(c_h[audio], harmful[audio].astype(int))
        aurocs[int(layer)] = auroc
        if not np.isnan(auroc) and auroc > best_auroc:
            best_layer, best_auroc = j, auroc
    return best_layer, aurocs


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    probe = cfg.conversion_probe
    if probe is None:
        raise SystemExit("this config has no `conversion_probe` block (Run 4 Stage B)")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    axis_path = args.axis_artifact or probe.frozen_axis_artifact
    if axis_path is None:
        raise SystemExit("provide the frozen r_A axis via --axis-artifact or config")
    r_a, _ = load_axis(Path(axis_path))

    data = np.load(run_dir / probe.activations_file, allow_pickle=True)
    meta = load_jsonl(run_dir / probe.metadata_file)
    ch_stack = data["ch_stack"]
    cr_hidden = data["cr_hidden"]
    cr_by_layer = data["cr_by_layer"]
    c_h_layers = list(data["c_h_layers"])

    primary_j, ch_aurocs = _select_primary_ch_layer(
        ch_stack, meta, c_h_layers, probe.n_cross_fit_folds, cfg.seed
    )
    report = adjudicate_conversion(
        cr_hidden,
        ch_stack[:, primary_j, :],
        meta,
        r_a,
        n_cross_fit_folds=probe.n_cross_fit_folds,
        n_random_directions=probe.n_random_directions,
        harmfulness_preserved_max_sd=probe.harmfulness_preserved_max_sd,
        refusal_underdriven_min_sd=probe.refusal_underdriven_min_sd,
        specificity_min_ratio=probe.specificity_min_ratio,
        readout_min_auroc=probe.readout_min_auroc,
        seed=cfg.seed,
    )
    report["primary_c_h_layer"] = int(c_h_layers[primary_j])
    report["c_h_layer_audio_native_auroc"] = ch_aurocs

    # block-level writer profile on harmful items
    modality = np.array([str(m["modality"]) for m in meta])
    harmful = np.array([str(m["safety_label"]) == "harmful" for m in meta])
    report["block_writer"] = block_writer_gap(
        cr_by_layer[(modality == "text") & harmful],
        cr_by_layer[(modality == "audio") & harmful],
    )

    out = run_dir / probe.report_file
    save_json(report, out)
    print(f"[probe] mechanism call: {report['status']}")
    for reason in report["reasons"]:
        print(f"[probe]   {reason}")
    print(f"[probe] report -> {out}")


if __name__ == "__main__":
    main()
