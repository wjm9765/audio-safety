#!/usr/bin/env -S uv run python
"""Train and validate Audio-RDO axes over configured layer/position candidates."""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.models.qwen2_audio import load_qwen2_audio
from audio_safety.pipelines.rdo_gate import (
    Site,
    rows_for_split,
    save_axis,
    save_selected_site,
    split_ids,
    train_and_validate_site,
)
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug limit per site")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(
        cfg.paths,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    train_rows = rows_for_split(rows, split_map, "train")
    validation_rows = rows_for_split(rows, split_map, "validation")
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    candidates = [
        Site(layer=layer, position=position)
        for layer in cfg.hidden.layers
        for position in cfg.hidden.positions
    ]
    all_metrics = []
    best_axis = None
    best_site = None
    best_metrics = None
    for site in candidates:
        axis, metrics = train_and_validate_site(
            model,
            processor,
            train_rows,
            validation_rows,
            cfg,
            paths.data_dir,
            site,
            limit=args.limit,
        )
        all_metrics.append(metrics)
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_axis = axis
            best_site = site
            best_metrics = metrics

    if best_axis is None or best_site is None or best_metrics is None:
        raise RuntimeError("no RDO site was trained")

    save_axis(run_dir / cfg.rdo.axis_file, np.asarray(best_axis), best_site)
    save_selected_site(run_dir / cfg.rdo.selected_site_file, best_site, best_metrics)
    save_json({"candidates": all_metrics, "selected": best_metrics}, run_dir / cfg.rdo.validation_metrics_file)
    print(f"[rdo] selected layer={best_site.layer} position={best_site.position}")
    print(f"[rdo] axis -> {run_dir / cfg.rdo.axis_file}")


if __name__ == "__main__":
    main()
