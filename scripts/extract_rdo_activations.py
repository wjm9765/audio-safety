#!/usr/bin/env -S uv run python
"""Extract selected-site activations and baseline vectors for Audio-RDO."""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.models.qwen2_audio import load_qwen2_audio
from audio_safety.pipelines.rdo_gate import (
    capture_refusal_continuation_hidden,
    compute_baseline_vectors,
    extract_selected_site_activations,
    load_selected_site,
    split_ids,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
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
    site = load_selected_site(run_dir / cfg.rdo.selected_site_file)
    rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    activations, metadata = extract_selected_site_activations(
        model,
        processor,
        rows,
        cfg,
        paths.data_dir,
        run_dir,
        site,
    )

    train_harmful_rows = [
        row
        for row in rows
        if row["item_id"] in split_map["train"] and row["safety_label"] == "harmful"
    ]
    sar_hidden = np.stack(
        [
            capture_refusal_continuation_hidden(
                model,
                processor,
                row,
                cfg,
                paths.data_dir,
                site,
            )
            for row in train_harmful_rows
        ]
    )
    vectors = compute_baseline_vectors(
        activations,
        metadata,
        split_map["train"],
        sar_refusal_hidden=sar_hidden,
    )
    baseline_path = run_dir / cfg.rdo.baseline_vectors_file
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(baseline_path, **vectors)
    print(f"[activations] saved {activations.shape} -> {run_dir / cfg.rdo.activations_file}")
    print(f"[baselines] saved {list(vectors)} -> {baseline_path}")


if __name__ == "__main__":
    main()
