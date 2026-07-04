#!/usr/bin/env -S uv run python
"""Generate a harmful-benign pair draft manifest with OpenRouter.

Example:
    ./scripts/prepare_audio_rdo_pairs.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --limit 150
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import generate_pair_manifest, load_harmful_seed_rows
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--limit", type=int, default=None, help="max seed rows to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)
    seed_path = paths.data_dir / cfg.dataset.seed_file
    output_path = paths.data_dir / cfg.dataset.source_file
    seed_rows = load_harmful_seed_rows(seed_path, source=cfg.dataset.harmful_source)
    pairs = generate_pair_manifest(
        seed_rows,
        cfg.dataset.pair_generation,
        output_path,
        limit=args.limit or cfg.dataset.n_pairs,
    )
    print(f"[pairs] wrote {len(pairs)} draft pairs -> {output_path}")
    if cfg.dataset.pair_generation.review_required:
        print("[pairs] review_required=true; inspect/edit needs_review rows before running geometry.")


if __name__ == "__main__":
    main()
