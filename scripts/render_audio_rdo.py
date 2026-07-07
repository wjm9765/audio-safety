#!/usr/bin/env -S uv run python
"""Render Audio-RDO harmful/benign pairs with the configured TTS adapter.

Example dry run:
    ./scripts/render_audio_rdo.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --dry-run
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs, render_audio_records
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--dry-run", action="store_true", help="write planned manifest only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    records = render_audio_records(
        pairs,
        cfg.dataset,
        cfg.dataset.tts,
        paths.data_dir,
        dry_run=args.dry_run,
    )
    manifest = paths.data_dir / cfg.dataset.tts.manifest_file
    print(f"[render] wrote {len(records)} render records -> {manifest}")


if __name__ == "__main__":
    main()
