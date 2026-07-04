#!/usr/bin/env -S uv run python
"""Download/cache Qwen2-Audio assets via the official transformers path.

Example:
    ./scripts/download_qwen2_audio.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.models.qwen2_audio import download_qwen2_audio
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override model cache root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, cache_dir=args.cache_dir)
    download_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    print(f"[qwen2-audio] cached {cfg.model.model_id} under {paths.cache_dir}")


if __name__ == "__main__":
    main()
