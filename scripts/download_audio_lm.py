#!/usr/bin/env -S uv run python
"""Download/cache any configured audio LM into the project cache.

Generalises ``download_qwen2_audio.py`` to the exp2 model pool. Dispatch follows
``model.loader`` in the config, so an exp1 config still caches through the exact
Qwen2-Audio path it always did.

Takes either a full experiment config (uses its ``model:`` block) or a bare model
config, so a model can be fetched on a fresh GPU box before any experiment YAML
for it exists.

Examples:
    # via an experiment config
    ./scripts/download_audio_lm.py \\
        --config configs/experiments/exp1_refusal_cone_drift.yaml

    # via a bare model config (exp2 pool)
    ./scripts/download_audio_lm.py --model-config configs/models/minicpm_o_2_6.yaml

    # dry run: resolve and print, download nothing
    ./scripts/download_audio_lm.py --model-config configs/models/kimi_audio.yaml --dry-run
"""

import argparse
from pathlib import Path

import yaml

from audio_safety.config import load_experiment_config
from audio_safety.config.schema import ModelConfig, PathsConfig
from audio_safety.models.hf_audio_lm import download_hf_audio_lm
from audio_safety.models.qwen2_audio import download_qwen2_audio
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="experiment YAML (uses its model block)")
    source.add_argument("--model-config", type=Path, help="bare model YAML under configs/models/")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override model cache root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config and cache path, then exit without downloading",
    )
    return parser.parse_args()


def _load_model_config(args: argparse.Namespace) -> tuple[ModelConfig, PathsConfig | None]:
    if args.config is not None:
        cfg = load_experiment_config(args.config)
        return cfg.model, cfg.paths
    raw = yaml.safe_load(args.model_config.read_text(encoding="utf-8")) or {}
    return ModelConfig(**raw), None


def main() -> None:
    args = parse_args()
    model_cfg, paths_cfg = _load_model_config(args)
    paths = resolve_paths(paths_cfg, cache_dir=args.cache_dir)

    print(f"[audio-lm] model_id           = {model_cfg.model_id}")
    print(f"[audio-lm] loader             = {model_cfg.loader}")
    print(f"[audio-lm] revision           = {model_cfg.revision or '<repo default>'}")
    print(f"[audio-lm] trust_remote_code  = {model_cfg.trust_remote_code}")
    print(f"[audio-lm] dtype / device_map = {model_cfg.dtype} / {model_cfg.device_map}")
    print(f"[audio-lm] cache_dir          = {paths.cache_dir}")

    if args.dry_run:
        print("[audio-lm] dry run: nothing downloaded")
        return

    if model_cfg.loader == "qwen2_audio":
        download_qwen2_audio(model_cfg, cache_dir=paths.cache_dir)
    elif model_cfg.loader == "hf_auto_multimodal":
        download_hf_audio_lm(model_cfg, cache_dir=paths.cache_dir)
    else:
        raise ValueError(f"unknown loader {model_cfg.loader!r}")

    print(f"[audio-lm] cached {model_cfg.model_id} under {paths.cache_dir}")


if __name__ == "__main__":
    main()
