#!/usr/bin/env -S uv run python
"""Run the fast pitch-only encoder/projector/LLM representation gate.

Example:
    ./scripts/run_pitch_representation.py \
        --config configs/experiments/run5_pitch_representation_fast.yaml \
        --run-name run5_20260713_pitch_fast \
        --phase all
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.utils.io import snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir
from audio_safety.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--phase", choices=("extract", "analyze", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="override item count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.pitch_representation is None or not cfg.pitch_representation.enabled:
        raise SystemExit("config must enable `pitch_representation`")
    paths = resolve_paths(
        cfg.paths,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    set_seed(cfg.seed)
    run_name = args.run_name or f"{cfg.name}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)

    if args.phase in {"extract", "all"}:
        from audio_safety.models.qwen2_audio import load_qwen2_audio
        from audio_safety.pipelines.pitch_representation import extract_pitch_representation

        model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
        arrays, cells = extract_pitch_representation(
            model,
            processor,
            cfg,
            paths.data_dir,
            run_dir,
            item_limit=args.limit,
        )
        print(
            f"[pitch] captured {len(cells)} cells; arrays={sorted(arrays)} -> "
            f"{run_dir / cfg.pitch_representation.activations_file}"
        )
        del model, processor

    if args.phase in {"analyze", "all"}:
        from audio_safety.pipelines.pitch_representation import analyze_pitch_artifacts

        metrics = analyze_pitch_artifacts(cfg, run_dir, seed=cfg.seed)
        print(
            f"[pitch] outcome={metrics['screening_outcome']} -> "
            f"{run_dir / cfg.pitch_representation.report_file}"
        )


if __name__ == "__main__":
    main()
