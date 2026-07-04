#!/usr/bin/env -S uv run python
"""CLI entry point for experiment runs. Thin by design (AGENTS.md): parses args,
resolves config/paths/seed, snapshots reproducibility info, dispatches to pipelines.

Example:
    ./scripts/run_experiment.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --run-name exp1_20260704_1200_audio_rdo_gate \
        --override stats.n_permutations=1000
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.utils.io import snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir
from audio_safety.utils.seed import set_seed

STAGES = (
    "data",
    "behavior",
    "rdo",
    "baselines",
    "style_escape",
    "restoration",
    "stats",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. stats.n_permutations=1000)",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--output-dir", type=Path, default=None, help="override output root")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)
    set_seed(cfg.seed)

    run_name = args.run_name or f"{cfg.name}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)
    print(f"[run] {run_name}")
    print(f"[run] config snapshot -> {run_dir / 'config_snapshot.yaml'}")

    # Stage dispatch. Stages fill in as GPU/server scripts land (design.md §6):
    #   data          -> harmful/benign pair manifest + CosyVoice2 render manifest
    #   behavior      -> 4-way output labels, decoding-failure accounting
    #   rdo           -> layer/position sweep + Audio-RDO axis optimization
    #   baselines     -> MDSteer-c2r, SARSteer-style text vector, random controls
    #   style_escape  -> benign-controlled occupancy/escape analysis
    #   restoration   -> axis-coordinate restoration intervention
    #   stats         -> gate decision + bootstrap CIs
    raise NotImplementedError(
        f"stage {args.stage!r} not wired yet — implement per design.md §6 skeleton"
    )


if __name__ == "__main__":
    main()
