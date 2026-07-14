#!/usr/bin/env -S uv run python
"""Run the exploratory COAST-R Stage-A score, fit, and causal phases.

Example:
    ./scripts/run_coast_r_stage_a.py \
        --config configs/experiments/run7_coast_r_stage_a.yaml \
        --run-name run7_20260714_coast_r_stage_a \
        --phase all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.coast_r import (
    fit_coast_r,
    generate_coast_r_interventions,
    load_coast_r_source,
    require_coast_r_config,
    score_continuation_bank,
)
from audio_safety.utils.io import get_git_commit, snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir
from audio_safety.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--phase",
        choices=("score", "fit", "intervene", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def _ensure_snapshot(cfg, run_dir: Path) -> Path:
    """Write once and reject phase-to-phase config drift within one run."""
    path = run_dir / "config_snapshot.yaml"
    if not path.exists():
        return snapshot_config(cfg, run_dir)
    existing = yaml.safe_load(path.read_text())
    expected = cfg.model_dump(mode="json")
    if not isinstance(existing, dict) or existing.get("config") != expected:
        raise RuntimeError(
            f"config differs from the frozen snapshot in {run_dir}; use a new run name"
        )
    current_commit = get_git_commit()
    if existing.get("git_commit") != current_commit:
        raise RuntimeError(
            f"git commit differs from the frozen snapshot in {run_dir}; use a new run name"
        )
    return path


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    gate = require_coast_r_config(cfg)
    paths = resolve_paths(
        cfg.paths,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    set_seed(cfg.seed)
    requested_run_dir = paths.output_dir / args.run_name
    source_run_dir = paths.output_dir / gate.source_run_name
    if requested_run_dir.resolve() == source_run_dir.resolve():
        raise ValueError("Run 7 --run-name must differ from the immutable source run name")
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    snapshot_path = _ensure_snapshot(cfg, run_dir)
    needs_audio = args.phase in {"score", "intervene", "all"}
    source = load_coast_r_source(cfg, paths, require_audio=needs_audio)
    print(f"[coast-r] run dir: {run_dir}")
    print(f"[coast-r] source: {source.run_dir}")
    print(f"[coast-r] config snapshot: {snapshot_path}")
    print(
        "[coast-r] source hashes: "
        f"activations={source.activations_sha256[:12]} "
        f"cells={source.cells_sha256[:12]}"
    )

    model = processor = None

    def ensure_model():
        nonlocal model, processor
        if model is None or processor is None:
            from audio_safety.models.qwen2_audio import load_qwen2_audio

            model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
        return model, processor

    if args.phase in {"score", "all"}:
        active_model, active_processor = ensure_model()
        summary = score_continuation_bank(
            active_model,
            active_processor,
            cfg,
            source,
            run_dir,
        )
        print(
            f"[coast-r score] complete={summary['n_complete']}/{summary['n_selected']} "
            f"new={summary['n_written']} -> {summary['output']}"
        )

    if args.phase in {"fit", "all"}:
        metrics = fit_coast_r(cfg, source, run_dir)
        print(f"[coast-r fit] status={metrics.get('status')} -> {run_dir / gate.fit_metrics_file}")

    if args.phase in {"intervene", "all"}:
        active_model, active_processor = ensure_model()
        summary = generate_coast_r_interventions(
            active_model,
            active_processor,
            cfg,
            source,
            run_dir,
        )
        print(
            f"[coast-r intervene] complete={summary['n_complete']} "
            f"selected={summary['n_selected']} new={summary['n_written']} -> "
            f"{summary['output']}"
        )


if __name__ == "__main__":
    main()
