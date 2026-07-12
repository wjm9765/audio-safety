#!/usr/bin/env -S uv run python
"""Capture matched text/audio activations for Run 4 Stage B (mechanism adjudication).

Reuses the Stage A behavior manifests (audio + text arms), restricts to the
neutral matched cell, and captures residual-stream activations at the content
position P1 and decision position P2 through the frozen audio refusal axis r_A.

Example:
    ./scripts/extract_conversion_activations.py \
        --config configs/experiments/run4_conversion_gap.yaml \
        --run-name run4_$(date +%Y%m%d_%H%M)_probe \
        --axis-artifact /workspace/.../exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.models.qwen2_audio import load_qwen2_audio
from audio_safety.pipelines.conversion_probe import extract_conversion_activations
from audio_safety.pipelines.rdo_gate import load_axis
from audio_safety.utils.io import load_jsonl, snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--axis-artifact", type=Path, default=None, help="frozen r_A rdo_axis.npz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--style", type=str, default="neutral", help="matched-cell style")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.conversion_probe is None or cfg.conversion_gap is None:
        raise SystemExit(
            "Stage B needs both `conversion_gap` (Stage A arms) and `conversion_probe`"
        )
    paths = resolve_paths(
        cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir, cache_dir=args.cache_dir
    )

    axis_path = args.axis_artifact or cfg.conversion_probe.frozen_axis_artifact
    if axis_path is None:
        raise SystemExit("provide the frozen r_A axis via --axis-artifact or config")
    r_a, site = load_axis(Path(axis_path))
    print(f"[probe] frozen r_A from {axis_path} (trained at layer {site.layer}/{site.position})")

    audio_rows = [
        {**r, "modality": "audio"}
        for r in load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    ]
    text_rows = load_jsonl(paths.data_dir / cfg.conversion_gap.text_arm.text_outputs_file)

    def eligible(row: dict) -> bool:
        return str(row.get("style")) == args.style and bool(
            row.get("transcript_control_passed", True)
        )

    rows = [r for r in (*audio_rows, *text_rows) if eligible(r)]
    if args.limit is not None:
        rows = rows[: args.limit]
    n_text = sum(r["modality"] == "text" for r in rows)
    print(
        f"[probe] rows: {len(rows)} (text={n_text}, audio={len(rows) - n_text}), "
        f"style={args.style}"
    )

    run_name = args.run_name or f"{cfg.name}_probe_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    _, metadata = extract_conversion_activations(
        model, processor, rows, cfg, paths.data_dir, run_dir, r_a
    )
    print(
        f"[probe] captured {len(metadata)} rows -> "
        f"{run_dir / cfg.conversion_probe.activations_file}"
    )


if __name__ == "__main__":
    main()
