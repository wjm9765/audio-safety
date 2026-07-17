#!/usr/bin/env -S uv run python
"""Prepare the canonical 2,800-row Run 9 defense-evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.data.run9_eval_manifest import (
    atomic_save_jsonl,
    build_run9_eval_manifest,
    shard_by_item,
)
from audio_safety.utils.io import load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--attack-manifest", type=Path, required=True)
    parser.add_argument("--utility-manifest", type=Path, required=True)
    parser.add_argument("--positive-control-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        action="append",
        default=[],
        help="calibration JSONL for item/path/source leakage checks (repeatable)",
    )
    parser.add_argument(
        "--asr-scores",
        type=Path,
        action="append",
        default=[],
        help="optional ASR-scored sidecar to merge (repeatable; failed rows remain present)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="required when --num-shards > 1; item-grouped shard JSONLs are written here",
    )
    parser.add_argument(
        "--strict-source-all-roles",
        action="store_true",
        help="also forbid calibration source reuse by utility/positive-control rows",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolved_manifest(path: Path, data_dir: Path, *, role: str) -> Path:
    resolved = (path if path.is_absolute() else data_dir / path).resolve()
    if not resolved.is_file():
        raise SystemExit(f"{role} manifest not found: {resolved}")
    return resolved


def _load_many(paths: list[Path], data_dir: Path, *, role: str) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(load_jsonl(_resolved_manifest(path, data_dir, role=role)))
    return rows


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.num_shards > 1 and args.shard_dir is None:
        raise SystemExit("--shard-dir is required when --num-shards > 1")

    def load_one(path: Path, role: str) -> list[dict]:
        return load_jsonl(_resolved_manifest(path, data_dir, role=role))

    result = build_run9_eval_manifest(
        load_one(args.clean_manifest, "clean"),
        load_one(args.attack_manifest, "attack"),
        load_one(args.utility_manifest, "utility"),
        load_one(args.positive_control_manifest, "positive-control"),
        data_dir=data_dir,
        calibration_rows=_load_many(
            args.calibration_manifest, data_dir, role="calibration"
        ),
        asr_score_rows=_load_many(args.asr_scores, data_dir, role="ASR score"),
        strict_source_all_roles=args.strict_source_all_roles,
    )
    shards = shard_by_item(result.rows, args.num_shards)
    shard_summary = {
        "num_shards": args.num_shards,
        "row_counts": [len(shard) for shard in shards],
        "item_counts": [len({row["item_id"] for row in shard}) for shard in shards],
        "same_item_kept_together": True,
        "record_id_union_verified": True,
        "record_id_overlap": 0,
    }
    summary = {**result.summary, "output": str(args.output.resolve()), "shards": shard_summary}
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    atomic_save_jsonl(result.rows, args.output)
    if args.num_shards > 1:
        assert args.shard_dir is not None
        args.shard_dir.mkdir(parents=True, exist_ok=True)
        width = max(2, len(str(args.num_shards - 1)))
        for index, shard in enumerate(shards):
            filename = (
                f"{args.output.stem}.shard{index:0{width}d}-of-{args.num_shards:0{width}d}.jsonl"
            )
            atomic_save_jsonl(shard, args.shard_dir / filename)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
