#!/usr/bin/env -S uv run python
"""Validate and merge sharded SARSteer pairs in canonical manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.evaluation.sarsteer_shards import (
    ShardPair,
    validate_and_merge,
    write_merged,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="canonical input JSONL")
    parser.add_argument(
        "--shard",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("MANIFEST", "OUTPUT"),
        help="input shard manifest and its completed apply output; repeat per shard",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing merged output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform all validation without writing the merged output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    canonical = args.manifest.resolve()
    output = args.output.resolve()
    shards = [
        ShardPair(manifest=manifest.resolve(), output=shard_output.resolve())
        for manifest, shard_output in args.shard
    ]
    protected_paths = {canonical}
    protected_paths.update(shard.manifest for shard in shards)
    protected_paths.update(shard.output for shard in shards)
    if output in protected_paths:
        raise SystemExit("--output must differ from every input manifest and shard output")
    if output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"merged output already exists; pass --overwrite to replace it: {output}")

    try:
        result = validate_and_merge(canonical, shards)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.dry_run:
        write_merged(result, output)

    summary = {
        "status": "validated" if args.dry_run else "merged",
        "canonical_rows": result.canonical_count,
        "shard_rows": list(result.shard_counts),
        "merged_rows": len(result.rows),
        "canonical_order_verified": True,
        "stable_key_union_verified": True,
        "generation_bodies_logged": False,
        "output": None if args.dry_run else str(output),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
