#!/usr/bin/env -S uv run python
"""Fail-closed merge of ALMGuard full/sharded/positive-control arm outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.pipelines.almguard_run9 import (
    ALMGuardRun9Error,
    ArmView,
    atomic_save_jsonl,
    merge_aligned_views,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--view",
        type=Path,
        nargs=3,
        action="append",
        required=True,
        metavar=("MANIFEST", "UNDEFENDED", "DEFENDED"),
        help="canonical manifest and its completed arm outputs; repeat in final order",
    )
    parser.add_argument("--undefended-out", type=Path, required=True)
    parser.add_argument("--defended-out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    undefended_out = args.undefended_out.resolve()
    defended_out = args.defended_out.resolve()
    if undefended_out == defended_out:
        raise SystemExit("the two merged output paths must differ")
    views = [
        ArmView(manifest.resolve(), undefended.resolve(), defended.resolve())
        for manifest, undefended, defended in args.view
    ]
    protected = {path for view in views for path in (view.manifest, view.undefended, view.defended)}
    if undefended_out in protected or defended_out in protected:
        raise SystemExit("merged outputs must not overwrite any input view")
    existing = [path for path in (undefended_out, defended_out) if path.exists()]
    if existing and not args.overwrite and not args.dry_run:
        raise SystemExit("merged output exists; pass --overwrite to replace both arms")
    try:
        result = merge_aligned_views(views, data_dir=args.data_dir.resolve())
    except ALMGuardRun9Error as exc:
        raise SystemExit(str(exc)) from exc
    if not args.dry_run:
        atomic_save_jsonl(result.undefended, undefended_out)
        atomic_save_jsonl(result.defended, defended_out)
    print(
        json.dumps(
            {
                "status": "validated" if args.dry_run else "merged",
                "view_rows": list(result.view_counts),
                "total_rows": len(result.undefended),
                "arm_alignment_verified": True,
                "canonical_order_verified": True,
                "generation_bodies_logged": False,
                "undefended_out": None if args.dry_run else str(undefended_out),
                "defended_out": None if args.dry_run else str(defended_out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
