#!/usr/bin/env -S uv run python
"""Normalize the held-out 27-row ALMGuard SAP positive-control manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.pipelines.almguard_run9 import (
    ALMGUARD_POSITIVE_CONTROL,
    ALMGuardRun9Error,
    atomic_save_jsonl,
    index_eval_rows,
    load_object_jsonl,
    normalize_positive_control_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", default=ALMGUARD_POSITIVE_CONTROL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    try:
        full = load_object_jsonl(args.full_manifest.resolve(), role="full Run9 manifest")
        full_order, _ = index_eval_rows(
            full, data_dir=data_dir, role="full Run9 manifest"
        )
        holdout = normalize_positive_control_rows(
            load_object_jsonl(args.holdout_manifest.resolve(), role="SAP holdout manifest"),
            data_dir=data_dir,
            condition=args.condition,
        )
        holdout_order, _ = index_eval_rows(
            holdout, data_dir=data_dir, role="normalized SAP holdout"
        )
        overlap = set(full_order).intersection(holdout_order)
        if overlap:
            raise ALMGuardRun9Error(
                f"full/holdout record_id overlap: {next(iter(overlap))}"
            )
    except ALMGuardRun9Error as exc:
        raise SystemExit(str(exc)) from exc
    if output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"output exists; pass --overwrite to replace it: {output}")
    if not args.dry_run:
        atomic_save_jsonl(holdout, output)
    print(
        json.dumps(
            {
                "status": "validated" if args.dry_run else "prepared",
                "full_rows": len(full),
                "positive_control_rows": len(holdout),
                "condition": args.condition,
                "record_id_overlap": 0,
                "generation_bodies_logged": False,
                "output": None if args.dry_run else str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
