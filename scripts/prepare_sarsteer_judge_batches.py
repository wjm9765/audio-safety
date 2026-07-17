#!/usr/bin/env -S uv run python
"""Split paired SARSteer generations into batches for local agent adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.evaluation.agent_judge_io import build_batches
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=None,
        help="keep only sweep rows with this sweep_alpha (alpha-sweep inputs only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_jsonl(args.paired)
    if args.filter_alpha is not None:
        rows = [r for r in rows if float(r.get("sweep_alpha", -1)) == args.filter_alpha]
        if not rows:
            raise SystemExit(f"no row with sweep_alpha={args.filter_alpha}")
    batches = build_batches(rows, batch_size=args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for existing in args.out_dir.glob("batch_*.json"):
        existing.unlink()
    for index, batch in enumerate(batches):
        path = args.out_dir / f"batch_{index:03d}.json"
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(f"[judge] {len(rows)} paired rows -> {len(batches)} batches in {args.out_dir}")


if __name__ == "__main__":
    main()
