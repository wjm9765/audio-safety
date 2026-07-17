#!/usr/bin/env -S uv run python
"""Emit SARSteer's paired harmful/safe calibration manifests (paper §3.2).

Writes a refusal manifest (harmful audio -> Eq. 4 contrast) and a benign manifest
(the purified safe counterparts -> §4.2 safe-space PCA), restricted to items that
appear in NO evaluation partition. Also writes a summary recording the exclusions,
the SHA-256 of each manifest, and a disjointness proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audio_safety.data.run9_sarsteer_calibration import (
    calibration_rows,
    paired_neutral_items,
    select_calibration_items,
)
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True, help="attack-flip renders JSONL")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        metavar="JSONL",
        help="evaluation partition whose item_ids must not be calibrated on (repeatable)",
    )
    parser.add_argument("--out-refusal", type=Path, required=True)
    parser.add_argument("--out-benign", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--source", type=str, default="figstep_safebench")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="calibration pairs to keep (default: every eligible pair; paper uses 100)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    renders = load_jsonl(args.renders)
    pairs = paired_neutral_items(renders)

    excluded: set[str] = set()
    exclusions: dict[str, int] = {}
    for path in args.exclude:
        ids = {r["item_id"] for r in load_jsonl(path) if isinstance(r.get("item_id"), str)}
        exclusions[str(path)] = len(ids & set(pairs))
        excluded |= ids

    item_ids = select_calibration_items(
        renders, excluded_item_ids=excluded, n=args.n, seed=args.seed
    )
    refusal = calibration_rows(renders, item_ids, label="harmful", source=args.source)
    benign = calibration_rows(renders, item_ids, label="benign", source=args.source)
    _write_jsonl(args.out_refusal, refusal)
    _write_jsonl(args.out_benign, benign)

    leaked = sorted(set(item_ids) & excluded)
    if leaked:
        raise SystemExit(f"BUG: calibration leaked evaluation items: {leaked[:10]}")

    summary = {
        "role": "sarsteer_paper_calibration",
        "protocol": "arXiv:2510.17633 §3.2 paired harmful/purified-safe; Eq.4 + §4.2 PCA",
        "renders": str(args.renders),
        "seed": args.seed,
        "n_requested": args.n,
        "n_selected": len(item_ids),
        "paired_pool": len(pairs),
        "excluded_partitions": exclusions,
        "excluded_item_ids_total": len(excluded),
        "eval_disjoint": True,
        "refusal_manifest": str(args.out_refusal),
        "benign_manifest": str(args.out_benign),
        "refusal_sha256": _sha256(args.out_refusal),
        "benign_sha256": _sha256(args.out_benign),
        "item_ids": item_ids,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(
        f"[calib] paired pool={len(pairs)} excluded={len(excluded & set(pairs))} "
        f"-> selected {len(item_ids)} pairs\n"
        f"[calib] refusal -> {args.out_refusal}\n"
        f"[calib] benign  -> {args.out_benign}\n"
        f"[calib] summary -> {args.out_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
