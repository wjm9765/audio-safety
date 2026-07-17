#!/usr/bin/env -S uv run python
"""Build the reduced, record-id-keyed ALMGuard gate cohort.

Combines three conditions into ONE manifest whose rows carry stable ``record_id``s
so the isolated ALMGuard undefended and defended arms align by ``record_id`` (never
by line order):

* ``clean``  harmful  — defines the per-item undefended refusal baseline.
* ``pv_standard``     — the low-level channel attack under test (both signs); defines
                        the vulnerable set S and the survival numerator.
* ``clean``  benign   — over-refusal (benign-cost) control.

The 27-row AdvWave/PAIR positive control keeps its own manifest and is evaluated
separately; it is intentionally NOT merged here (distinct record_id namespace).

The cohort can be deterministically subsampled by harmful/benign *item* (keeping an
item's clean + both attack signs together) so the in-child eval fits the GPU budget.
This is a direction-finding gate (not a pre-registered §0 gate), so a right-sized
cohort is a scoping choice, recorded in the emitted summary.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import (
    atomic_save_jsonl,
    record_id_for_key,
    stable_row_key,
)

ATTACK_UNDER_TEST = "pv_standard"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _subsample_items(item_ids: list[str], n: int, rng: random.Random) -> set[str]:
    unique = sorted(set(item_ids))
    if n <= 0 or n >= len(unique):
        return set(unique)
    return set(rng.sample(unique, n))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True, help="run9_fresh_clean.jsonl")
    parser.add_argument("--attacks", type=Path, required=True, help="run9_fresh_attacks.jsonl")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n-harmful-items", type=int, default=0, help="0 = all harmful clean items"
    )
    parser.add_argument(
        "--n-benign-items", type=int, default=0, help="0 = all benign clean items"
    )
    parser.add_argument("--attack-condition", default=ATTACK_UNDER_TEST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.out}")

    clean = _load_jsonl(args.clean.resolve())
    attacks = _load_jsonl(args.attacks.resolve())

    harmful_clean = [
        r for r in clean if r.get("safety_label") == "harmful" and r.get("condition") == "clean"
    ]
    benign_clean = [
        r for r in clean if r.get("safety_label") == "benign" and r.get("condition") == "clean"
    ]
    attack_rows = [r for r in attacks if r.get("condition") == args.attack_condition]
    if not harmful_clean or not attack_rows:
        raise SystemExit("need at least one harmful-clean and one attack row")

    rng = random.Random(args.seed)
    keep_harmful = _subsample_items([r["item_id"] for r in harmful_clean], args.n_harmful_items, rng)
    keep_benign = _subsample_items([r["item_id"] for r in benign_clean], args.n_benign_items, rng)

    # An attack item with no surviving clean-harmful partner could never enter S,
    # so drop it: keep only attack rows whose item is a kept harmful-clean item.
    selected: list[dict[str, Any]] = []
    selected += [r for r in harmful_clean if r["item_id"] in keep_harmful]
    selected += [r for r in attack_rows if r["item_id"] in keep_harmful]
    selected += [r for r in benign_clean if r["item_id"] in keep_benign]

    out_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        key = stable_row_key(row)
        record_id = record_id_for_key(key)
        if record_id in seen:
            raise SystemExit(f"record_id collision {record_id} for key {key}")
        seen.add(record_id)
        out_rows.append({**row, "record_id": record_id})

    # Stable order: clean-harmful, then attack (by item, sign), then benign — so the
    # staged wav order is deterministic across undefended/defended arms.
    def _order(row: dict[str, Any]) -> tuple[int, str, str]:
        rank = {"clean-h": 0, "attack": 1, "clean-b": 2}
        if row["condition"] == args.attack_condition:
            bucket = "attack"
        elif row["safety_label"] == "harmful":
            bucket = "clean-h"
        else:
            bucket = "clean-b"
        return (rank[bucket], str(row["item_id"]), str(row.get("sign", "")))

    out_rows.sort(key=_order)
    atomic_save_jsonl(out_rows, args.out.resolve())

    condition_counts = Counter((r["condition"], r["safety_label"]) for r in out_rows)
    summary = {
        "out": str(args.out.resolve()),
        "rows": len(out_rows),
        "unique_record_ids": len(seen),
        "harmful_items": len(keep_harmful),
        "benign_items": len(keep_benign),
        "attack_condition": args.attack_condition,
        "condition_counts": {f"{c}|{s}": n for (c, s), n in sorted(condition_counts.items())},
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
