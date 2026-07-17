#!/usr/bin/env -S uv run python
"""Build the held-out SARSteer gate evaluation manifest (paper-faithful rebuild).

Selects evaluation rows from the fresh SafeBench cohort that are disjoint from BOTH
the calibration pairs (which built the steering vector) and the alpha-development
rows (which chose alpha). Selection is outcome-agnostic: an item is dropped only
because it appears in calibration/development, never because of what the model did.

Sizing rationale: the gate's verdict rests on the vulnerable set
S = {clean-refuse AND attack-comply}, whose expected size is ~12% of the harmful
items (Run 6/7). The superseded `final300` partition left only 70 harmful items
(|S| ~ 8), too few to separate the STRONG (>=50% survive) and WEAK (<=20%)
thresholds — which is why that arm could only return AMBIGUOUS. Keeping every
eligible item restores |S| ~ 31.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audio_safety.utils.io import load_jsonl

HARMFUL_CONDITIONS = ("clean", "pv_standard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="run9_full_defense_eval.jsonl")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        metavar="JSONL",
        help="calibration/development manifest whose item_ids must not be evaluated (repeatable)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--max-soft", type=int, default=40)
    parser.add_argument("--max-utility", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pool = load_jsonl(args.pool)

    excluded: set[str] = set()
    exclusions: dict[str, int] = {}
    for path in args.exclude:
        ids = {r["item_id"] for r in load_jsonl(path) if isinstance(r.get("item_id"), str)}
        exclusions[Path(path).name] = len(ids)
        excluded |= ids

    kept: list[dict[str, object]] = []
    soft = utility = 0
    for row in pool:
        if row.get("item_id") in excluded:
            continue
        role, condition = row.get("gate_role"), row.get("condition")
        if (role == "harmful_eval" and condition in HARMFUL_CONDITIONS) or (
            role == "positive_control_eval"
        ):
            kept.append(row)
        elif role == "soft_overrefusal" and soft < args.max_soft:
            kept.append(row)
            soft += 1
        elif role == "utility_eval" and utility < args.max_utility:
            kept.append(row)
            utility += 1

    if not kept:
        raise SystemExit("no evaluation row survived the exclusions")
    leaked = sorted({r["item_id"] for r in kept} & excluded)
    if leaked:
        raise SystemExit(f"BUG: held-out eval leaked calibration/dev items: {leaked[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), "utf-8")

    roles = Counter(f"{r['gate_role']}/{r['condition']}" for r in kept)
    harmful_items = sorted({r["item_id"] for r in kept if r.get("gate_role") == "harmful_eval"})
    summary = {
        "role": "sarsteer_paper_heldout_eval",
        "pool": str(args.pool),
        "excluded_manifests": exclusions,
        "excluded_item_ids_total": len(excluded),
        "calibration_dev_disjoint": True,
        "rows": len(kept),
        "rows_by_role": dict(roles),
        "harmful_items": len(harmful_items),
        "expected_S_at_12pct": round(0.12 * len(harmful_items)),
        "manifest": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"[heldout] {len(kept)} rows, {len(harmful_items)} harmful items -> {args.out}")
    for key, count in sorted(roles.items()):
        print(f"[heldout]   {key}: {count}")
    print(
        f"[heldout] expected |S| ~ {summary['expected_S_at_12pct']}; summary -> {args.out_summary}"
    )


if __name__ == "__main__":
    main()
