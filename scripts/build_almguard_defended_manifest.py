#!/usr/bin/env -S uv run python
"""Two-phase ALMGuard gate: build the DEFENDED cohort from undefended labels.

The frozen vulnerable set S is finalized from the UNDEFENDED arm's labels ONLY —
before any defended label exists — so the defended eval cannot influence S. To
respect the evaluator's item-clustered averaging, we defend BOTH attack signs for
every item that has at least one vulnerable observation (never only the sign that
happened to comply). All clean-harmful and benign rows are also defended (the
defended arm is a full deployment: the SAP is applied to every input). The 27-row
positive control is defended separately from its own manifest.

The emitted manifest is a strict subset of the full gate manifest (same record_ids),
so ``defense_gate.load_aligned_observations`` can align the two arms after the
undefended arm is subset to these record_ids.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl
from audio_safety.evaluation.almguard_gate import (
    compute_vulnerable_items,
    select_defended_rows,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-manifest", type=Path, required=True, help="full 1200-row gate manifest")
    parser.add_argument("--undefended-labels", type=Path, required=True, help="merged undefended sidecar")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attack-condition", default="pv_standard")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.out}")

    gate = _load_jsonl(args.gate_manifest.resolve())
    labels = _load_jsonl(args.undefended_labels.resolve())
    try:
        vulnerable_items = compute_vulnerable_items(
            labels, attack_condition=args.attack_condition
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    defended = select_defended_rows(
        gate, vulnerable_items, attack_condition=args.attack_condition
    )
    atomic_save_jsonl(defended, args.out.resolve())

    counts = Counter((r["condition"], r["safety_label"]) for r in defended)
    summary = {
        "out": str(args.out.resolve()),
        "attack_condition": args.attack_condition,
        "n_vulnerable_items": len(vulnerable_items),
        "defended_rows": len(defended),
        "condition_counts": {f"{c}|{s}": n for (c, s), n in sorted(counts.items())},
        "note": "S finalized from undefended labels only; both attack signs defended per S item",
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
