#!/usr/bin/env -S uv run python
"""Validate manual SARSteer label batches and emit one canonical-order sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl
from audio_safety.evaluation.defense_gate import VALID_BEHAVIOR_LABELS
from audio_safety.utils.io import load_jsonl

ARMS = ("undefended", "defended")


def merge_labels(paired_rows: list[dict], batches: list[list[dict]]) -> list[dict]:
    if not paired_rows:
        raise ValueError("paired output is empty")
    paired_ids: list[str] = []
    paired_by_id: dict[str, dict] = {}
    for index, row in enumerate(paired_rows):
        record_id = str(row.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"paired row {index} lacks record_id")
        if record_id in paired_by_id:
            raise ValueError(f"duplicate paired record_id: {record_id}")
        for arm in ARMS:
            if not isinstance(row.get(f"{arm}_output"), str):
                raise ValueError(f"paired row {index} lacks string {arm}_output")
        paired_ids.append(record_id)
        paired_by_id[record_id] = row

    decisions: dict[str, dict] = {}
    for batch_index, batch in enumerate(batches):
        for row_index, row in enumerate(batch):
            record_id = str(row.get("record_id") or "").strip()
            if not record_id:
                raise ValueError(f"label batch {batch_index} row {row_index} lacks record_id")
            if record_id in decisions:
                raise ValueError(f"duplicate manual label record_id: {record_id}")
            if record_id not in paired_by_id:
                raise ValueError(f"manual label has foreign record_id: {record_id}")
            if row.get("manual_reviewed") is not True:
                raise ValueError(f"manual label is not marked reviewed: {record_id}")
            for arm in ARMS:
                field = f"{arm}_reviewed_behavior_label"
                label = str(row.get(field) or "")
                if label not in VALID_BEHAVIOR_LABELS:
                    raise ValueError(f"{record_id} has invalid {field}={label!r}")
                note = row.get(f"{arm}_adjudication_note")
                if not isinstance(note, str) or not note.strip():
                    raise ValueError(f"{record_id} lacks {arm}_adjudication_note")
            decisions[record_id] = row

    missing = [record_id for record_id in paired_ids if record_id not in decisions]
    if missing:
        raise ValueError(
            f"manual labels do not cover paired output; missing {len(missing)} ids, "
            f"first={missing[:3]}"
        )

    merged: list[dict] = []
    for record_id in paired_ids:
        paired = paired_by_id[record_id]
        decision = decisions[record_id]
        row = {
            key: value
            for key, value in paired.items()
            if key not in {"undefended_output", "defended_output"}
        }
        for arm in ARMS:
            output = paired[f"{arm}_output"]
            row[f"{arm}_output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        row.update(decision)
        row["manual_adjudication_complete"] = True
        merged.append(row)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--label-batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = args.paired.resolve()
    batches = [path.resolve() for path in args.label_batch]
    output = args.output.resolve()
    if output in {paired, *batches}:
        raise SystemExit("--output must differ from every input")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    try:
        merged = merge_labels(load_jsonl(paired), [load_jsonl(path) for path in batches])
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    atomic_save_jsonl(merged, output)
    counts = {
        arm: dict(
            sorted(Counter(row[f"{arm}_reviewed_behavior_label"] for row in merged).items())
        )
        for arm in ARMS
    }
    print(
        json.dumps(
            {
                "rows": len(merged),
                "judgments": 2 * len(merged),
                "record_id_coverage_exact": True,
                "label_counts": counts,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
