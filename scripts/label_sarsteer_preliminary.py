#!/usr/bin/env -S uv run python
"""Create non-authoritative local heuristic labels for paired SARSteer outputs.

This is an offline direction check only.  It reuses the project's lexical
``label_output`` helper and does not replace blinded four-way adjudication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl
from audio_safety.evaluation.labeling import label_output
from audio_safety.utils.io import load_jsonl, save_json

ARMS = ("undefended", "defended")


def _required_text(row: Mapping[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paired row {index} requires non-empty string {field!r}")
    return value.strip()


def label_paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("paired SARSteer output is empty")
    sidecar: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    label_counts = {arm: Counter() for arm in ARMS}
    review_counts = Counter()
    transitions = Counter()

    for index, row in enumerate(rows):
        record_id = _required_text(row, "record_id", index=index)
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate paired record_id: {record_id}")
        seen_record_ids.add(record_id)
        safety_label = _required_text(row, "safety_label", index=index)
        if safety_label not in {"harmful", "benign"}:
            raise ValueError(f"paired row {index} has invalid safety_label={safety_label!r}")
        reference_text = row.get("reference_text")
        reference = str(reference_text) if reference_text is not None else None

        labeled = {
            key: row[key]
            for key in (
                "record_id",
                "item_id",
                "safety_label",
                "condition",
                "style",
                "sign",
                "path",
            )
            if key in row
        }
        arm_labels: dict[str, str] = {}
        for arm in ARMS:
            output = row.get(f"{arm}_output")
            if not isinstance(output, str):
                raise ValueError(f"paired row {index} lacks string {arm}_output")
            label, failure_mode, needs_review = label_output(
                output,
                safety_label=safety_label,
                reference_text=reference,
            )
            arm_labels[arm] = label
            label_counts[arm][label] += 1
            review_counts[arm] += int(needs_review)
            labeled[f"{arm}_behavior_label"] = label
            labeled[f"{arm}_decoding_failure_mode"] = failure_mode
            labeled[f"{arm}_needs_manual_review"] = needs_review
            labeled[f"{arm}_label_resolution"] = "heuristic_preliminary_unreviewed"
        transitions[(arm_labels["undefended"], arm_labels["defended"])] += 1
        labeled.update(
            {
                "preliminary_only": True,
                "authoritative_for_final_gate": False,
                "labeler": "audio_safety.evaluation.labeling.label_output",
            }
        )
        sidecar.append(labeled)

    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_HEURISTIC_NON_AUTHORITATIVE",
        "rows": len(sidecar),
        "unique_record_ids": len(seen_record_ids),
        "labeler": "audio_safety.evaluation.labeling.label_output",
        "offline": True,
        "requires_blinded_four_way_adjudication_for_final_gate": True,
        "label_counts": {arm: dict(sorted(label_counts[arm].items())) for arm in ARMS},
        "needs_manual_review_counts": dict(sorted(review_counts.items())),
        "transition_counts": {
            f"{before}->{after}": count for (before, after), count in sorted(transitions.items())
        },
    }
    return sidecar, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = args.paired.resolve()
    if not paired.is_file():
        raise SystemExit(f"paired SARSteer output not found: {paired}")
    try:
        labels, summary = label_paired_rows(load_jsonl(paired))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary.update(
        {
            "paired": str(paired),
            "labels_out": str(args.labels_out.resolve()),
        }
    )
    atomic_save_jsonl(labels, args.labels_out.resolve())
    save_json(summary, args.summary_out.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
