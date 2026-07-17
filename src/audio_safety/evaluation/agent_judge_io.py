"""Batch/merge helpers for local agent adjudication of defense-gate outputs.

The gate is judged locally by agents reading the generations — no OpenRouter, no
external judge API. This module owns the two boundaries where a silent error would
corrupt the verdict:

* :func:`build_batches` emits self-describing batches keyed by ``record_id``, so a
  labeller never has to infer which row it is looking at from position.
* :func:`merge_label_batches` refuses to guess: every requested ``record_id`` must
  come back exactly once with both arms labelled from the fixed taxonomy. Missing,
  extra, duplicated, or off-taxonomy labels raise instead of silently dropping a
  row — a dropped row would quietly shrink the vulnerable set S.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from audio_safety.evaluation.defense_gate import VALID_BEHAVIOR_LABELS

JUDGE_CONTEXT_FIELDS = ("record_id", "gate_role", "safety_label", "condition", "reference_text")
ARMS = ("undefended", "defended")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_batches(
    rows: Sequence[Mapping[str, Any]], *, batch_size: int
) -> list[list[dict[str, Any]]]:
    """Split paired rows into judging batches carrying only what a labeller needs."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"paired row {index} lacks a string record_id")
        if record_id in seen:
            raise ValueError(f"duplicate record_id in paired input: {record_id}")
        seen.add(record_id)
        item = {field: row.get(field) for field in JUDGE_CONTEXT_FIELDS}
        for arm in ARMS:
            output = row.get(f"{arm}_output")
            if not isinstance(output, str):
                raise ValueError(f"paired row {record_id} lacks string {arm}_output")
            item[f"{arm}_output"] = output
        items.append(item)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def merge_label_batches(
    rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    resolution: str,
) -> list[dict[str, Any]]:
    """Validate agent labels against the paired rows and emit the label sidecar.

    Returns rows carrying ``{arm}_reviewed_behavior_label`` plus the SHA-256 of the
    exact judged text, so a later reader can prove the label refers to the string
    that was actually generated.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"paired row {index} lacks a string record_id")
        if record_id in by_id:
            raise ValueError(f"duplicate record_id in paired input: {record_id}")
        by_id[record_id] = row
    labels: dict[str, Mapping[str, Any]] = {}
    for index, label in enumerate(label_rows):
        record_id = label.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"label row {index} lacks a string record_id")
        if record_id in labels:
            raise ValueError(f"duplicate label for record_id {record_id}")
        if record_id not in by_id:
            raise ValueError(f"label references unknown record_id {record_id}")
        for arm in ARMS:
            value = label.get(f"{arm}_reviewed_behavior_label")
            if value not in VALID_BEHAVIOR_LABELS:
                raise ValueError(
                    f"record {record_id} arm {arm} has label {value!r}; expected one of "
                    f"{sorted(VALID_BEHAVIOR_LABELS)}"
                )
        labels[record_id] = label

    missing = sorted(set(by_id) - set(labels))
    if missing:
        raise ValueError(f"{len(missing)} paired rows were never labelled, e.g. {missing[:5]}")

    out: list[dict[str, Any]] = []
    for record_id, row in by_id.items():
        label = labels[record_id]
        merged: dict[str, Any] = {
            "record_id": record_id,
            "item_id": row.get("item_id"),
            "safety_label": row.get("safety_label"),
            "condition": row.get("condition"),
            "style": row.get("style"),
            "sign": row.get("sign"),
            "gate_role": row.get("gate_role"),
            "manual_local_no_external_judge": True,
        }
        for arm in ARMS:
            merged[f"{arm}_reviewed_behavior_label"] = label[f"{arm}_reviewed_behavior_label"]
            merged[f"{arm}_output_sha256"] = _sha256(row[f"{arm}_output"])
            merged[f"{arm}_label_resolution"] = resolution
            reason = label.get(f"{arm}_reason")
            if reason is not None:
                merged[f"{arm}_reason"] = reason
        out.append(merged)
    return out
