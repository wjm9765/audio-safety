"""Single-arm local-agent adjudication for the ALMGuard aligned gate.

The ALMGuard arm generates its undefended and defended responses in separate,
in-child passes (``load_aligned_observations``), and — unlike the SARSteer paired
flow — the two arms are judged INDEPENDENTLY:

* It suits the two-phase schedule: the undefended arm is judged while the SAP is
  still training, before the defended arm exists.
* Each generation is labelled blind to the other arm, so a defended refusal can
  never anchor the undefended label (or vice versa).

Like :mod:`audio_safety.evaluation.agent_judge_io`, this owns the two boundaries a
silent error would corrupt: batch emission (self-describing, keyed by ``record_id``)
and label merge (every requested ``record_id`` returns exactly once with an on-taxonomy
label, bound to the SHA-256 of the exact judged text). Missing/extra/duplicate/
off-taxonomy labels raise rather than silently dropping a row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from audio_safety.evaluation.defense_gate import (
    BENIGN,
    COMPLIANCE,
    HARMFUL,
    VALID_BEHAVIOR_LABELS,
)

JUDGE_CONTEXT_FIELDS = (
    "record_id",
    "item_id",
    "gate_role",
    "safety_label",
    "condition",
    "sign",
    "reference_text",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _labels_allowed(safety_label: str) -> set[str]:
    """Behaviour labels admissible for a row's safety class.

    Harmful audio can never be a ``benign_answer`` and benign audio can never be a
    ``harmful_compliance`` — the same rule ``defense_gate._validate_observations``
    enforces downstream. Rejecting the impossible label at judge time turns a
    mislabel into a loud error instead of a silent downstream ValueError.
    """
    if safety_label == HARMFUL:
        return VALID_BEHAVIOR_LABELS - {"benign_answer"}
    if safety_label == BENIGN:
        return VALID_BEHAVIOR_LABELS - {COMPLIANCE}
    raise ValueError(f"unsupported safety_label {safety_label!r}")


def build_arm_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_field: str = "output",
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    """Split one arm's generations into independent judging batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"arm row {index} lacks a string record_id")
        if record_id in seen:
            raise ValueError(f"duplicate record_id in arm input: {record_id}")
        seen.add(record_id)
        output = row.get(output_field)
        if not isinstance(output, str):
            raise ValueError(f"arm row {record_id} lacks string {output_field!r}")
        item = {field: row.get(field) for field in JUDGE_CONTEXT_FIELDS}
        item["output"] = output
        item["allowed_labels"] = sorted(
            _labels_allowed(str(row.get("safety_label")))
        )
        items.append(item)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def merge_arm_labels(
    rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    output_field: str = "output",
    resolution: str,
) -> list[dict[str, Any]]:
    """Validate one arm's agent labels and emit a single-label sidecar.

    Each emitted row carries ``reviewed_behavior_label`` (consumed by
    ``defense_gate.load_aligned_observations``) plus the SHA-256 of the judged text.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"arm row {index} lacks a string record_id")
        if record_id in by_id:
            raise ValueError(f"duplicate record_id in arm input: {record_id}")
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
        value = label.get("reviewed_behavior_label")
        if value not in VALID_BEHAVIOR_LABELS:
            raise ValueError(
                f"record {record_id} has label {value!r}; expected one of "
                f"{sorted(VALID_BEHAVIOR_LABELS)}"
            )
        allowed = _labels_allowed(str(by_id[record_id].get("safety_label")))
        if value not in allowed:
            raise ValueError(
                f"record {record_id} ({by_id[record_id].get('safety_label')}) cannot be "
                f"labelled {value!r}; allowed: {sorted(allowed)}"
            )
        labels[record_id] = label

    missing = sorted(set(by_id) - set(labels))
    if missing:
        raise ValueError(f"{len(missing)} arm rows were never labelled, e.g. {missing[:5]}")

    out: list[dict[str, Any]] = []
    for record_id, row in by_id.items():
        label = labels[record_id]
        merged: dict[str, Any] = {
            "record_id": record_id,
            "item_id": row.get("item_id"),
            "safety_label": row.get("safety_label"),
            "condition": row.get("condition"),
            "sign": row.get("sign"),
            "gate_role": row.get("gate_role"),
            "reviewed_behavior_label": label["reviewed_behavior_label"],
            "output_sha256": _sha256(row[output_field]),
            "label_resolution": resolution,
            "manual_local_no_external_judge": True,
        }
        reason = label.get("reason")
        if reason is not None:
            merged["reason"] = reason
        out.append(merged)
    return out
