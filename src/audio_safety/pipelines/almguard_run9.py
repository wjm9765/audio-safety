"""Fail-closed Run 9 manifest and arm-output helpers for ALMGuard.

ALMGuard is executed in an isolated environment, so this module intentionally
contains no torch or upstream imports.  It normalizes the held-out SAP positive
control, checks that each returned response still belongs to its canonical input
row, and merges independently generated views without logging request/response
bodies.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALMGUARD_POSITIVE_CONTROL = "almguard_sap_holdout"


class ALMGuardRun9Error(ValueError):
    """Raised when a Run 9 ALMGuard artifact violates its data contract."""


@dataclass(frozen=True)
class ArmView:
    """Canonical manifest and the two ALMGuard generation arms for one view."""

    manifest: Path
    undefended: Path
    defended: Path


@dataclass(frozen=True)
class AlignedArms:
    """Canonical-order ALMGuard arms and non-sensitive view counts."""

    undefended: list[dict[str, Any]]
    defended: list[dict[str, Any]]
    view_counts: tuple[int, ...]


def load_object_jsonl(path: Path, *, role: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ALMGuardRun9Error(f"{role} not found: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ALMGuardRun9Error(
                        f"{role} line {line_number} is not a JSON object"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ALMGuardRun9Error(f"failed to read {role} {path}: {exc}") from exc
    if not rows:
        raise ALMGuardRun9Error(f"{role} is empty: {path}")
    return rows


def atomic_save_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically replace a JSONL artifact after flushing file and directory."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _text(row: Mapping[str, Any], field: str, *, role: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ALMGuardRun9Error(
            f"{role} row {index} requires a non-empty string field {field!r}"
        )
    return value.strip()


def condition_of(row: Mapping[str, Any], *, role: str, index: int) -> str:
    value = row.get("condition")
    if value is None:
        value = row.get("style")
    if not isinstance(value, str) or not value.strip():
        raise ALMGuardRun9Error(
            f"{role} row {index} requires a non-empty condition (or legacy style)"
        )
    return value.strip()


def sign_token(value: Any, *, role: str, index: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ALMGuardRun9Error(f"{role} row {index} has an invalid boolean sign")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ALMGuardRun9Error(
            f"{role} row {index} has a non-numeric sign {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ALMGuardRun9Error(f"{role} row {index} has a non-finite sign")
    if number == 0:
        number = 0.0
    return f"{number:.12g}"


def _resolved_audio(row: Mapping[str, Any], data_dir: Path, *, role: str, index: int) -> Path:
    raw = _text(row, "path", role=role, index=index)
    path = Path(raw)
    resolved = (path if path.is_absolute() else data_dir / path).resolve()
    if not resolved.is_file():
        raise ALMGuardRun9Error(f"{role} row {index} audio not found: {resolved}")
    return resolved


def index_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    role: str,
    require_audio: bool = True,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    """Validate canonical judge/evaluator fields and index by ``record_id``."""

    if not rows:
        raise ALMGuardRun9Error(f"{role} rows are empty")
    order: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        record_id = _text(row, "record_id", role=role, index=index)
        if record_id in indexed:
            raise ALMGuardRun9Error(f"duplicate record_id in {role}: {record_id}")
        _text(row, "item_id", role=role, index=index)
        safety = _text(row, "safety_label", role=role, index=index)
        if safety not in {"harmful", "benign"}:
            raise ALMGuardRun9Error(
                f"{role} row {index} has unsupported safety_label {safety!r}"
            )
        condition_of(row, role=role, index=index)
        sign_token(row.get("sign"), role=role, index=index)
        _text(row, "reference_text", role=role, index=index)
        if require_audio:
            _resolved_audio(row, data_dir, role=role, index=index)
        order.append(record_id)
        indexed[record_id] = row
    return order, indexed


def normalize_positive_control_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    condition: str = ALMGUARD_POSITIVE_CONTROL,
) -> list[dict[str, Any]]:
    """Add the fields required by the shared judge/evaluator to SAP holdouts."""

    if not condition.strip():
        raise ALMGuardRun9Error("positive-control condition must be non-empty")
    if condition.strip() in {"clean", "pv_standard"}:
        raise ALMGuardRun9Error("positive-control condition must differ from clean/pv_standard")
    normalized: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        record_id = _text(row, "record_id", role="positive-control input", index=index)
        if record_id in seen_record_ids:
            raise ALMGuardRun9Error(f"duplicate positive-control record_id: {record_id}")
        seen_record_ids.add(record_id)
        _text(row, "item_id", role="positive-control input", index=index)
        _text(row, "attack_family", role="positive-control input", index=index)
        _text(row, "reference_text", role="positive-control input", index=index)
        _resolved_audio(row, data_dir, role="positive-control input", index=index)
        row.update(
            {
                "safety_label": "harmful",
                "condition": condition.strip(),
                "sign": None,
                "control_type": "almguard_official_attack_holdout",
                "data_partition": "evaluation",
                "eval_only": True,
                "evaluation_eligible": True,
                "almguard_training_eligible": False,
            }
        )
        normalized.append(row)
    index_eval_rows(
        normalized,
        data_dir=data_dir,
        role="normalized positive-control manifest",
    )
    return normalized


def _metadata_tuple(row: Mapping[str, Any], *, role: str, index: int) -> tuple[str, ...]:
    return (
        _text(row, "item_id", role=role, index=index),
        _text(row, "safety_label", role=role, index=index),
        condition_of(row, role=role, index=index),
        sign_token(row.get("sign"), role=role, index=index),
        _text(row, "path", role=role, index=index),
        _text(row, "reference_text", role=role, index=index),
    )


def align_arm_to_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    role: str,
    expected_defense: str,
) -> list[dict[str, Any]]:
    """Validate a generated arm and return it in canonical manifest order."""

    order, manifest = index_eval_rows(
        manifest_rows, data_dir=data_dir, role=f"{role} manifest"
    )
    _, arm = index_eval_rows(
        arm_rows,
        data_dir=data_dir,
        role=f"{role} output",
    )
    if set(manifest) != set(arm):
        missing = set(manifest) - set(arm)
        extra = set(arm) - set(manifest)
        raise ALMGuardRun9Error(
            f"{role} record_ids do not match its manifest "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    aligned: list[dict[str, Any]] = []
    for index, record_id in enumerate(order):
        expected = manifest[record_id]
        actual = arm[record_id]
        if _metadata_tuple(actual, role=f"{role} output", index=index) != _metadata_tuple(
            expected, role=f"{role} manifest", index=index
        ):
            raise ALMGuardRun9Error(
                f"{role} output metadata differs from manifest for record_id {record_id}"
            )
        if actual.get("defense") != expected_defense:
            raise ALMGuardRun9Error(
                f"{role} output has wrong defense tag for record_id {record_id}"
            )
        if not isinstance(actual.get("output"), str):
            raise ALMGuardRun9Error(
                f"{role} output lacks a completed response for record_id {record_id}"
            )
        aligned.append(dict(actual))
    return aligned


def merge_aligned_views(views: Sequence[ArmView], *, data_dir: Path) -> AlignedArms:
    """Merge full/sharded/positive-control views into two canonical aligned arms."""

    if not views:
        raise ALMGuardRun9Error("at least one ALMGuard view is required")
    all_undefended: list[dict[str, Any]] = []
    all_defended: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: list[int] = []
    for view_index, view in enumerate(views):
        manifest_rows = load_object_jsonl(view.manifest, role=f"view {view_index} manifest")
        order, _ = index_eval_rows(
            manifest_rows,
            data_dir=data_dir,
            role=f"view {view_index} manifest",
        )
        overlap = seen.intersection(order)
        if overlap:
            raise ALMGuardRun9Error(
                f"record_id appears in multiple canonical views: {next(iter(overlap))}"
            )
        seen.update(order)
        undefended = align_arm_to_manifest(
            manifest_rows,
            load_object_jsonl(view.undefended, role=f"view {view_index} undefended output"),
            data_dir=data_dir,
            role=f"view {view_index} undefended",
            expected_defense="none",
        )
        defended = align_arm_to_manifest(
            manifest_rows,
            load_object_jsonl(view.defended, role=f"view {view_index} defended output"),
            data_dir=data_dir,
            role=f"view {view_index} defended",
            expected_defense="almguard",
        )
        all_undefended.extend(undefended)
        all_defended.extend(defended)
        counts.append(len(manifest_rows))

    if len(all_undefended) != len(all_defended):
        raise AssertionError("validated ALMGuard arms unexpectedly differ in length")
    for canonical_index, (undefended, defended) in enumerate(
        zip(all_undefended, all_defended, strict=True)
    ):
        if undefended["record_id"] != defended["record_id"]:
            raise AssertionError("validated ALMGuard arms unexpectedly lost alignment")
        for row in (undefended, defended):
            if "index" in row:
                row["invocation_index"] = row["index"]
            row["index"] = canonical_index
            row["canonical_index"] = canonical_index
    return AlignedArms(
        undefended=all_undefended,
        defended=all_defended,
        view_counts=tuple(counts),
    )
