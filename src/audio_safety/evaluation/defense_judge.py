"""Blinded judge adaptation for Run 9 defense outputs.

The defense runners emit either paired SARSteer generations or two aligned
ALMGuard arms.  This module normalizes both layouts to the existing
``judge_records`` contract, gives every request a content-addressed identity for
safe resume, and converts unanimous judge verdicts into the four behavior labels
accepted by :mod:`audio_safety.evaluation.defense_gate`.

No function in this module performs network IO on import.  Checkpoints and label
sidecars deliberately omit request and response bodies; only the in-memory rows
passed to ``judge_records`` contain those bodies.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.config.schema import JudgeConfig
from audio_safety.evaluation.defense_gate import VALID_BEHAVIOR_LABELS
from audio_safety.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    behavior_label_from_verdict,
    judge_records,
)

SCHEMA_VERSION = "run9-defense-judge-v1"
ARMS = ("undefended", "defended")


@dataclass(frozen=True)
class LabelResolution:
    """One resolved four-way behavior label and its audit metadata."""

    behavior_label: str
    resolution: str
    per_judge_behavior_label: dict[str, str]
    reviewed_by: str | None = None


JudgeRunner = Callable[..., Iterator[dict[str, Any]]]


def _require_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{context} is missing non-empty {field!r}")
    return str(value).strip()


def condition_of(row: Mapping[str, Any]) -> str:
    """Return the authoritative Run 9 condition (legacy ``style`` fallback)."""

    value = row.get("condition")
    if value is None:
        value = row.get("style")
    if value is None or not str(value).strip():
        raise ValueError("row needs a non-empty 'condition' (or legacy 'style')")
    return str(value).strip()


def alignment_key(row: Mapping[str, Any]) -> str:
    """Mirror the defense-gate evaluator's stable alignment precedence."""

    if row.get("record_id") is not None:
        return f"record_id:{_require_text(row, 'record_id', context='row')}"
    if row.get("index") is not None:
        return f"index:{row['index']}"
    item_id = _require_text(row, "item_id", context="row")
    safety_label = _require_text(row, "safety_label", context=f"row {item_id!r}")
    condition = condition_of(row)
    sign = "" if row.get("sign") is None else str(row["sign"])
    path = "" if row.get("path") is None else str(row["path"])
    return "metadata:" + "|".join((item_id, safety_label, condition, sign, path))


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_common_row(row: Mapping[str, Any], *, context: str) -> None:
    item_id = _require_text(row, "item_id", context=context)
    safety_label = _require_text(row, "safety_label", context=f"{context} {item_id!r}")
    if safety_label not in {"harmful", "benign"}:
        raise ValueError(f"{context} {item_id!r} has unsupported safety_label {safety_label!r}")
    condition_of(row)
    _require_text(row, "reference_text", context=f"{context} {item_id!r}")


def _expanded_record(
    row: Mapping[str, Any],
    *,
    arm: str,
    response: Any,
    layout: str,
    row_alignment_key: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown defense arm {arm!r}")
    if not isinstance(response, str):
        raise ValueError(
            f"{layout} {arm} output for alignment key {row_alignment_key!r} must be a string"
        )
    updated = dict(row)
    for field in (
        "undefended_output",
        "defended_output",
        "judge_verdicts",
        "judge_attack_success",
        "behavior_label",
        "reviewed_behavior_label",
        "undefended_behavior_label",
        "defended_behavior_label",
    ):
        updated.pop(field, None)
    updated.update(
        {
            "gate_input_layout": layout,
            "gate_alignment_key": row_alignment_key,
            "defense_arm": arm,
            "judge_request_text": str(row["reference_text"]),
            "output": response,
        }
    )
    return updated


def expand_paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand each SARSteer pair into blinded undefended and defended records."""

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _validate_common_row(row, context=f"paired row {index}")
        key = alignment_key(row)
        if key in seen:
            raise ValueError(f"duplicate paired alignment key {key!r}")
        seen.add(key)
        for arm in ARMS:
            output_field = f"{arm}_output"
            if output_field not in row:
                raise ValueError(f"paired row {key!r} is missing {output_field!r}")
            expanded.append(
                _expanded_record(
                    row,
                    arm=arm,
                    response=row[output_field],
                    layout="paired",
                    row_alignment_key=key,
                )
            )
    return expanded


def _index_arm(
    rows: Sequence[Mapping[str, Any]], *, arm: str
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    order: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        _validate_common_row(row, context=f"{arm} row {index}")
        key = alignment_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {arm} alignment key {key!r}")
        order.append(key)
        indexed[key] = row
    return order, indexed


def expand_aligned_rows(
    undefended_rows: Sequence[Mapping[str, Any]],
    defended_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand separately stored ALMGuard arms after strict metadata alignment."""

    order, undefended = _index_arm(undefended_rows, arm="undefended")
    _, defended = _index_arm(defended_rows, arm="defended")
    if set(undefended) != set(defended):
        missing = sorted(set(undefended) - set(defended))[:3]
        extra = sorted(set(defended) - set(undefended))[:3]
        raise ValueError(
            f"ALMGuard arms do not align (missing defended={missing}, extra defended={extra})"
        )

    expanded: list[dict[str, Any]] = []
    for key in order:
        urow = undefended[key]
        drow = defended[key]
        u_meta = (
            str(urow["item_id"]).strip(),
            str(urow["safety_label"]).strip(),
            condition_of(urow),
            str(urow["reference_text"]),
        )
        d_meta = (
            str(drow["item_id"]).strip(),
            str(drow["safety_label"]).strip(),
            condition_of(drow),
            str(drow["reference_text"]),
        )
        if u_meta != d_meta:
            raise ValueError(f"aligned key {key!r} has conflicting arm metadata")
        for arm, row in (("undefended", urow), ("defended", drow)):
            if "output" not in row:
                raise ValueError(f"{arm} row {key!r} is missing 'output'")
            expanded.append(
                _expanded_record(
                    row,
                    arm=arm,
                    response=row["output"],
                    layout="aligned",
                    row_alignment_key=key,
                )
            )
    return expanded


def judge_spec_fingerprint(cfg: JudgeConfig) -> str:
    """Fingerprint semantic judge settings so changed judges cannot reuse cache."""

    return _canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "models": list(cfg.models),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "mapping": "behavior_label_from_verdict-v1-unanimous",
        }
    )


def bind_judge_identities(
    records: Sequence[Mapping[str, Any]], cfg: JudgeConfig
) -> list[dict[str, Any]]:
    """Attach content-addressed IDs covering input bodies and judge semantics."""

    if len(cfg.models) < 2:
        raise ValueError("Run 9 defense adjudication requires at least two judge models")
    if len(cfg.models) != len(set(cfg.models)):
        raise ValueError("judge model ids must be unique")
    spec = judge_spec_fingerprint(cfg)
    bound: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        input_fingerprint = _canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "layout": row["gate_input_layout"],
                "alignment_key": row["gate_alignment_key"],
                "arm": row["defense_arm"],
                "safety_label": row["safety_label"],
                "request": row["judge_request_text"],
                "response": row["output"],
            }
        )
        judge_record_id = "defense-judge:" + _canonical_digest(
            {"input_fingerprint": input_fingerprint, "judge_spec_fingerprint": spec}
        )
        if judge_record_id in seen:
            raise ValueError(f"duplicate bound judge record id {judge_record_id!r}")
        seen.add(judge_record_id)
        updated = dict(row)
        updated.update(
            {
                "defense_judge_schema": SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "judge_spec_fingerprint": spec,
                "judge_record_id": judge_record_id,
            }
        )
        bound.append(updated)
    return bound


def _checkpoint_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep audit information while excluding request and response bodies."""

    return {
        "defense_judge_schema": SCHEMA_VERSION,
        "judge_record_id": row["judge_record_id"],
        "input_fingerprint": row["input_fingerprint"],
        "judge_spec_fingerprint": row["judge_spec_fingerprint"],
        "gate_input_layout": row["gate_input_layout"],
        "gate_alignment_key": row["gate_alignment_key"],
        "defense_arm": row["defense_arm"],
        "item_id": row["item_id"],
        "safety_label": row["safety_label"],
        "condition": condition_of(row),
        "judge_verdicts": row["judge_verdicts"],
        "judge_attack_success": row.get("judge_attack_success", {}),
    }


def atomic_save_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically replace ``path`` with a flushed same-directory JSONL temp file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    for field in (
        "input_fingerprint",
        "judge_spec_fingerprint",
        "gate_input_layout",
        "gate_alignment_key",
        "defense_arm",
    ):
        if row.get(field) != expected.get(field):
            raise ValueError(
                f"checkpoint field {field!r} does not match current input for "
                f"{expected['judge_record_id']}"
            )
    verdicts = row.get("judge_verdicts")
    if not isinstance(verdicts, Mapping) or set(verdicts) != set(models):
        raise ValueError(
            f"checkpoint {expected['judge_record_id']} does not contain exactly "
            "the configured judge verdicts"
        )
    safety_label = str(expected["safety_label"])
    for model in models:
        verdict = verdicts[model]
        if not isinstance(verdict, Mapping):
            raise ValueError(f"checkpoint verdict for model {model!r} is not an object")
        behavior_label_from_verdict(dict(verdict), safety_label=safety_label)


def load_completed_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_records: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate a resumable checkpoint as a subset of the current bound inputs."""

    expected = {str(row["judge_record_id"]): row for row in expected_records}
    if len(expected) != len(expected_records):
        raise ValueError("current judge inputs contain duplicate judge_record_id values")
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("judge_record_id") or "")
        if not record_id:
            raise ValueError("checkpoint row is missing judge_record_id")
        if record_id in completed:
            raise ValueError(f"duplicate checkpoint judge_record_id {record_id!r}")
        if record_id not in expected:
            raise ValueError(
                f"checkpoint contains a record absent from current inputs: {record_id}"
            )
        _validate_checkpoint_row(row, expected=expected[record_id], models=models)
        completed[record_id] = dict(row)
    return completed


def run_judge_checkpoint(
    records: Sequence[Mapping[str, Any]],
    cfg: JudgeConfig,
    *,
    checkpoint_path: Path,
    save_every: int = 16,
    overwrite: bool = False,
    api_key: str | None = None,
    show_progress: bool = True,
    judge_runner: JudgeRunner = judge_records,
) -> list[dict[str, Any]]:
    """Judge pending records and atomically checkpoint in deterministic input order."""

    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if len(cfg.models) < 2:
        raise ValueError("Run 9 defense adjudication requires at least two judge models")
    expected = {str(row["judge_record_id"]): row for row in records}
    order = [str(row["judge_record_id"]) for row in records]
    if len(expected) != len(records):
        raise ValueError("judge inputs contain duplicate judge_record_id values")

    if overwrite:
        atomic_save_jsonl([], checkpoint_path)
        completed: dict[str, dict[str, Any]] = {}
    elif checkpoint_path.exists():
        checkpoint_rows = [
            json.loads(line)
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed = load_completed_checkpoint(
            checkpoint_rows,
            expected_records=records,
            models=cfg.models,
        )
    else:
        completed = {}

    pending = [dict(row) for row in records if str(row["judge_record_id"]) not in completed]
    since_save = 0
    for judged in judge_runner(
        pending,
        cfg,
        request_field="judge_request_text",
        response_field="output",
        api_key=api_key,
        show_progress=show_progress,
    ):
        record_id = str(judged.get("judge_record_id") or "")
        if record_id not in expected:
            raise ValueError("judge runner returned a row absent from current inputs")
        projected = _checkpoint_projection(judged)
        _validate_checkpoint_row(projected, expected=expected[record_id], models=cfg.models)
        completed[record_id] = projected
        since_save += 1
        if since_save >= save_every:
            atomic_save_jsonl(
                (completed[key] for key in order if key in completed), checkpoint_path
            )
            since_save = 0

    if set(completed) != set(order):
        missing_count = len(set(order) - set(completed))
        raise RuntimeError(f"judge runner completed without returning {missing_count} records")
    ordered = [completed[key] for key in order]
    atomic_save_jsonl(ordered, checkpoint_path)
    return ordered


def _checked_behavior_label(value: Any, *, context: str) -> str:
    label = str(value).strip()
    if label not in VALID_BEHAVIOR_LABELS:
        raise ValueError(
            f"{context} has invalid behavior label {label!r}; "
            f"expected one of {sorted(VALID_BEHAVIOR_LABELS)}"
        )
    return label


def _index_overrides(
    rows: Sequence[Mapping[str, Any]], *, valid_record_ids: set[str]
) -> dict[str, Mapping[str, Any]]:
    overrides: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = _require_text(row, "judge_record_id", context="reviewed override")
        if record_id not in valid_record_ids:
            raise ValueError(f"reviewed override targets an unknown judge_record_id: {record_id}")
        if record_id in overrides:
            raise ValueError(f"duplicate reviewed override for {record_id}")
        _checked_behavior_label(row.get("reviewed_behavior_label"), context=record_id)
        _require_text(row, "reviewed_by", context=f"reviewed override {record_id}")
        overrides[record_id] = row
    return overrides


def resolve_checkpoint_labels(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    reviewed_overrides: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, LabelResolution], list[dict[str, Any]]]:
    """Resolve unanimous labels; disagreements require an explicit reviewed override.

    The unresolved report contains only IDs and categorical labels, never request
    or response bodies.  Callers must not emit evaluator sidecars while this list
    is non-empty.
    """

    valid_ids = {str(row["judge_record_id"]) for row in checkpoint_rows}
    overrides = _index_overrides(reviewed_overrides, valid_record_ids=valid_ids)
    resolved: dict[str, LabelResolution] = {}
    unresolved: list[dict[str, Any]] = []
    for row in checkpoint_rows:
        record_id = str(row["judge_record_id"])
        verdicts = row.get("judge_verdicts")
        if not isinstance(verdicts, Mapping) or len(verdicts) < 2:
            raise ValueError(f"checkpoint {record_id} needs at least two judge verdicts")
        safety_label = str(row["safety_label"])
        per_judge = {
            str(model): behavior_label_from_verdict(dict(verdict), safety_label=safety_label)
            for model, verdict in verdicts.items()
        }
        unanimous = len(set(per_judge.values())) == 1
        override = overrides.get(record_id)
        if unanimous:
            if override is not None:
                raise ValueError(
                    f"reviewed override {record_id} is unnecessary because judges are unanimous"
                )
            label = next(iter(per_judge.values()))
            resolved[record_id] = LabelResolution(
                behavior_label=label,
                resolution="unanimous_judges",
                per_judge_behavior_label=per_judge,
            )
            continue
        if override is not None:
            label = _checked_behavior_label(override["reviewed_behavior_label"], context=record_id)
            resolved[record_id] = LabelResolution(
                behavior_label=label,
                resolution="reviewed_override",
                per_judge_behavior_label=per_judge,
                reviewed_by=str(override["reviewed_by"]).strip(),
            )
            continue
        unresolved.append(
            {
                "defense_judge_schema": SCHEMA_VERSION,
                "judge_record_id": record_id,
                "gate_alignment_key": row["gate_alignment_key"],
                "defense_arm": row["defense_arm"],
                "item_id": row["item_id"],
                "safety_label": safety_label,
                "condition": row["condition"],
                "per_judge_behavior_label": per_judge,
                "reason": "judge_disagreement_requires_review",
            }
        )
    return resolved, unresolved


_SIDECAR_DROP_FIELDS = {
    "undefended_output",
    "defended_output",
    "output",
    "judge_request_text",
    "reference_text",
    "prompt",
    "request_text",
    "harmful_text",
    "benign_text",
    "transcript",
    "judge_verdicts",
    "judge_attack_success",
    "behavior_label",
    "reviewed_behavior_label",
    "undefended_behavior_label",
    "defended_behavior_label",
}


def _sidecar_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _SIDECAR_DROP_FIELDS
        and not key.startswith("judge_")
        and not key.startswith("reviewed_")
    }


def _record_by_alignment_arm(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row["gate_alignment_key"]), str(row["defense_arm"]))
        if key in indexed:
            raise ValueError(f"duplicate judge record for alignment/arm {key!r}")
        indexed[key] = row
    return indexed


def build_paired_label_sidecar(
    raw_rows: Sequence[Mapping[str, Any]],
    bound_records: Sequence[Mapping[str, Any]],
    resolutions: Mapping[str, LabelResolution],
) -> list[dict[str, Any]]:
    """Build the one-file sidecar consumed by ``--paired-labels``."""

    indexed = _record_by_alignment_arm(bound_records)
    sidecar: list[dict[str, Any]] = []
    for row in raw_rows:
        key = alignment_key(row)
        record = dict(_sidecar_metadata(row))
        for arm in ARMS:
            bound = indexed[(key, arm)]
            judge_record_id = str(bound["judge_record_id"])
            if judge_record_id not in resolutions:
                raise ValueError(f"missing label resolution for {judge_record_id}")
            resolution = resolutions[judge_record_id]
            record[f"{arm}_behavior_label"] = resolution.behavior_label
            record[f"{arm}_label_resolution"] = resolution.resolution
            record[f"{arm}_judge_record_id"] = judge_record_id
            record[f"{arm}_per_judge_behavior_label"] = resolution.per_judge_behavior_label
            if resolution.reviewed_by is not None:
                record[f"{arm}_reviewed_by"] = resolution.reviewed_by
        sidecar.append(record)
    return sidecar


def build_aligned_label_sidecars(
    undefended_rows: Sequence[Mapping[str, Any]],
    defended_rows: Sequence[Mapping[str, Any]],
    bound_records: Sequence[Mapping[str, Any]],
    resolutions: Mapping[str, LabelResolution],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the two arm sidecars consumed by aligned gate evaluation."""

    indexed = _record_by_alignment_arm(bound_records)

    def build(rows: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            key = alignment_key(row)
            bound = indexed[(key, arm)]
            judge_record_id = str(bound["judge_record_id"])
            if judge_record_id not in resolutions:
                raise ValueError(f"missing label resolution for {judge_record_id}")
            resolution = resolutions[judge_record_id]
            record = dict(_sidecar_metadata(row))
            record.update(
                {
                    "behavior_label": resolution.behavior_label,
                    "label_resolution": resolution.resolution,
                    "judge_record_id": judge_record_id,
                    "per_judge_behavior_label": resolution.per_judge_behavior_label,
                }
            )
            if resolution.reviewed_by is not None:
                record["reviewed_by"] = resolution.reviewed_by
            out.append(record)
        return out

    return build(undefended_rows, "undefended"), build(defended_rows, "defended")
