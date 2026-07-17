"""Fail-closed validation and deterministic merge for sharded SARSteer outputs.

The apply path resumes by the semantic key
``(item_id, safety_label, condition, sign)``.  This module treats the canonical
evaluation manifest as the ordering authority, verifies every shard output
against its own shard manifest, and only then emits one atomic JSONL artifact.
It deliberately never prints generation bodies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl, stable_row_key

StableRowKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class ShardPair:
    """One input shard manifest and its completed SARSteer output."""

    manifest: Path
    output: Path


@dataclass(frozen=True)
class MergeResult:
    """Validated rows in canonical input order plus non-sensitive counts."""

    rows: list[dict[str, Any]]
    canonical_count: int
    shard_counts: tuple[int, ...]


def _load_object_jsonl(path: Path, *, role: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{role} not found: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{role} line {line_number} is not a JSON object"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {role} {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{role} is empty: {path}")
    return rows


def _key(row: Mapping[str, Any], *, role: str, index: int) -> StableRowKey:
    try:
        return stable_row_key(row)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid stable key in {role} row {index}: {exc}") from exc


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, role: str
) -> tuple[list[StableRowKey], dict[StableRowKey, Mapping[str, Any]]]:
    keys: list[StableRowKey] = []
    indexed: dict[StableRowKey, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = _key(row, role=role, index=index)
        if key in indexed:
            raise ValueError(f"duplicate stable key in {role}: {key}")
        keys.append(key)
        indexed[key] = row
    return keys, indexed


def _validate_completed_output(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    role: str,
    index: int,
) -> None:
    for field in ("undefended_output", "defended_output"):
        if not isinstance(row.get(field), str):
            raise ValueError(f"{role} row {index} lacks completed string field '{field}'")
    if row.get("defense") != "sarsteer":
        raise ValueError(f"{role} row {index} must have defense='sarsteer'")

    expected_record_id = input_row.get("record_id")
    if expected_record_id is not None and row.get("record_id") != expected_record_id:
        raise ValueError(f"{role} row {index} record_id differs from its input row")

    _validate_preserved_metadata(row, input_row, role=role, index=index)


def _validate_preserved_metadata(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    role: str,
    index: int,
) -> None:
    """Require every input field to survive unchanged in a derived row.

    SARSteer apply outputs add generation and implementation fields, but they
    must not rewrite evaluation metadata. Stable-key checks alone cannot catch
    drift in fields such as ``gate_role``, ``path``, or ``reference_text``.
    """

    for field, expected in input_row.items():
        if field not in row:
            raise ValueError(f"{role} row {index} dropped input metadata field {field!r}")
        if row[field] != expected:
            raise ValueError(
                f"{role} row {index} changed input metadata field {field!r}: "
                f"{row[field]!r} != {expected!r}"
            )


def validate_and_merge(
    canonical_manifest: Path,
    shards: Sequence[ShardPair],
) -> MergeResult:
    """Validate shard coverage/completion/order and return canonical-order rows.

    Fail-closed checks cover duplicate and missing stable keys, foreign keys,
    shard-output order drift, incomplete generation pairs, and record-id drift.
    """

    if not shards:
        raise ValueError("at least one shard pair is required")

    canonical_rows = _load_object_jsonl(canonical_manifest, role="canonical manifest")
    canonical_keys, canonical_by_key = _index_unique(
        canonical_rows, role="canonical manifest"
    )
    canonical_key_set = set(canonical_keys)

    seen_manifest_keys: set[StableRowKey] = set()
    seen_output_keys: set[StableRowKey] = set()
    merged_by_key: dict[StableRowKey, dict[str, Any]] = {}
    shard_counts: list[int] = []

    for shard_index, shard in enumerate(shards):
        manifest_role = f"shard {shard_index} manifest"
        output_role = f"shard {shard_index} output"
        manifest_rows = _load_object_jsonl(shard.manifest, role=manifest_role)
        output_rows = _load_object_jsonl(shard.output, role=output_role)
        manifest_keys, manifest_by_key = _index_unique(
            manifest_rows, role=manifest_role
        )
        output_keys, _ = _index_unique(output_rows, role=output_role)

        foreign_manifest = set(manifest_keys) - canonical_key_set
        if foreign_manifest:
            first = next(iter(foreign_manifest))
            raise ValueError(
                f"{manifest_role} contains a key absent from the canonical manifest: {first}"
            )
        overlap = seen_manifest_keys.intersection(manifest_keys)
        if overlap:
            raise ValueError(
                f"stable key assigned to multiple shard manifests: {next(iter(overlap))}"
            )
        seen_manifest_keys.update(manifest_keys)

        for row_index, key in enumerate(manifest_keys):
            _validate_preserved_metadata(
                manifest_by_key[key],
                canonical_by_key[key],
                role=manifest_role,
                index=row_index,
            )

        if len(output_rows) != len(manifest_rows):
            raise ValueError(
                f"{output_role} row count {len(output_rows)} does not match "
                f"{manifest_role} row count {len(manifest_rows)}"
            )
        if output_keys != manifest_keys:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(output_keys, manifest_keys, strict=True)
                    )
                    if actual != expected
                ),
                None,
            )
            raise ValueError(
                f"{output_role} stable-key order differs from its input manifest "
                f"at row {mismatch}"
            )

        output_overlap = seen_output_keys.intersection(output_keys)
        if output_overlap:
            raise ValueError(
                f"stable key appears in multiple shard outputs: {next(iter(output_overlap))}"
            )
        seen_output_keys.update(output_keys)

        for row_index, (key, output_row) in enumerate(zip(output_keys, output_rows, strict=True)):
            _validate_completed_output(
                output_row,
                manifest_by_key[key],
                role=output_role,
                index=row_index,
            )
            merged_by_key[key] = dict(output_row)
        shard_counts.append(len(output_rows))

    missing_manifest = canonical_key_set - seen_manifest_keys
    if missing_manifest:
        raise ValueError(
            "shard manifests do not cover the canonical manifest; "
            f"missing {len(missing_manifest)} stable keys"
        )
    missing_output = canonical_key_set - seen_output_keys
    if missing_output:
        raise ValueError(
            f"shard outputs are missing {len(missing_output)} canonical stable keys"
        )
    if seen_manifest_keys != canonical_key_set or seen_output_keys != canonical_key_set:
        raise AssertionError("validated key sets unexpectedly differ")

    merged_rows = [merged_by_key[key] for key in canonical_keys]
    for index, (canonical_row, merged_row) in enumerate(
        zip(canonical_rows, merged_rows, strict=True)
    ):
        key = canonical_keys[index]
        if _key(merged_row, role="merged output", index=index) != key:
            raise AssertionError("canonical ordering unexpectedly changed during merge")
        expected_record_id = canonical_row.get("record_id")
        if expected_record_id is not None and merged_row.get("record_id") != expected_record_id:
            raise ValueError(f"merged output row {index} record_id differs from canonical input")

    return MergeResult(
        rows=merged_rows,
        canonical_count=len(canonical_rows),
        shard_counts=tuple(shard_counts),
    )


def write_merged(result: MergeResult, output: Path) -> None:
    """Atomically write a result already validated by :func:`validate_and_merge`."""

    atomic_save_jsonl(result.rows, output)
