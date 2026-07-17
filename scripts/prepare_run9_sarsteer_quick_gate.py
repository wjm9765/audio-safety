#!/usr/bin/env -S uv run python
"""Select the deterministic 165-row SARSteer directional quick gate.

This is a strict subset of the adapted held-out final300 manifest.  It is a
time-bounded directional check for the already-built vector at alpha=0.03, not
a replacement for the full advisor gate or an official SARSteer reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl, shard_by_item
from audio_safety.utils.io import load_jsonl, save_json

HARMFUL_CATEGORIES = tuple(range(1, 8))
SOFT_CATEGORIES = tuple(range(8, 11))
HARMFUL_PER_CATEGORY = 5
SOFT_PER_CATEGORY = 5
UTILITY_ROWS = 15
POSITIVE_CONTROL_ROWS = 30
QUICK_ROWS = 165
SOURCE_FINAL_ROWS = 300
SOURCE_DEV_ROWS = 76
SOURCE_CORE_ROWS = 300
EXPECTED_ROLE_COUNTS = {
    "harmful_eval": 105,
    "soft_overrefusal": 15,
    "utility_eval": 15,
    "positive_control_eval": 30,
}
EXPECTED_CONDITION_COUNTS = {
    "clean": 65,
    "pv_standard": 70,
    "positive_control": 30,
}


@dataclass(frozen=True)
class QuickGatePlan:
    rows: list[dict[str, Any]]
    shards: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    summary: dict[str, Any]


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty {field!r}")
    return value.strip()


def _category(row: Mapping[str, Any], *, context: str) -> int:
    value = row.get("category_id")
    if isinstance(value, bool):
        raise ValueError(f"{context} has invalid category_id={value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid category_id={value!r}") from exc


def _sign(value: Any, *, context: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} has invalid boolean sign")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid sign={value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} has non-finite sign={value!r}")
    if number == -3.0:
        return -3
    if number == 3.0:
        return 3
    return None


def _resolved_path(row: Mapping[str, Any], data_dir: Path, *, context: str) -> Path:
    raw = _required_text(row, "path", context=context)
    path = Path(raw)
    return (path if path.is_absolute() else data_dir / path).resolve()


def _normalized_text(row: Mapping[str, Any], *, context: str) -> str:
    return " ".join(_required_text(row, "reference_text", context=context).split()).casefold()


def _assets(
    rows: Sequence[Mapping[str, Any]], *, name: str, data_dir: Path
) -> tuple[set[str], set[Path], set[str]]:
    items: set[str] = set()
    paths: set[Path] = set()
    texts: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{name} row {index}"
        items.add(_required_text(row, "item_id", context=context))
        paths.add(_resolved_path(row, data_dir, context=context))
        texts.add(_normalized_text(row, context=context))
    return items, paths, texts


def _assert_disjoint(
    quick_rows: Sequence[Mapping[str, Any]],
    other_rows: Sequence[Mapping[str, Any]],
    *,
    other_name: str,
    data_dir: Path,
) -> dict[str, Any]:
    quick_assets = _assets(quick_rows, name="quick165", data_dir=data_dir)
    other_assets = _assets(other_rows, name=other_name, data_dir=data_dir)
    labels = ("item_id", "path", "reference_text")
    overlaps = [quick_assets[index] & other_assets[index] for index in range(3)]
    if any(overlaps):
        details = []
        for label, overlap in zip(labels, overlaps, strict=True):
            if overlap:
                values = sorted(str(value) for value in overlap)[:10]
                details.append(f"{label}={values}")
        raise ValueError(f"quick165/{other_name} leakage detected: " + "; ".join(details))
    return {"other": other_name, "checked": list(labels), "passed": True}


def _validate_source_final(rows: Sequence[Mapping[str, Any]], *, data_dir: Path) -> None:
    if len(rows) != SOURCE_FINAL_ROWS:
        raise ValueError(f"source final manifest needs 300 rows, found {len(rows)}")
    role_counts = Counter(str(row.get("gate_role")) for row in rows)
    expected_roles = {
        "harmful_eval": 210,
        "soft_overrefusal": 30,
        "utility_eval": 30,
        "positive_control_eval": 30,
    }
    if role_counts != Counter(expected_roles):
        raise ValueError(f"source final role counts mismatch: {dict(role_counts)}")
    condition_counts = Counter(str(row.get("condition")) for row in rows)
    expected_conditions = {"clean": 130, "pv_standard": 140, "positive_control": 30}
    if condition_counts != Counter(expected_conditions):
        raise ValueError(f"source final condition counts mismatch: {dict(condition_counts)}")

    record_ids: set[str] = set()
    paths: set[Path] = set()
    for index, row in enumerate(rows):
        context = f"source final row {index}"
        record_id = _required_text(row, "record_id", context=context)
        if record_id in record_ids:
            raise ValueError(f"source final has duplicate record_id={record_id!r}")
        record_ids.add(record_id)
        path = _resolved_path(row, data_dir, context=context)
        if not path.is_file():
            raise ValueError(f"{context} audio file not found: {path}")
        if path in paths:
            raise ValueError(f"source final reuses exact audio path: {path}")
        paths.add(path)
        _required_text(row, "item_id", context=context)
        _required_text(row, "safety_label", context=context)
        _normalized_text(row, context=context)
        if row.get("final_gate_eligible") is not True:
            raise ValueError(f"{context} is not final_gate_eligible=true")


def _select_harmful_indices(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[int], dict[int, tuple[str, ...]]]:
    cells: dict[tuple[int, str, str, int | None], int] = {}
    order: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if str(row.get("gate_role")) != "harmful_eval":
            continue
        category = _category(row, context=f"harmful row {index}")
        if category not in HARMFUL_CATEGORIES:
            continue
        condition = str(row.get("condition"))
        sign = _sign(row.get("sign"), context=f"harmful row {index}")
        if condition == "clean" and sign is None:
            cell_sign = None
        elif condition == "pv_standard" and sign in {-3, 3}:
            cell_sign = sign
        else:
            continue
        item_id = _required_text(row, "item_id", context=f"harmful row {index}")
        key = (category, item_id, condition, cell_sign)
        if key in cells:
            raise ValueError(f"duplicate harmful source cell={key}")
        cells[key] = index
        if item_id not in seen[category]:
            seen[category].add(item_id)
            order[category].append(item_id)

    selected: set[int] = set()
    item_ids_by_category: dict[int, tuple[str, ...]] = {}
    required = (("clean", None), ("pv_standard", -3), ("pv_standard", 3))
    for category in HARMFUL_CATEGORIES:
        complete = []
        for item_id in order[category]:
            if all((category, item_id, condition, sign) in cells for condition, sign in required):
                complete.append(item_id)
            if len(complete) == HARMFUL_PER_CATEGORY:
                break
        if len(complete) != HARMFUL_PER_CATEGORY:
            raise ValueError(
                f"category {category} needs {HARMFUL_PER_CATEGORY} complete harmful items, "
                f"found {len(complete)}"
            )
        item_ids_by_category[category] = tuple(complete)
        for item_id in complete:
            for condition, sign in required:
                selected.add(cells[(category, item_id, condition, sign)])
    return selected, item_ids_by_category


def _select_singleton_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    count: int,
    category: int | None = None,
) -> tuple[set[int], tuple[str, ...]]:
    selected: set[int] = set()
    item_ids = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("gate_role")) != role or str(row.get("condition")) != "clean":
            continue
        if category is not None and _category(row, context=f"{role} row {index}") != category:
            continue
        item_id = _required_text(row, "item_id", context=f"{role} row {index}")
        if item_id in seen:
            raise ValueError(f"duplicate {role} item_id={item_id!r}")
        seen.add(item_id)
        selected.add(index)
        item_ids.append(item_id)
        if len(item_ids) == count:
            break
    if len(item_ids) != count:
        label = f" category {category}" if category is not None else ""
        raise ValueError(f"{role}{label} needs {count} rows, found {len(item_ids)}")
    return selected, tuple(item_ids)


def select_quick_gate(
    final_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    core_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
) -> QuickGatePlan:
    """Select and audit the frozen quick165 subset."""

    data_dir = data_dir.resolve()
    _validate_source_final(final_rows, data_dir=data_dir)
    if len(dev_rows) != SOURCE_DEV_ROWS:
        raise ValueError(f"source dev manifest needs 76 rows, found {len(dev_rows)}")
    if len(core_rows) != SOURCE_CORE_ROWS:
        raise ValueError(f"source core manifest needs 300 rows, found {len(core_rows)}")

    selected, harmful_ids = _select_harmful_indices(final_rows)
    soft_ids: dict[int, tuple[str, ...]] = {}
    for category in SOFT_CATEGORIES:
        indices, ids = _select_singleton_indices(
            final_rows,
            role="soft_overrefusal",
            category=category,
            count=SOFT_PER_CATEGORY,
        )
        selected.update(indices)
        soft_ids[category] = ids
    utility_indices, utility_ids = _select_singleton_indices(
        final_rows,
        role="utility_eval",
        count=UTILITY_ROWS,
    )
    selected.update(utility_indices)

    positive_indices = []
    positive_ids = []
    for index, row in enumerate(final_rows):
        if str(row.get("gate_role")) != "positive_control_eval":
            continue
        if str(row.get("condition")) != "positive_control":
            raise ValueError(f"positive-control row {index} has wrong condition")
        if str(row.get("attack_family")) != "jb_pap":
            raise ValueError(f"positive-control row {index} is not frozen jb_pap")
        if row.get("historical_outcomes_used_for_selection") is not False:
            raise ValueError(f"positive-control row {index} lacks outcome-agnostic marker")
        positive_indices.append(index)
        positive_ids.append(_required_text(row, "item_id", context=f"PC row {index}"))
    if len(positive_indices) != POSITIVE_CONTROL_ROWS:
        raise ValueError(
            f"quick gate requires all {POSITIVE_CONTROL_ROWS} frozen PAP controls, "
            f"found {len(positive_indices)}"
        )
    selected.update(positive_indices)

    quick_rows = [dict(row) for index, row in enumerate(final_rows) if index in selected]
    if len(quick_rows) != QUICK_ROWS:
        raise ValueError(f"quick gate needs {QUICK_ROWS} rows, found {len(quick_rows)}")
    role_counts = Counter(str(row["gate_role"]) for row in quick_rows)
    condition_counts = Counter(str(row["condition"]) for row in quick_rows)
    if role_counts != Counter(EXPECTED_ROLE_COUNTS):
        raise ValueError(f"quick role counts mismatch: {dict(role_counts)}")
    if condition_counts != Counter(EXPECTED_CONDITION_COUNTS):
        raise ValueError(f"quick condition counts mismatch: {dict(condition_counts)}")

    source_by_record = {
        _required_text(row, "record_id", context="source final"): dict(row) for row in final_rows
    }
    for row in quick_rows:
        record_id = _required_text(row, "record_id", context="quick row")
        if source_by_record.get(record_id) != row:
            raise AssertionError(f"quick row is not an exact final300 subset row: {record_id}")

    leakage = [
        _assert_disjoint(quick_rows, dev_rows, other_name="alpha dev76", data_dir=data_dir),
        _assert_disjoint(quick_rows, core_rows, other_name="legacy core300", data_dir=data_dir),
    ]
    shards_raw = shard_by_item(quick_rows, 2)
    shards = (shards_raw[0], shards_raw[1])
    shard_counts = [len(shard) for shard in shards]
    if shard_counts != [83, 82]:
        raise ValueError(f"expected deterministic shard row counts [83, 82], got {shard_counts}")

    summary = {
        "schema_version": "run9-sarsteer-adapted-quick165-v1",
        "status": "READY",
        "experiment_contract": {
            "defense": "sarsteer",
            "alpha": 0.03,
            "vectors": "reuse_existing_local_vectors",
            "scope": "time_bounded_directional_gate",
            "not_official_reproduction": True,
        },
        "rows": len(quick_rows),
        "unique_item_ids": len({str(row["item_id"]) for row in quick_rows}),
        "role_counts": dict(sorted(role_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "selection": {
            "harmful_item_ids_by_category": {
                str(category): list(ids) for category, ids in harmful_ids.items()
            },
            "soft_item_ids_by_category": {
                str(category): list(ids) for category, ids in soft_ids.items()
            },
            "utility_item_ids": list(utility_ids),
            "positive_control_item_ids": positive_ids,
            "positive_control_rule": "all 30 frozen outcome-agnostic jb_pap rows",
        },
        "subset_check": {
            "source_final_rows": len(final_rows),
            "selected_record_ids_subset": True,
            "selected_rows_byte_semantically_unchanged": True,
        },
        "leakage": {
            "dimensions": ["item_id", "path", "reference_text"],
            "checks": leakage,
            "all_passed": True,
        },
        "shards": {
            "method": "item-grouped deterministic greedy balance",
            "row_counts": shard_counts,
            "item_counts": [len({str(row["item_id"]) for row in shard}) for shard in shards],
            "same_item_kept_together": True,
        },
    }
    return QuickGatePlan(rows=quick_rows, shards=shards, summary=summary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_paths(output: Path) -> tuple[Path, Path]:
    return tuple(
        output.with_name(f"{output.stem}.shard{index:02d}-of-02.jsonl") for index in range(2)
    )  # type: ignore[return-value]


def materialize_quick_gate(
    plan: QuickGatePlan,
    *,
    output: Path,
    summary_output: Path,
    input_paths: Mapping[str, Path],
    dry_run: bool = False,
) -> dict[str, Any]:
    shard_paths = _shard_paths(output)
    summary = {
        **plan.summary,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": {
            "manifest": str(output.resolve()),
            "shards": [str(path.resolve()) for path in shard_paths],
            "summary": str(summary_output.resolve()),
        },
    }
    if dry_run:
        return summary

    atomic_save_jsonl(plan.rows, output)
    for shard, path in zip(plan.shards, shard_paths, strict=True):
        atomic_save_jsonl(shard, path)
    summary["outputs"]["sha256"] = {
        "manifest": _sha256(output),
        "shards": [_sha256(path) for path in shard_paths],
    }
    save_json(summary, summary_output)
    summary_sha256 = _sha256(summary_output)
    sidecar = summary_output.with_suffix(summary_output.suffix + ".sha256")
    sidecar.write_text(f"{summary_sha256}  {summary_output.name}\n")
    return {
        **summary,
        "summary_sha256": summary_sha256,
        "summary_sha256_sidecar": str(sidecar.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_input(path: Path, data_dir: Path, *, role: str) -> Path:
    resolved = (path if path.is_absolute() else data_dir / path).resolve()
    if not resolved.is_file():
        raise SystemExit(f"{role} not found: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")
    final_path = _resolve_input(args.final_manifest, data_dir, role="final manifest")
    dev_path = _resolve_input(args.dev_manifest, data_dir, role="dev manifest")
    core_path = _resolve_input(args.core_manifest, data_dir, role="core manifest")
    output = (
        args.output.resolve()
        if args.output is not None
        else final_path.with_name("run9_sarsteer_adapted_quick165.jsonl")
    )
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output.with_name(f"{output.stem}_summary.json")
    )
    plan = select_quick_gate(
        load_jsonl(final_path),
        load_jsonl(dev_path),
        load_jsonl(core_path),
        data_dir=data_dir,
    )
    report = materialize_quick_gate(
        plan,
        output=output,
        summary_output=summary_output,
        input_paths={"final300": final_path, "dev76": dev_path, "core300": core_path},
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
