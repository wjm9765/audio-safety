#!/usr/bin/env -S uv run python
"""Build the deterministic 300-row Run 9 SARSteer directional gate subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl, shard_by_item
from audio_safety.utils.io import load_jsonl, save_json

HARMFUL_CATEGORIES = tuple(range(1, 8))
SOFT_CATEGORIES = tuple(range(8, 11))
HARMFUL_PER_CATEGORY = 10
SOFT_PER_CATEGORY = 10
UTILITY_ROWS = 30
POSITIVE_CONTROL_ROWS = 30
EXPECTED_ROWS = 300
EXPECTED_SHARD_ROWS = (150, 150)


@dataclass(frozen=True)
class CoreSelection:
    rows: list[dict[str, Any]]
    harmful_item_ids: dict[int, tuple[str, ...]]
    soft_item_ids: dict[int, tuple[str, ...]]


def _category(row: Mapping[str, Any], *, role: str, index: int) -> int:
    value = row.get("category_id")
    if isinstance(value, bool):
        raise ValueError(f"{role} row {index} has invalid category_id={value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} row {index} has invalid category_id={value!r}") from exc


def _sign(row: Mapping[str, Any], *, index: int) -> int | None:
    value = row.get("sign")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {index} has invalid sign={value!r}") from exc
    if number == -3.0:
        return -3
    if number == 3.0:
        return 3
    return None


def _first_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    condition: str,
    count: int,
) -> list[int]:
    selected: list[int] = []
    seen_items: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("evaluation_eligible") is not True:
            continue
        if str(row.get("gate_role")) != role or str(row.get("condition")) != condition:
            continue
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise ValueError(f"{role} row {index} lacks item_id")
        if item_id in seen_items:
            raise ValueError(f"duplicate eligible {role} item_id: {item_id}")
        seen_items.add(item_id)
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError(f"need {count} eligible {role} rows, found {len(selected)}")


def select_directional_core(rows: Sequence[Mapping[str, Any]]) -> CoreSelection:
    """Select the fixed gate cells and preserve canonical input row order."""

    if not rows:
        raise ValueError("input manifest is empty")

    harmful_cells: dict[tuple[int, str, str, int | None], int] = {}
    harmful_order: dict[int, list[str]] = defaultdict(list)
    harmful_seen: dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if row.get("evaluation_eligible") is not True:
            continue
        if str(row.get("gate_role")) != "harmful_eval":
            continue
        if str(row.get("safety_label")) != "harmful":
            raise ValueError(f"harmful_eval row {index} is not safety_label='harmful'")
        category = _category(row, role="harmful_eval", index=index)
        if category not in HARMFUL_CATEGORIES:
            continue
        condition = str(row.get("condition"))
        sign = _sign(row, index=index)
        if condition == "clean":
            if row.get("sign") is not None:
                raise ValueError(f"clean harmful row {index} must have sign=null")
            cell = (category, str(row.get("item_id", "")).strip(), condition, None)
        elif condition == "pv_standard" and sign in {-3, 3}:
            cell = (category, str(row.get("item_id", "")).strip(), condition, sign)
        else:
            continue
        item_id = cell[1]
        if not item_id:
            raise ValueError(f"harmful_eval row {index} lacks item_id")
        if cell in harmful_cells:
            raise ValueError(f"duplicate eligible harmful gate cell: {cell}")
        harmful_cells[cell] = index
        if item_id not in harmful_seen[category]:
            harmful_seen[category].add(item_id)
            harmful_order[category].append(item_id)

    selected_indices: set[int] = set()
    harmful_selected: dict[int, tuple[str, ...]] = {}
    required = (("clean", None), ("pv_standard", -3), ("pv_standard", 3))
    for category in HARMFUL_CATEGORIES:
        complete: list[str] = []
        for item_id in harmful_order[category]:
            keys = [(category, item_id, condition, sign) for condition, sign in required]
            if all(key in harmful_cells for key in keys):
                complete.append(item_id)
            if len(complete) == HARMFUL_PER_CATEGORY:
                break
        if len(complete) != HARMFUL_PER_CATEGORY:
            raise ValueError(
                f"category {category} needs {HARMFUL_PER_CATEGORY} complete eligible "
                f"clean/pv_standard(-3,+3) items, found {len(complete)}"
            )
        harmful_selected[category] = tuple(complete)
        for item_id in complete:
            for condition, sign in required:
                selected_indices.add(harmful_cells[(category, item_id, condition, sign)])

    soft_selected: dict[int, tuple[str, ...]] = {}
    for category in SOFT_CATEGORIES:
        item_ids: list[str] = []
        category_indices: list[int] = []
        seen_items: set[str] = set()
        for index, row in enumerate(rows):
            if row.get("evaluation_eligible") is not True:
                continue
            if str(row.get("gate_role")) != "soft_overrefusal":
                continue
            if str(row.get("condition")) != "clean":
                continue
            if _category(row, role="soft_overrefusal", index=index) != category:
                continue
            if str(row.get("safety_label")) != "benign":
                raise ValueError(f"soft_overrefusal row {index} is not benign")
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                raise ValueError(f"soft_overrefusal row {index} lacks item_id")
            if item_id in seen_items:
                raise ValueError(
                    f"duplicate eligible soft item_id in category {category}: {item_id}"
                )
            seen_items.add(item_id)
            item_ids.append(item_id)
            category_indices.append(index)
            if len(item_ids) == SOFT_PER_CATEGORY:
                break
        if len(item_ids) != SOFT_PER_CATEGORY:
            raise ValueError(
                f"category {category} needs {SOFT_PER_CATEGORY} eligible soft clean rows, "
                f"found {len(item_ids)}"
            )
        soft_selected[category] = tuple(item_ids)
        selected_indices.update(category_indices)

    selected_indices.update(
        _first_unique_rows(
            rows,
            role="utility_eval",
            condition="clean",
            count=UTILITY_ROWS,
        )
    )
    selected_indices.update(
        _first_unique_rows(
            rows,
            role="positive_control_eval",
            condition="positive_control",
            count=POSITIVE_CONTROL_ROWS,
        )
    )

    selected = [dict(row) for index, row in enumerate(rows) if index in selected_indices]
    if len(selected) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} selected rows, got {len(selected)}")
    if any(row.get("evaluation_eligible") is not True for row in selected):
        raise AssertionError("ineligible row entered directional core")
    record_ids = [str(row.get("record_id", "")).strip() for row in selected]
    if any(not record_id for record_id in record_ids):
        raise ValueError("every selected row must have a non-empty record_id")
    if len(set(record_ids)) != EXPECTED_ROWS:
        raise ValueError("selected record_ids are not unique")
    if any(row.get("condition") in {"pv_locked", "mel_matched_ctrl"} for row in selected):
        raise AssertionError("non-primary attack control entered directional core")

    role_counts = Counter(str(row.get("gate_role")) for row in selected)
    expected_roles = {
        "harmful_eval": 210,
        "soft_overrefusal": 30,
        "utility_eval": 30,
        "positive_control_eval": 30,
    }
    if role_counts != expected_roles:
        raise ValueError(f"unexpected selected role counts: {dict(role_counts)}")

    return CoreSelection(
        rows=selected,
        harmful_item_ids=harmful_selected,
        soft_item_ids=soft_selected,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <input-dir>/run9_sarsteer_directional_core300.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"input manifest not found: {source}")
    output = (
        args.output.resolve()
        if args.output is not None
        else source.with_name("run9_sarsteer_directional_core300.jsonl")
    )
    selection = select_directional_core(load_jsonl(source))
    shards = shard_by_item(selection.rows, 2)
    shard_counts = tuple(len(shard) for shard in shards)
    if shard_counts != EXPECTED_SHARD_ROWS:
        raise ValueError(f"expected shard rows {EXPECTED_SHARD_ROWS}, got {shard_counts}")

    shard_paths = [
        output.with_name(f"{output.stem}.shard{index:02d}-of-02.jsonl") for index in range(2)
    ]
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "rows": len(selection.rows),
        "unique_record_ids": len({row["record_id"] for row in selection.rows}),
        "unique_item_ids": len({row["item_id"] for row in selection.rows}),
        "all_evaluation_eligible": True,
        "role_counts": dict(sorted(Counter(row["gate_role"] for row in selection.rows).items())),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in selection.rows).items())
        ),
        "harmful_item_ids_by_category": {
            str(key): list(value) for key, value in selection.harmful_item_ids.items()
        },
        "soft_item_ids_by_category": {
            str(key): list(value) for key, value in selection.soft_item_ids.items()
        },
        "selection_rule": {
            "harmful": "categories 1-7, first 10 complete eligible clean+pv_standard(-3,+3)",
            "soft": "categories 8-10, first 10 eligible clean",
            "utility": "first 30 unique eligible clean",
            "positive_control": "first 30 unique eligible positive_control",
            "canonical_input_order_preserved": True,
            "excluded_conditions": ["pv_locked", "mel_matched_ctrl"],
        },
        "shards": {
            "method": "audio_safety.data.run9_eval_manifest.shard_by_item",
            "paths": [str(path) for path in shard_paths],
            "row_counts": list(shard_counts),
            "item_counts": [len({row["item_id"] for row in shard}) for shard in shards],
            "same_item_kept_together": True,
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    atomic_save_jsonl(selection.rows, output)
    for shard, path in zip(shards, shard_paths, strict=True):
        atomic_save_jsonl(shard, path)
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
