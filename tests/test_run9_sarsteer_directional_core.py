from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from audio_safety.data.run9_eval_manifest import shard_by_item


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_run9_sarsteer_directional_core.py"
    spec = importlib.util.spec_from_file_location("_test_run9_sar_core", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load_script()


def _row(
    item_id: str,
    role: str,
    condition: str,
    *,
    category: int | None = None,
    sign: int | None = None,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "record_id": f"record-{item_id}-{condition}-{sign}",
        "gate_role": role,
        "safety_label": "harmful"
        if role in {"harmful_eval", "positive_control_eval"}
        else "benign",
        "condition": condition,
        "category_id": category,
        "sign": sign,
        "evaluation_eligible": eligible,
    }


def _fixture() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    harmful_ids: dict[int, list[str]] = {}
    for category in range(1, 8):
        # The lexically first/first-seen item is incomplete and must be skipped.
        incomplete = f"h{category}-00-incomplete"
        rows.append(_row(incomplete, "harmful_eval", "clean", category=category))
        harmful_ids[category] = [f"h{category}-{index:02d}" for index in range(1, 12)]
        rows.extend(
            _row(item_id, "harmful_eval", "clean", category=category)
            for item_id in harmful_ids[category]
        )
    for category in range(8, 11):
        rows.extend(
            _row(f"soft{category}-{index:02d}", "soft_overrefusal", "clean", category=category)
            for index in range(11)
        )
    rows.extend(_row(f"utility-{index:02d}", "utility_eval", "clean") for index in range(31))
    rows.extend(
        _row(f"pc-{index:02d}", "positive_control_eval", "positive_control") for index in range(31)
    )
    for sign in (-3, 3):
        for category in range(1, 8):
            rows.extend(
                _row(
                    item_id,
                    "harmful_eval",
                    "pv_standard",
                    category=category,
                    sign=sign,
                )
                for item_id in harmful_ids[category]
            )
            # These controls must never enter the directional subset.
            rows.extend(
                _row(
                    item_id,
                    "harmful_eval",
                    "pv_locked",
                    category=category,
                    sign=sign,
                )
                for item_id in harmful_ids[category]
            )
    return rows


def test_selects_exact_core_in_input_order_and_balances_item_shards():
    rows = _fixture()
    selection = CORE.select_directional_core(rows)

    assert len(selection.rows) == 300
    assert len({row["record_id"] for row in selection.rows}) == 300
    assert Counter(row["gate_role"] for row in selection.rows) == {
        "harmful_eval": 210,
        "soft_overrefusal": 30,
        "utility_eval": 30,
        "positive_control_eval": 30,
    }
    assert Counter(row["condition"] for row in selection.rows) == {
        "clean": 130,
        "positive_control": 30,
        "pv_standard": 140,
    }
    source_order = {row["record_id"]: index for index, row in enumerate(rows)}
    selected_order = [source_order[row["record_id"]] for row in selection.rows]
    assert selected_order == sorted(selected_order)
    assert all("incomplete" not in item_id for item_id in selection.harmful_item_ids[1])
    assert selection.harmful_item_ids[1] == tuple(f"h1-{index:02d}" for index in range(1, 11))

    shards = shard_by_item(selection.rows, 2)
    assert [len(shard) for shard in shards] == [150, 150]
    assignment: dict[str, set[int]] = {}
    for shard_index, shard in enumerate(shards):
        for row in shard:
            assignment.setdefault(str(row["item_id"]), set()).add(shard_index)
    assert all(len(shard_ids) == 1 for shard_ids in assignment.values())


def test_fails_closed_when_a_harmful_category_lacks_ten_complete_items():
    rows = [
        row
        for row in _fixture()
        if not (row["item_id"] == "h7-10" and row["condition"] == "pv_standard")
    ]
    # h7-11 remains complete, so remove it too; only nine complete candidates remain.
    rows = [row for row in rows if row["item_id"] != "h7-11"]
    with pytest.raises(ValueError, match="category 7 needs 10 complete"):
        CORE.select_directional_core(rows)
