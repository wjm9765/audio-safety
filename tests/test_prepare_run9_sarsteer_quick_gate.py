from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_run9_sarsteer_quick_gate.py"
    spec = importlib.util.spec_from_file_location("_test_sarsteer_quick_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUICK = _load_script()


def _touch(data_dir: Path, relative: str) -> str:
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())
    return relative


def _row(
    data_dir: Path,
    item_id: str,
    role: str,
    condition: str,
    *,
    category: int | None = None,
    sign: int | None = None,
    prefix: str = "final",
) -> dict[str, object]:
    sign_token = "none" if sign is None else str(sign)
    path = _touch(
        data_dir,
        f"{prefix}/{role}/{condition}/{sign_token}/{item_id}.wav",
    )
    harmful = role in {"harmful_eval", "positive_control_eval"}
    row: dict[str, object] = {
        "item_id": item_id,
        "record_id": f"{prefix}-{role}-{item_id}-{condition}-{sign_token}",
        "source": prefix,
        "category_id": category,
        "safety_label": "harmful" if harmful else "benign",
        "gate_role": role,
        "condition": condition,
        "style": condition,
        "sign": sign,
        "path": path,
        "reference_text": f"{prefix} text for {item_id}",
        "evaluation_eligible": True,
        "final_gate_eligible": prefix == "final",
    }
    if role == "positive_control_eval":
        row.update(
            {
                "attack_family": "jb_pap",
                "historical_outcomes_used_for_selection": False,
            }
        )
    return row


def _fixture(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    final: list[dict[str, object]] = []
    harmful_ids: dict[int, list[str]] = {}
    for category in QUICK.HARMFUL_CATEGORIES:
        harmful_ids[category] = [f"h{category}-{index:02d}" for index in range(10)]
        for item_id in harmful_ids[category]:
            final.extend(
                [
                    _row(
                        data_dir,
                        item_id,
                        "harmful_eval",
                        "clean",
                        category=category,
                    ),
                    _row(
                        data_dir,
                        item_id,
                        "harmful_eval",
                        "pv_standard",
                        category=category,
                        sign=-3,
                    ),
                    _row(
                        data_dir,
                        item_id,
                        "harmful_eval",
                        "pv_standard",
                        category=category,
                        sign=3,
                    ),
                ]
            )
    for category in QUICK.SOFT_CATEGORIES:
        final.extend(
            _row(
                data_dir,
                f"soft{category}-{index:02d}",
                "soft_overrefusal",
                "clean",
                category=category,
            )
            for index in range(10)
        )
    final.extend(
        _row(data_dir, f"utility-{index:02d}", "utility_eval", "clean") for index in range(30)
    )
    final.extend(
        _row(
            data_dir,
            f"pap-{index:02d}",
            "positive_control_eval",
            "positive_control",
        )
        for index in range(30)
    )
    assert len(final) == 300

    core = [
        _row(
            data_dir,
            f"core-{index:03d}",
            "utility_eval",
            "clean",
            prefix="core",
        )
        for index in range(300)
    ]
    dev = [dict(row) for row in core[:76]]
    return data_dir, final, dev, core, harmful_ids


def test_selects_exact_quick165_and_keeps_items_together(tmp_path: Path):
    data_dir, final, dev, core, harmful_ids = _fixture(tmp_path)
    first = QUICK.select_quick_gate(final, dev, core, data_dir=data_dir)
    second = QUICK.select_quick_gate(final, dev, core, data_dir=data_dir)

    assert first.rows == second.rows
    assert len(first.rows) == 165
    assert Counter(row["gate_role"] for row in first.rows) == QUICK.EXPECTED_ROLE_COUNTS
    assert Counter(row["condition"] for row in first.rows) == QUICK.EXPECTED_CONDITION_COUNTS
    assert [len(shard) for shard in first.shards] == [83, 82]
    selected_harmful = {row["item_id"] for row in first.rows if row["gate_role"] == "harmful_eval"}
    assert selected_harmful == {
        item_id for category in QUICK.HARMFUL_CATEGORIES for item_id in harmful_ids[category][:5]
    }
    assignment: dict[str, set[int]] = {}
    for shard_index, shard in enumerate(first.shards):
        for row in shard:
            assignment.setdefault(str(row["item_id"]), set()).add(shard_index)
    assert all(len(shard_ids) == 1 for shard_ids in assignment.values())
    assert first.summary["experiment_contract"]["alpha"] == 0.03
    assert first.summary["leakage"]["all_passed"] is True


def test_materializes_hashes_and_summary_sidecar(tmp_path: Path):
    data_dir, final, dev, core, _harmful_ids = _fixture(tmp_path)
    plan = QUICK.select_quick_gate(final, dev, core, data_dir=data_dir)
    inputs = {}
    for name, rows in (("final300", final), ("dev76", dev), ("core300", core)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        inputs[name] = path
    output = tmp_path / "run9_sarsteer_adapted_quick165.jsonl"
    summary = tmp_path / "run9_sarsteer_adapted_quick165_summary.json"
    report = QUICK.materialize_quick_gate(
        plan,
        output=output,
        summary_output=summary,
        input_paths=inputs,
    )

    assert output.is_file()
    assert output.with_name(f"{output.stem}.shard00-of-02.jsonl").is_file()
    assert output.with_name(f"{output.stem}.shard01-of-02.jsonl").is_file()
    assert summary.is_file()
    assert summary.with_suffix(".json.sha256").is_file()
    assert len(report["outputs"]["sha256"]["manifest"]) == 64
    assert len(report["summary_sha256"]) == 64


def test_fails_closed_on_dev_item_leakage(tmp_path: Path):
    data_dir, final, dev, core, _harmful_ids = _fixture(tmp_path)
    dev[0] = {
        **dev[0],
        "item_id": final[0]["item_id"],
    }
    with pytest.raises(ValueError, match="quick165/alpha dev76 leakage"):
        QUICK.select_quick_gate(final, dev, core, data_dir=data_dir)


def test_fails_closed_when_source_final_contract_changes(tmp_path: Path):
    data_dir, final, dev, core, _harmful_ids = _fixture(tmp_path)
    final[0] = {**final[0], "condition": "pv_locked"}
    with pytest.raises(ValueError, match="source final condition counts mismatch"):
        QUICK.select_quick_gate(final, dev, core, data_dir=data_dir)
