from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from audio_safety.evaluation.sarsteer_shards import ShardPair, validate_and_merge


def _row(item: str, condition: str, sign: float | None = None) -> dict[str, object]:
    return {
        "item_id": item,
        "record_id": f"record-{item}-{condition}-{sign}",
        "safety_label": "harmful",
        "condition": condition,
        "sign": sign,
        "path": f"{item}.wav",
    }


def _completed(row: dict[str, object]) -> dict[str, object]:
    return {
        **row,
        "defense": "sarsteer",
        "undefended_output": "opaque-undefended",
        "defended_output": "opaque-defended",
    }


def _save(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path):
    rows = [
        _row("a", "clean"),
        _row("b", "clean"),
        _row("a", "pv_standard", -3),
        _row("b", "pv_standard", 3),
    ]
    canonical = tmp_path / "canonical.jsonl"
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    output0 = tmp_path / "output0.jsonl"
    output1 = tmp_path / "output1.jsonl"
    _save(canonical, rows)
    _save(shard0, [rows[0], rows[2]])
    _save(shard1, [rows[1], rows[3]])
    _save(output0, [_completed(rows[0]), _completed(rows[2])])
    _save(output1, [_completed(rows[1]), _completed(rows[3])])
    pairs = [ShardPair(shard0, output0), ShardPair(shard1, output1)]
    return rows, canonical, pairs


def test_merge_restores_canonical_input_order(tmp_path):
    rows, canonical, pairs = _fixture(tmp_path)

    result = validate_and_merge(canonical, pairs)

    assert result.canonical_count == 4
    assert result.shard_counts == (2, 2)
    assert [row["record_id"] for row in result.rows] == [
        row["record_id"] for row in rows
    ]


def test_merge_rejects_missing_duplicate_and_foreign_shard_keys(tmp_path):
    rows, canonical, pairs = _fixture(tmp_path)
    _save(pairs[1].manifest, [rows[1]])
    _save(pairs[1].output, [_completed(rows[1])])
    with pytest.raises(ValueError, match="do not cover.*missing 1"):
        validate_and_merge(canonical, pairs)

    rows, canonical, pairs = _fixture(tmp_path)
    _save(pairs[1].manifest, [rows[0], rows[3]])
    _save(pairs[1].output, [_completed(rows[0]), _completed(rows[3])])
    with pytest.raises(ValueError, match="multiple shard manifests"):
        validate_and_merge(canonical, pairs)

    rows, canonical, pairs = _fixture(tmp_path)
    foreign = _row("foreign", "clean")
    _save(pairs[1].manifest, [rows[1], foreign])
    _save(pairs[1].output, [_completed(rows[1]), _completed(foreign)])
    with pytest.raises(ValueError, match="absent from the canonical"):
        validate_and_merge(canonical, pairs)


def test_merge_rejects_output_order_drift_and_incomplete_rows(tmp_path):
    rows, canonical, pairs = _fixture(tmp_path)
    _save(pairs[0].output, [_completed(rows[2]), _completed(rows[0])])
    with pytest.raises(ValueError, match="stable-key order differs"):
        validate_and_merge(canonical, pairs)

    rows, canonical, pairs = _fixture(tmp_path)
    incomplete = _completed(rows[0])
    incomplete.pop("defended_output")
    _save(pairs[0].output, [incomplete, _completed(rows[2])])
    with pytest.raises(ValueError, match="defended_output"):
        validate_and_merge(canonical, pairs)


def test_merge_rejects_stable_key_and_record_id_drift(tmp_path):
    rows, canonical, pairs = _fixture(tmp_path)
    drifted_key = _completed(rows[0])
    drifted_key["sign"] = 3
    _save(pairs[0].output, [drifted_key, _completed(rows[2])])
    with pytest.raises(ValueError, match="stable-key order differs"):
        validate_and_merge(canonical, pairs)

    rows, canonical, pairs = _fixture(tmp_path)
    drifted_id = _completed(rows[0])
    drifted_id["record_id"] = "wrong"
    _save(pairs[0].output, [drifted_id, _completed(rows[2])])
    with pytest.raises(ValueError, match="record_id differs"):
        validate_and_merge(canonical, pairs)


def test_merge_rejects_non_key_metadata_drift(tmp_path):
    rows, canonical, pairs = _fixture(tmp_path)
    drifted_manifest = dict(rows[0])
    drifted_manifest["path"] = "wrong.wav"
    _save(pairs[0].manifest, [drifted_manifest, rows[2]])
    with pytest.raises(ValueError, match="changed input metadata field 'path'"):
        validate_and_merge(canonical, pairs)

    rows, canonical, pairs = _fixture(tmp_path)
    drifted_output = _completed(rows[0])
    drifted_output["path"] = "wrong.wav"
    _save(pairs[0].output, [drifted_output, _completed(rows[2])])
    with pytest.raises(ValueError, match="changed input metadata field 'path'"):
        validate_and_merge(canonical, pairs)


def test_cli_atomic_write_dry_run_and_overwrite_guard(tmp_path, capsys):
    rows, canonical, pairs = _fixture(tmp_path)
    script_path = Path(__file__).parents[1] / "scripts" / "merge_sarsteer_shards.py"
    spec = importlib.util.spec_from_file_location("_test_merge_sarsteer_shards", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = tmp_path / "merged.jsonl"
    argv = [
        "--manifest",
        str(canonical),
        "--shard",
        str(pairs[0].manifest),
        str(pairs[0].output),
        "--shard",
        str(pairs[1].manifest),
        str(pairs[1].output),
        "--output",
        str(output),
    ]

    module.main([*argv, "--dry-run"])
    assert not output.exists()
    assert '"generation_bodies_logged": false' in capsys.readouterr().out

    module.main(argv)
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["record_id"] for row in actual] == [row["record_id"] for row in rows]
    with pytest.raises(SystemExit, match="already exists"):
        module.main(argv)
    module.main([*argv, "--overwrite"])
