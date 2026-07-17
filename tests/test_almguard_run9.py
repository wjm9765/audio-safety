from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_safety.evaluation.defense_judge import expand_aligned_rows
from audio_safety.pipelines.almguard_run9 import (
    ALMGUARD_POSITIVE_CONTROL,
    ALMGuardRun9Error,
    ArmView,
    merge_aligned_views,
    normalize_positive_control_rows,
)


def _save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _eval_row(record_id: str, path: str, *, condition: str = "clean") -> dict:
    return {
        "record_id": record_id,
        "item_id": f"item-{record_id}",
        "safety_label": "harmful",
        "condition": condition,
        "sign": None,
        "path": path,
        "reference_text": f"request-{record_id}",
        "asr_status": "passed",
    }


def _arm(row: dict, defense: str, response: str, *, index: int) -> dict:
    return {
        **row,
        "index": index,
        "staged_wav_name": f"{index:06d}.wav",
        "defense": defense,
        "output": response,
    }


def test_positive_control_normalization_preserves_metadata(tmp_path):
    audio = tmp_path / "pc.wav"
    audio.write_bytes(b"wav")
    raw = {
        "record_id": "pc-family-1",
        "item_id": "advbench-1",
        "attack_family": "advwave_p",
        "path": str(audio),
        "reference_text": "held-out request",
        "source_index": 1,
    }
    rows = normalize_positive_control_rows([raw], data_dir=tmp_path)
    assert rows == [
        {
            **raw,
            "safety_label": "harmful",
            "condition": ALMGUARD_POSITIVE_CONTROL,
            "sign": None,
            "control_type": "almguard_official_attack_holdout",
            "data_partition": "evaluation",
            "eval_only": True,
            "evaluation_eligible": True,
            "almguard_training_eligible": False,
        }
    ]


def test_merge_views_restores_order_and_preserves_staging_metadata(tmp_path):
    for name in ("a.wav", "b.wav", "pc.wav"):
        (tmp_path / name).write_bytes(b"wav")
    full = [_eval_row("r0", "a.wav"), _eval_row("r1", "b.wav", condition="pv_standard")]
    pc = [_eval_row("pc0", "pc.wav", condition=ALMGUARD_POSITIVE_CONTROL)]
    views = []
    for view_index, rows in enumerate((full, pc)):
        manifest = tmp_path / f"manifest{view_index}.jsonl"
        undefended = tmp_path / f"undefended{view_index}.jsonl"
        defended = tmp_path / f"defended{view_index}.jsonl"
        _save(manifest, rows)
        _save(
            undefended,
            [_arm(row, "none", f"u-{row['record_id']}", index=i) for i, row in enumerate(rows)],
        )
        _save(
            defended,
            list(
                reversed(
                    [
                        _arm(row, "almguard", f"d-{row['record_id']}", index=i)
                        for i, row in enumerate(rows)
                    ]
                )
            ),
        )
        views.append(ArmView(manifest, undefended, defended))

    result = merge_aligned_views(views, data_dir=tmp_path)
    assert result.view_counts == (2, 1)
    assert [row["record_id"] for row in result.undefended] == ["r0", "r1", "pc0"]
    assert [row["record_id"] for row in result.defended] == ["r0", "r1", "pc0"]
    assert [row["index"] for row in result.undefended] == [0, 1, 2]
    assert result.undefended[2]["invocation_index"] == 0
    assert result.undefended[0]["staged_wav_name"] == "000000.wav"
    expanded = expand_aligned_rows(result.undefended, result.defended)
    assert len(expanded) == 6


def test_merge_fails_on_missing_output_or_metadata_drift(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")
    row = _eval_row("r0", "a.wav")
    manifest = tmp_path / "manifest.jsonl"
    undefended = tmp_path / "undefended.jsonl"
    defended = tmp_path / "defended.jsonl"
    _save(manifest, [row])
    _save(undefended, [_arm(row, "none", "u", index=0)])
    _save(defended, [])
    with pytest.raises(ALMGuardRun9Error, match="empty"):
        merge_aligned_views([ArmView(manifest, undefended, defended)], data_dir=tmp_path)

    drift = _arm(row, "almguard", "d", index=0)
    drift["sign"] = 3
    _save(defended, [drift])
    with pytest.raises(ALMGuardRun9Error, match="metadata differs"):
        merge_aligned_views([ArmView(manifest, undefended, defended)], data_dir=tmp_path)


def test_merge_fails_on_duplicate_cross_view_record_id(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")
    row = _eval_row("r0", "a.wav")
    views = []
    for i in range(2):
        manifest = tmp_path / f"m{i}.jsonl"
        undefended = tmp_path / f"u{i}.jsonl"
        defended = tmp_path / f"d{i}.jsonl"
        _save(manifest, [row])
        _save(undefended, [_arm(row, "none", "u", index=0)])
        _save(defended, [_arm(row, "almguard", "d", index=0)])
        views.append(ArmView(manifest, undefended, defended))
    with pytest.raises(ALMGuardRun9Error, match="multiple canonical views"):
        merge_aligned_views(views, data_dir=tmp_path)
