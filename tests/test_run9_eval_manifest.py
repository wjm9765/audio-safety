from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from audio_safety.data.run9_eval_manifest import (
    ATTACK_CONDITIONS,
    build_run9_eval_manifest,
    shard_by_item,
    stable_row_key,
    validate_calibration_disjoint,
)


def _touch(data_dir: Path, relative: str) -> str:
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return relative


def _fixture_rows(data_dir: Path):
    clean = []
    for item_id in ("h0", "h1"):
        clean.append(
            {
                "item_id": item_id,
                "source": "figstep_safebench",
                "safety_label": "harmful",
                "gate_role": "harmful_eval",
                "condition": "clean",
                "style": "neutral",
                "path": _touch(data_dir, f"clean/{item_id}.wav"),
                "reference_text": f"harmful question {item_id}",
                "asr_required": True,
            }
        )
    clean.append(
        {
            "item_id": "soft0",
            "source": "figstep_safebench",
            "safety_label": "benign",
            "gate_role": "soft_overrefusal",
            "condition": "clean",
            "style": "neutral",
            "path": _touch(data_dir, "clean/soft0.wav"),
            "reference_text": "soft topic",
            "asr_required": True,
        }
    )

    attacks = []
    for item_id in ("h0", "h1"):
        for condition in ATTACK_CONDITIONS:
            for sign in (-3.0, 3.0):
                under_test = condition == "pv_standard"
                attacks.append(
                    {
                        "item_id": item_id,
                        "source": "figstep_safebench",
                        "safety_label": "harmful",
                        "gate_role": "harmful_eval",
                        "condition": condition,
                        "style": f"{condition}_{sign:g}",
                        "sign": sign,
                        "path": _touch(
                            data_dir,
                            f"attacks/{condition}/{sign:g}/{item_id}.wav",
                        ),
                        "reference_text": f"harmful question {item_id}",
                        "phase_under_test": under_test,
                        "phase_under_test_tag": (
                            "run7_frozen_pv_standard" if under_test else None
                        ),
                        "almguard_training_policy": (
                            "exclude_primary_phase_attack_under_test"
                            if under_test
                            else "eligible_non_target_control"
                        ),
                        "asr_required": True,
                    }
                )
    utility = [
        {
            "item_id": "libri0",
            "dataset": "librispeech_test_clean",
            "path": _touch(data_dir, "utility/libri0.flac"),
            "reference_text": "utility speech",
        }
    ]
    positive = [
        {
            "item_id": "adv0",
            "source": "advbench_audio",
            "path": _touch(data_dir, "positive/adv0.wav"),
            "reference_text": "positive control harmful request",
        }
    ]
    return clean, attacks, utility, positive


def _expected_counts() -> dict[tuple[str, str, str], int]:
    return {
        ("harmful_eval", "clean", ""): 2,
        ("soft_overrefusal", "clean", ""): 1,
        ("utility_eval", "clean", ""): 1,
        ("positive_control_eval", "positive_control", ""): 1,
        **{
            ("harmful_eval", condition, sign): 2
            for condition in ATTACK_CONDITIONS
            for sign in ("-3", "3")
        },
    }


def _build(data_dir: Path, **kwargs):
    clean, attacks, utility, positive = _fixture_rows(data_dir)
    return build_run9_eval_manifest(
        clean,
        attacks,
        utility,
        positive,
        data_dir=data_dir,
        expected_counts=_expected_counts(),
        **kwargs,
    )


def test_builder_normalizes_all_cells_and_never_marks_eval_training_eligible(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = _build(data_dir)

    assert len(result.rows) == 17
    assert len({row["record_id"] for row in result.rows}) == 17
    assert Counter(row["condition"] for row in result.rows) == {
        "clean": 4,
        "pv_standard": 4,
        "pv_locked": 4,
        "mel_matched_ctrl": 4,
        "positive_control": 1,
    }
    assert all(row["eval_only"] is True for row in result.rows)
    assert all(row["almguard_training_eligible"] is False for row in result.rows)
    target = [row for row in result.rows if row["condition"] == "pv_standard"]
    assert all(row["phase_under_test"] is True for row in target)
    assert all(row["almguard_training_policy"].startswith("exclude") for row in target)
    utility = next(row for row in result.rows if row["gate_role"] == "utility_eval")
    assert utility["source"] == "librispeech_test_clean"
    assert utility["safety_label"] == "benign"
    assert result.summary["rows"] == 17
    assert result.summary["almguard_training_eligible_rows"] == 0


def test_builder_fails_on_duplicate_stable_key_and_incomplete_cells(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    clean, attacks, utility, positive = _fixture_rows(data_dir)
    attacks.append(dict(attacks[0], path=attacks[1]["path"]))
    expected = dict(_expected_counts())
    expected[("harmful_eval", "pv_standard", "-3")] = 3
    with pytest.raises(ValueError, match="duplicate stable evaluation key"):
        build_run9_eval_manifest(
            clean,
            attacks,
            utility,
            positive,
            data_dir=data_dir,
            expected_counts=expected,
        )

    with pytest.raises(ValueError, match="cell counts"):
        build_run9_eval_manifest(
            clean,
            attacks[:-2],
            utility,
            positive,
            data_dir=data_dir,
            expected_counts=_expected_counts(),
        )


def test_primary_source_and_all_exact_asset_leakage_fail_closed(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = _build(data_dir)
    rows = result.rows

    with pytest.raises(ValueError, match="item_id"):
        validate_calibration_disjoint(
            rows,
            [{"item_id": "libri0", "source": "other"}],
            data_dir=data_dir,
        )
    with pytest.raises(ValueError, match="path="):
        validate_calibration_disjoint(
            rows,
            [
                {
                    "item_id": "different",
                    "source": "other",
                    "path": rows[-1]["path"],
                }
            ],
            data_dir=data_dir,
        )
    with pytest.raises(ValueError, match="source="):
        validate_calibration_disjoint(
            rows,
            [{"item_id": "different", "source": "figstep_safebench"}],
            data_dir=data_dir,
        )

    # Held-out positive-control/utility assets intentionally share their method's
    # dataset domain; exact item/path leakage remains prohibited above.
    report = validate_calibration_disjoint(
        rows,
        [
            {"item_id": "adv-calib", "source": "advbench_audio"},
            {"item_id": "libri-calib", "source": "librispeech_test_clean"},
        ],
        data_dir=data_dir,
    )
    assert report["passed"] is True
    with pytest.raises(ValueError, match="source="):
        validate_calibration_disjoint(
            rows,
            [{"item_id": "adv-calib", "source": "advbench_audio"}],
            data_dir=data_dir,
            strict_source_all_roles=True,
        )


def test_asr_merge_keeps_failed_and_missing_rows_with_fail_closed_flags(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base = _build(data_dir)
    clean = next(row for row in base.rows if row["item_id"] == "h0" and row["condition"] == "clean")
    attack = next(
        row
        for row in base.rows
        if row["item_id"] == "h1" and row["condition"] == "pv_standard"
    )
    scores = [
        {
            **clean,
            "transcript": clean["reference_text"],
            "wer": 0.0,
            "token_overlap": 1.0,
            "transcript_control_passed": True,
            "asr_status": "ok",
        },
        {
            **attack,
            "transcript": "unrelated",
            "wer": 1.0,
            "token_overlap": 0.0,
            "transcript_control_passed": False,
            "asr_status": "ok",
        },
    ]
    result = _build(data_dir, asr_score_rows=scores)

    assert len(result.rows) == len(base.rows)
    by_id = {row["record_id"]: row for row in result.rows}
    assert by_id[clean["record_id"]]["evaluation_eligible"] is True
    assert by_id[attack["record_id"]]["evaluation_eligible"] is False
    assert by_id[attack["record_id"]]["eligibility_reasons"] == [
        "transcript_control_failed"
    ]
    missing = next(
        row
        for row in result.rows
        if row["item_id"] == "h1" and row["condition"] == "clean"
    )
    assert missing["evaluation_eligible"] is False
    assert missing["eligibility_reasons"] == ["asr_score_missing"]
    assert result.summary["asr"]["rows_dropped"] == 0


def test_item_grouped_shards_are_balanced_disjoint_and_complete(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = _build(data_dir).rows
    shards = shard_by_item(rows, 2)

    assert [len(shard) for shard in shards] == [9, 8]
    record_sets = [{row["record_id"] for row in shard} for shard in shards]
    assert record_sets[0].isdisjoint(record_sets[1])
    assert record_sets[0] | record_sets[1] == {row["record_id"] for row in rows}
    item_assignment = {
        item_id: {
            index
            for index, shard in enumerate(shards)
            if any(row["item_id"] == item_id for row in shard)
        }
        for item_id in {row["item_id"] for row in rows}
    }
    assert all(len(assignments) == 1 for assignments in item_assignment.values())
    assert all(stable_row_key(row) for row in rows)

