from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "diagnose_sarsteer_sweep.py"
    spec = importlib.util.spec_from_file_location("_test_diagnose_sarsteer_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_script()


def _paired(record_id: str, path: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "item_id": f"item-{record_id}",
        "path": path,
        "defense": "sarsteer",
        "undefended_output": f"baseline-{record_id}",
        "defended_output": f"all-alpha-0.1-{record_id}",
        "extra_metadata": {"kept": True},
    }


def test_parse_settings_supports_all_single_comma_and_range():
    settings = SWEEP.parse_settings(
        ["all-low:0.003:all", "single:-0.01:17", "band:0.01:8,12,16-19"]
    )
    assert settings[0].layers is None
    assert settings[1].layers == (17,)
    assert settings[2].layers == (8, 12, 16, 17, 18, 19)
    assert settings[2].layer_spec == "8,12,16,17,18,19"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("missing-parts", "NAME:ALPHA:LAYERS"),
        ("bad name:0.1:all", "invalid setting name"),
        ("x:nan:all", "non-finite"),
        ("x:0.1:5-3", "descending"),
        ("x:0.1:2,2", "duplicate layer"),
        ("x:0.1:1,,2", "empty component"),
    ],
)
def test_parse_setting_rejects_ambiguous_specs(raw, match):
    with pytest.raises(ValueError, match=match):
        SWEEP.parse_setting(raw)


def test_duplicate_settings_and_missing_or_effectively_duplicate_layers_fail_closed():
    with pytest.raises(ValueError, match="duplicate setting name"):
        SWEEP.parse_settings(["same:0.1:all", "same:0.01:1"])
    with pytest.raises(ValueError, match="duplicate effective setting"):
        SWEEP.parse_settings(["a:0.1:1-2", "b:0.1:1,2"])

    settings = SWEEP.parse_settings(["all:0.1:all", "missing:0.01:9"])
    with pytest.raises(ValueError, match="absent from vectors"):
        SWEEP.resolve_settings(settings, [0, 1, 2])

    equivalent = SWEEP.parse_settings(["all:0.1:all", "explicit:0.1:0-2"])
    with pytest.raises(ValueError, match="resolve to the same"):
        SWEEP.resolve_settings(equivalent, [0, 1, 2])


def test_select_exact_records_preserves_order_metadata_and_resolves_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    relative = data_dir / "relative.wav"
    absolute = tmp_path / "absolute.wav"
    relative.write_bytes(b"wav")
    absolute.write_bytes(b"wav")
    rows = [_paired("r1", "relative.wav"), _paired("r2", str(absolute))]

    selected = SWEEP.select_paired_records(rows, ["r2", "r1"], data_dir=data_dir)

    assert [record.record_id for record in selected] == ["r2", "r1"]
    assert [record.audio_path for record in selected] == [absolute.resolve(), relative.resolve()]
    assert selected[0].row["extra_metadata"] == {"kept": True}
    assert len(selected[0].input_sha256) == 64


def test_select_records_rejects_duplicate_missing_and_incomplete_ids(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")
    row = _paired("r1", str(audio))
    with pytest.raises(ValueError, match="duplicate requested"):
        SWEEP.select_paired_records([row], ["r1", "r1"], data_dir=tmp_path)
    with pytest.raises(ValueError, match="missing from paired"):
        SWEEP.select_paired_records([row], ["missing"], data_dir=tmp_path)
    with pytest.raises(ValueError, match="duplicate paired"):
        SWEEP.select_paired_records([row, dict(row)], ["r1"], data_dir=tmp_path)
    incomplete = dict(row)
    incomplete.pop("defended_output")
    with pytest.raises(ValueError, match="defended_output"):
        SWEEP.select_paired_records([incomplete], ["r1"], data_dir=tmp_path)


def test_resume_plan_is_record_major_and_skips_exact_completed_key(tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"wav")
    records = SWEEP.select_paired_records(
        [_paired("r1", "a.wav"), _paired("r2", "b.wav")],
        ["r1", "r2"],
        data_dir=tmp_path,
    )
    settings = SWEEP.resolve_settings(
        SWEEP.parse_settings(["low:0.003:all", "mid:0.01:1-2"]), [0, 1, 2]
    )

    pending = SWEEP.plan_pending_jobs(records, settings, {("r1", "low"): {}})

    assert [job.key for job in pending] == [
        ("r1", "mid"),
        ("r2", "low"),
        ("r2", "mid"),
    ]
    with pytest.raises(ValueError, match="foreign resume keys"):
        SWEEP.plan_pending_jobs(records, settings, {("foreign", "low"): {}})


def test_completed_rows_validate_preserved_baselines_and_setting_identity(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")
    record = SWEEP.select_paired_records([_paired("r1", "a.wav")], ["r1"], data_dir=tmp_path)[0]
    setting = SWEEP.resolve_settings(SWEEP.parse_settings(["low:0.003:0-1"]), [0, 1])[0]
    job = SWEEP.SweepJob(record, setting)
    output_row = SWEEP._build_output_row(job, "generated", max_new_tokens=32)

    completed = SWEEP.validate_completed_rows([output_row], [record], [setting], max_new_tokens=32)

    assert completed[("r1", "low")]["baseline_undefended_output"] == "baseline-r1"
    assert completed[("r1", "low")]["current_all_alpha_0_1_output"] == ("all-alpha-0.1-r1")
    changed = dict(output_row)
    changed["sweep_alpha"] = 0.004
    with pytest.raises(ValueError, match="incompatible 'sweep_alpha'"):
        SWEEP.validate_completed_rows([changed], [record], [setting], max_new_tokens=32)
    with pytest.raises(ValueError, match="duplicate existing"):
        SWEEP.validate_completed_rows(
            [output_row, dict(output_row)], [record], [setting], max_new_tokens=32
        )
