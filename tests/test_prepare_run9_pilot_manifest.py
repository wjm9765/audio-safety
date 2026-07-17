import importlib.util
import json
import sys
from pathlib import Path

import pytest

from audio_safety.config import load_experiment_config
from audio_safety.config.schema import (
    AudioRdoDatasetConfig,
    DataSplitConfig,
    ExperimentConfig,
    ModelConfig,
    SARSteerConfig,
    TtsConfig,
)
from audio_safety.data.datasets import AudioRdoPair
from audio_safety.pipelines.rdo_gate import split_ids


def _load_adapter_script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_run9_pilot_manifest.py"
    spec = importlib.util.spec_from_file_location("prepare_run9_pilot_manifest_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter_script()


def _fixture(tmp_path: Path):
    data_dir = tmp_path / "data"
    variant_root = tmp_path / "run7"
    pairs = [
        AudioRdoPair(
            item_id=f"item_{index}",
            category="test",
            harmful_text=f"harmful {index}",
            benign_text=f"benign {index}",
            source="test",
        )
        for index in range(6)
    ]
    cfg = ExperimentConfig(
        name="pilot_test",
        seed=0,
        model=ModelConfig(model_id="Qwen/Qwen2-Audio-7B-Instruct"),
        dataset=AudioRdoDatasetConfig(
            harmful_source="test",
            source_file=Path("pairs.jsonl"),
            n_pairs=6,
            min_pairs=1,
            splits=DataSplitConfig(train=0.5, validation=0.0, heldout=0.5),
            tts=TtsConfig(audio_subdir=Path("pilot_audio")),
            styles=[
                "neutral",
                "phase_pv_standard_m3",
                "phase_pv_standard_p3",
                "eq_mel_matched_ctrl_m3",
                "eq_mel_matched_ctrl_p3",
            ],
        ),
        sarsteer=SARSteerConfig(n_refusal_calib=3, n_benign_pca=3),
    )
    split_map = split_ids(pairs, cfg)

    calibration_rows = []
    for pair in pairs:
        for label, text in (("harmful", pair.harmful_text), ("benign", pair.benign_text)):
            audio = data_dir / "source" / label / f"{pair.item_id}.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(f"{pair.item_id}-{label}".encode())
            calibration_rows.append(
                {
                    "item_id": pair.item_id,
                    "safety_label": label,
                    "style": "neutral",
                    "path": audio.relative_to(data_dir).as_posix(),
                    "reference_text": text,
                    "status": "rendered",
                    "transcript_control_passed": True,
                }
            )

    cells = []
    for pair in pairs:
        clean = data_dir / "source" / "harmful" / f"{pair.item_id}.wav"
        cells.append(
            {
                "item_id": pair.item_id,
                "condition": "neutral",
                "sign": 0.0,
                "variant_path": str(clean.resolve()),
                "reference_text": pair.harmful_text,
                "safety_label": "harmful",
                "reviewed_behavior_label": "policy_refusal",
                "wer": None,
            }
        )
        for condition in ("pv_standard", "mel_matched_ctrl"):
            for sign in (-3.0, 3.0):
                relative = (
                    Path("pitch_frontend") / "audio" / (f"{pair.item_id}_{condition}_{sign:+g}.wav")
                )
                audio = variant_root / relative
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_bytes(f"{pair.item_id}-{condition}-{sign}".encode())
                cells.append(
                    {
                        "item_id": pair.item_id,
                        "condition": condition,
                        "sign": sign,
                        "variant_path": relative.as_posix(),
                        "reference_text": pair.harmful_text,
                        "safety_label": "harmful",
                        "reviewed_behavior_label": "policy_refusal",
                        "wer": 0.1 if sign < 0 else None,
                    }
                )
    cells_path = tmp_path / "cells.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    cells_path.write_text("\n".join(json.dumps(row) for row in cells) + "\n")
    calibration_path.write_text("\n".join(json.dumps(row) for row in calibration_rows) + "\n")
    return (
        cfg,
        pairs,
        cells,
        calibration_rows,
        cells_path,
        calibration_path,
        data_dir,
        variant_root,
        split_map,
    )


def _build(tmp_path: Path, *, stage_mode: str = "symlink"):
    fixture = _fixture(tmp_path)
    (
        cfg,
        pairs,
        cells,
        calibration_rows,
        cells_path,
        calibration_path,
        data_dir,
        variant_root,
        split_map,
    ) = fixture
    plan = adapter.build_pilot_plan(
        cfg=cfg,
        pairs=pairs,
        cells=cells,
        calibration_rows=calibration_rows,
        cells_path=cells_path,
        calibration_manifest_path=calibration_path,
        data_dir=data_dir,
        variant_root=variant_root,
        stage_subdir=Path("pilot_audio"),
        stage_mode=stage_mode,
    )
    return plan, fixture, split_map


def test_plan_keeps_train_calibration_disjoint_from_heldout_eval(tmp_path: Path):
    plan, _fixture_data, split_map = _build(tmp_path)
    calibration = [row for row in plan.records if row["source_role"] == "sarsteer_calibration"]
    evaluation = [
        row for row in plan.records if row["source_role"] == "run7_phase_frontend_pilot_eval"
    ]

    assert {row["item_id"] for row in calibration} == split_map["train"]
    assert {row["item_id"] for row in evaluation} == split_map["heldout"]
    assert not ({row["item_id"] for row in calibration} & {row["item_id"] for row in evaluation})
    assert len(calibration) == 2 * len(split_map["train"])
    assert len(evaluation) == 5 * len(split_map["heldout"])
    assert plan.metadata["split"]["leakage_check_passed"] is True
    assert all(row["transcript_control_passed"] is False for row in evaluation)


def test_plan_resolves_absolute_clean_and_relative_variants_into_data_paths(tmp_path: Path):
    plan, fixture, _split_map = _build(tmp_path)
    data_dir = fixture[6]
    evaluation = [
        row for row in plan.records if row["source_role"] == "run7_phase_frontend_pilot_eval"
    ]

    assert len(plan.stage_actions) == len(evaluation)
    assert all(not Path(row["path"]).is_absolute() for row in evaluation)
    assert all((data_dir / row["path"]).is_relative_to(data_dir) for row in evaluation)
    assert {row["style"] for row in evaluation} == {
        "neutral",
        "phase_pv_standard_m3",
        "phase_pv_standard_p3",
        "eq_mel_matched_ctrl_m3",
        "eq_mel_matched_ctrl_p3",
    }


def test_dry_run_preflight_writes_nothing(tmp_path: Path):
    plan, fixture, _split_map = _build(tmp_path)
    data_dir = fixture[6]
    output = data_dir / "manifests" / "pilot.jsonl"
    metadata = data_dir / "manifests" / "pilot.metadata.json"

    summary = adapter.materialize_plan(
        plan,
        output_path=output,
        metadata_path=metadata,
        stage_mode="symlink",
        dry_run=True,
        overwrite=False,
    )

    assert summary["leakage_check_passed"] is True
    assert summary["stage_new"] == len(plan.stage_actions)
    assert not output.exists()
    assert not metadata.exists()
    assert not (data_dir / "pilot_audio").exists()


def test_materialize_symlinks_and_manifest_is_idempotent_with_overwrite(tmp_path: Path):
    plan, fixture, _split_map = _build(tmp_path)
    data_dir = fixture[6]
    output = data_dir / "manifests" / "pilot.jsonl"
    metadata = data_dir / "manifests" / "pilot.metadata.json"

    adapter.materialize_plan(
        plan,
        output_path=output,
        metadata_path=metadata,
        stage_mode="symlink",
        dry_run=False,
        overwrite=False,
    )
    assert output.is_file()
    assert metadata.is_file()
    assert all(action.destination.is_symlink() for action in plan.stage_actions)

    summary = adapter.materialize_plan(
        plan,
        output_path=output,
        metadata_path=metadata,
        stage_mode="symlink",
        dry_run=False,
        overwrite=True,
    )
    assert summary["stage_reused"] == len(plan.stage_actions)


def test_missing_required_condition_fails_preflight(tmp_path: Path):
    fixture = _fixture(tmp_path)
    cfg, pairs, cells, calibration_rows = fixture[:4]
    cells_path, calibration_path, data_dir, variant_root = fixture[4:8]
    heldout = split_ids(pairs, cfg)["heldout"]
    missing_id = next(iter(heldout))
    cells = [
        row
        for row in cells
        if not (
            row["item_id"] == missing_id
            and row["condition"] == "pv_standard"
            and row["sign"] == 3.0
        )
    ]

    with pytest.raises(ValueError, match="missing required Run 7 cell"):
        adapter.build_pilot_plan(
            cfg=cfg,
            pairs=pairs,
            cells=cells,
            calibration_rows=calibration_rows,
            cells_path=cells_path,
            calibration_manifest_path=calibration_path,
            data_dir=data_dir,
            variant_root=variant_root,
            stage_subdir=Path("pilot_audio"),
            stage_mode="symlink",
        )


def test_committed_run9_pilot_config_loads():
    root = Path(__file__).parents[1]
    cfg = load_experiment_config(root / "configs" / "experiments" / "run9_defense_gate_pilot.yaml")
    assert cfg.name == "run9_defense_gate_run7_pilot"
    assert cfg.dataset.target_generation.outputs_file == Path(
        "manifests/run9_pilot_phase_frontend_outputs.jsonl"
    )
    assert cfg.dataset.styles == [
        "neutral",
        "phase_pv_standard_m3",
        "phase_pv_standard_p3",
        "eq_mel_matched_ctrl_m3",
        "eq_mel_matched_ctrl_p3",
    ]
    assert cfg.sarsteer is not None
    assert cfg.sarsteer.n_refusal_calib == 60
    assert cfg.sarsteer.n_benign_pca == 60
    assert cfg.sarsteer.max_new_tokens == 64
