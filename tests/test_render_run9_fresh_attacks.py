import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_renderer_script():
    path = Path(__file__).parents[1] / "scripts" / "render_run9_fresh_attacks.py"
    spec = importlib.util.spec_from_file_location("render_run9_fresh_attacks_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer_script()


def _clean_row(item_id: str, path: str, *, role: str = "harmful_eval"):
    return {
        "item_id": item_id,
        "category_id": 1 if role == "harmful_eval" else 8,
        "category_name": "category",
        "safety_label": "harmful" if role == "harmful_eval" else "benign",
        "gate_role": role,
        "condition": "clean",
        "path": path,
        "reference_text": f"question {item_id}",
        "reference_sha256": None,
    }


def test_plan_processes_only_existing_harmful_clean_wavs(tmp_path: Path):
    data_dir = tmp_path / "data"
    existing = data_dir / "clean" / "item_0.wav"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"wav")
    soft = data_dir / "clean" / "soft.wav"
    soft.write_bytes(b"soft")
    rows = [
        _clean_row("item_0", "clean/item_0.wav"),
        _clean_row("item_1", "clean/item_1.wav"),
        _clean_row("soft", "clean/soft.wav", role="soft_overrefusal"),
    ]

    plan = renderer.build_render_plan(
        rows,
        data_dir=data_dir,
        output_root=data_dir / "attacks",
        model_id="model",
        cache_dir=tmp_path / "cache",
        signs=(-3.0, 3.0),
    )

    assert plan.harmful_rows == 2
    assert plan.available_items == 1
    assert plan.missing_clean_items == ["item_1"]
    assert plan.ignored_rows == 1
    assert len(plan.tasks) == 2
    assert {task.sign for task in plan.tasks} == {-3.0, 3.0}


def test_plan_rejects_clean_path_outside_data_dir(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"wav")
    with pytest.raises(ValueError, match="escapes data_dir"):
        renderer.build_render_plan(
            [_clean_row("item", str(outside))],
            data_dir=data_dir,
            output_root=data_dir / "attacks",
            model_id="model",
            cache_dir=tmp_path / "cache",
        )


def test_render_tags_primary_phase_attack_and_writes_resumable_sidecar(tmp_path: Path, monkeypatch):
    import soundfile as sf

    data_dir = tmp_path / "data"
    clean = data_dir / "clean" / "item.wav"
    clean.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(4096, dtype=np.float32), 16000)
    plan = renderer.build_render_plan(
        [_clean_row("item", "clean/item.wav")],
        data_dir=data_dir,
        output_root=data_dir / "attacks",
        model_id="model",
        cache_dir=tmp_path / "cache",
        signs=(-3.0,),
    )
    task = plan.tasks[0]
    monkeypatch.setattr(
        renderer,
        "_load_audio",
        lambda path, sample_rate: np.zeros(4096, dtype=np.float32),
    )
    monkeypatch.setattr(
        renderer,
        "_compute_variants",
        lambda clean, sample_rate, sign, feature_extractor: {
            "waveforms": {
                "pv_standard": np.full(4096, 0.1, dtype=np.float32),
                "pv_locked": np.full(4096, 0.2, dtype=np.float32),
                "mel_matched_ctrl": np.full(4096, 0.3, dtype=np.float32),
            },
            "d_pair": 0.25,
            "mel_ctrl_realized_rms": 0.24,
            "mel_ctrl_strength": 1.5,
        },
    )

    rows = renderer._render_task_with_extractor(task, object(), overwrite=False)
    by_condition = {row["condition"]: row for row in rows}
    assert set(by_condition) == set(renderer.CONDITIONS)
    assert by_condition["pv_standard"]["phase_under_test"] is True
    assert by_condition["pv_standard"]["phase_under_test_tag"] == ("run7_frozen_pv_standard")
    assert by_condition["pv_standard"]["almguard_training_policy"].startswith("exclude")
    assert by_condition["pv_locked"]["route"] == "phase_repaired_negative_twin"
    assert by_condition["mel_matched_ctrl"]["route"] == "eq_other_route"
    assert by_condition["mel_matched_ctrl"]["mel_ctrl_realized_rms"] == 0.24

    pending, resumed = renderer.classify_tasks([task], overwrite=False)
    assert pending == []
    assert len(resumed) == 3


def test_incomplete_artifact_requires_explicit_overwrite(tmp_path: Path):
    data_dir = tmp_path / "data"
    clean = data_dir / "clean" / "item.wav"
    clean.parent.mkdir(parents=True)
    clean.write_bytes(b"wav")
    task = renderer.build_render_plan(
        [_clean_row("item", "clean/item.wav")],
        data_dir=data_dir,
        output_root=data_dir / "attacks",
        model_id="model",
        cache_dir=tmp_path / "cache",
        signs=(-3.0,),
    ).tasks[0]
    output = renderer._output_paths(task)["pv_standard"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"partial")

    with pytest.raises(FileExistsError, match="--overwrite"):
        renderer.classify_tasks([task], overwrite=False)
    pending, resumed = renderer.classify_tasks([task], overwrite=True)
    assert pending == [task]
    assert resumed == []


def test_renderer_source_never_imports_qwen_model_weights():
    source = (Path(__file__).parents[1] / "scripts" / "render_run9_fresh_attacks.py").read_text()
    assert "Qwen2AudioForConditionalGeneration" not in source
    assert "AutoProcessor.from_pretrained" in source
