import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_run9_fresh_attacks.py"
    spec = importlib.util.spec_from_file_location("validate_run9_fresh_attacks_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_grid_rejects_duplicate_stable_key():
    row = {
        "item_id": "item",
        "condition": "pv_standard",
        "sign": -3.0,
        "operator_version": validator.EXPECTED_OPERATOR_VERSION,
        "processor_model_id": validator.EXPECTED_PROCESSOR,
        "sample_rate": 16000,
        "render_status": "rendered",
        "safety_label": "harmful",
        "gate_role": "harmful_eval",
        "asr_required": True,
        **validator.EXPECTED_TAGS["pv_standard"],
        "reference_text": "secret",
        "reference_sha256": hashlib.sha256(b"secret").hexdigest(),
    }
    issues = validator.Issues()

    validator._validate_grid([row, row], expected_items=1, issues=issues)

    assert issues.error_counts["duplicate_stable_key"] == 1
    assert issues.error_counts["row_count_mismatch"] == 1


def test_full_fixture_passes_without_prompt_text_in_report(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    docs_root = repo_root / "docs/experiments/exp1_refusal_cone_drift"
    data_dir = tmp_path / "data"
    attack_root = data_dir / "audio_run9" / "attacks"
    clean_path = data_dir / "clean" / "item.wav"
    clean_path.parent.mkdir(parents=True)
    t = np.linspace(0, 1, 1600, endpoint=False)
    sf.write(clean_path, 0.1 * np.sin(2 * np.pi * 220 * t), 24000)
    reference_text = "SECRET PROMPT MUST NOT APPEAR"
    reference_sha = hashlib.sha256(reference_text.encode()).hexdigest()
    clean_row = {
        "item_id": "item",
        "source": "figstep_safebench",
        "category_id": 1,
        "category_name": "category",
        "safety_label": "harmful",
        "gate_role": "harmful_eval",
        "condition": "clean",
        "path": "clean/item.wav",
        "reference_text": reference_text,
        "reference_sha256": reference_sha,
    }
    clean_manifest = data_dir / "manifests" / "clean.jsonl"
    _write_jsonl(clean_manifest, [clean_row])

    rows: list[dict] = []
    run7_rows: list[dict] = []
    run7_root = tmp_path / "run7"
    for sign_index, sign in enumerate(validator.EXPECTED_SIGNS):
        group: list[dict] = []
        for condition_index, condition in enumerate(validator.EXPECTED_CONDITIONS):
            tag = validator._sign_tag(sign)
            output = attack_root / condition / tag / "item.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            frequency = 300 + sign_index * 100 + condition_index * 25
            waveform = (0.02 + condition_index * 0.01) * np.sin(2 * np.pi * frequency * t)
            sf.write(output, waveform, 16000)
            d_pair = 0.2 + sign_index * 0.01
            row = {
                "item_id": "item",
                "source": "figstep_safebench",
                "category_id": 1,
                "category_name": "category",
                "safety_label": "harmful",
                "gate_role": "harmful_eval",
                "condition": condition,
                "sign": sign,
                **validator.EXPECTED_TAGS[condition],
                "path": output.relative_to(data_dir).as_posix(),
                "output_sha256": _digest(output),
                "reference_text": reference_text,
                "reference_sha256": reference_sha,
                "source_clean_path": "clean/item.wav",
                "source_clean_sha256": _digest(clean_path),
                "source_text_hash_verified": False,
                "operator_version": validator.EXPECTED_OPERATOR_VERSION,
                "processor_model_id": validator.EXPECTED_PROCESSOR,
                "sample_rate": 16000,
                "d_pair": d_pair,
                "mel_ctrl_realized_rms": d_pair if condition == "mel_matched_ctrl" else None,
                "mel_ctrl_strength": 1.0 if condition == "mel_matched_ctrl" else None,
                "render_status": "rendered",
                "asr_required": True,
            }
            rows.append(row)
            group.append(row)
            old_path = run7_root / "audio" / f"{tag}_{condition}.wav"
            old_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, old_path)
            run7_rows.append(
                {
                    "item_id": "item",
                    "condition": condition,
                    "sign": sign,
                    "variant_path": old_path.relative_to(run7_root).as_posix(),
                    "d_pair": d_pair,
                    "mel_ctrl_realized_rms": (d_pair if condition == "mel_matched_ctrl" else None),
                }
            )
        sidecar = attack_root / "_metadata" / f"item_{validator._sign_tag(sign)}.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "identity": {
                        "operator_version": validator.EXPECTED_OPERATOR_VERSION,
                        "item_id": "item",
                        "sign": sign,
                        "source_clean_sha256": _digest(clean_path),
                        "processor_model_id": validator.EXPECTED_PROCESSOR,
                        "sample_rate": 16000,
                    },
                    "rows": group,
                }
            ),
            encoding="utf-8",
        )

    manifest = data_dir / "manifests" / "attacks.jsonl"
    cells = run7_root / "cells.jsonl"
    _write_jsonl(manifest, rows)
    _write_jsonl(cells, run7_rows)
    report = validator.validate(
        manifest=manifest,
        clean_manifest=clean_manifest,
        data_dir=data_dir,
        attack_root=attack_root,
        run7_cells=cells,
        run7_root=run7_root,
        direction_doc=docs_root / "run9_advisor_defense_gate_direction_20260717.md",
        run7_doc=docs_root / "run7_phase_frontend_distortion_direction_20260714.md",
        repo_root=repo_root,
        expected_items=1,
        workers=2,
    )

    assert report["status"] == "pass"
    assert report["attack_audio"]["unique_paths_checked"] == 6
    assert report["sidecars"]["checked"] == 2
    assert report["frozen_operator_audit"]["run7_byte_exact_matches"] == 6
    assert "SECRET PROMPT" not in json.dumps(report)
