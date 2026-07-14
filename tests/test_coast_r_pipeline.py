"""CPU-only preflight tests for the Run 7 COAST-R orchestration."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.coast_r import (
    _layer_view,
    _run_crossfit_fit,
    load_coast_r_source,
)
from audio_safety.utils.io import save_jsonl
from audio_safety.utils.paths import ResolvedPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN7_CONFIG = REPO_ROOT / "configs" / "experiments" / "run7_coast_r_stage_a.yaml"


def _load_run7_script():
    path = REPO_ROOT / "scripts" / "run_coast_r_stage_a.py"
    spec = importlib.util.spec_from_file_location("run_coast_r_stage_a_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run7_script = _load_run7_script()


def _config(source_run_name: str = "source_run"):
    cfg = load_experiment_config(RUN7_CONFIG)
    assert cfg.coast_r is not None
    coast = cfg.coast_r.model_copy(update={"source_run_name": source_run_name})
    return cfg.model_copy(update={"coast_r": coast})


def _paths(tmp_path: Path) -> ResolvedPaths:
    return ResolvedPaths(
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )


def _write_source(tmp_path: Path, cells: list[dict]) -> tuple[Path, Path]:
    source = tmp_path / "outputs" / "source_run"
    activations = source / "pitch_representation" / "activations.npz"
    cells_path = source / "pitch_representation" / "cells.jsonl"
    activations.parent.mkdir(parents=True)
    np.savez(
        activations,
        llm_p2=np.zeros((len(cells), 3, 2), dtype=np.float32),
        llm_layers=np.array([16, 18, 20], dtype=np.int16),
    )
    save_jsonl(cells, cells_path)
    return activations, cells_path


def test_source_preflight_resolves_variant_and_data_audio_and_records_hashes(tmp_path):
    variant = tmp_path / "outputs" / "source_run" / "audio" / "variant.wav"
    original = tmp_path / "data" / "audio" / "original.wav"
    variant.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    variant.write_bytes(b"variant")
    original.write_bytes(b"original")
    activations, cells_path = _write_source(
        tmp_path,
        [
            {"activation_index": 0, "variant_path": "audio/variant.wav"},
            {"activation_index": 1, "source_path": "audio/original.wav"},
        ],
    )

    source = load_coast_r_source(_config(), _paths(tmp_path), require_audio=True)

    assert source.audio_paths == {0: variant, 1: original}
    assert source.activations_sha256 == hashlib.sha256(activations.read_bytes()).hexdigest()
    assert source.cells_sha256 == hashlib.sha256(cells_path.read_bytes()).hexdigest()
    assert source.hashes == {
        "activations_sha256": source.activations_sha256,
        "cells_sha256": source.cells_sha256,
    }


def test_source_preflight_can_skip_audio_presence_for_cpu_fit(tmp_path):
    _write_source(
        tmp_path,
        [{"activation_index": 0, "variant_path": "audio/not_local.wav"}],
    )

    source = load_coast_r_source(_config(), _paths(tmp_path), require_audio=False)

    assert source.audio_paths[0] == tmp_path / "outputs/source_run/audio/not_local.wav"
    assert not source.audio_paths[0].exists()


def test_source_preflight_rejects_noncontiguous_activation_indices(tmp_path):
    _write_source(
        tmp_path,
        [{"activation_index": 1, "variant_path": "audio/not_local.wav"}],
    )

    with pytest.raises(ValueError, match="contiguous activation_index"):
        load_coast_r_source(_config(), _paths(tmp_path), require_audio=False)


def test_layer_view_uses_explicit_layer_ids_and_rejects_absent_layer():
    cfg = _config().coast_r
    assert cfg is not None
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    arrays = {
        "llm_layers": np.array([16, 18, 20], dtype=np.int16),
        "llm_p2": values,
    }

    np.testing.assert_array_equal(_layer_view(arrays, cfg, 18), values[:, 1, :])
    with pytest.raises(ValueError, match="layer 17 is absent"):
        _layer_view(arrays, cfg, 17)


def test_run7_snapshot_is_reused_but_rejects_phase_to_phase_config_drift(tmp_path):
    cfg = _config()
    first = run7_script._ensure_snapshot(cfg, tmp_path)
    second = run7_script._ensure_snapshot(cfg, tmp_path)

    assert first == second == tmp_path / "config_snapshot.yaml"
    drifted = cfg.model_copy(update={"seed": cfg.seed + 1})
    with pytest.raises(RuntimeError, match="differs from the frozen snapshot"):
        run7_script._ensure_snapshot(drifted, tmp_path)


def test_crossfit_fit_end_to_end_keeps_roles_and_frozen_vectors_aligned(monkeypatch):
    """Exercise the complete CPU fit with all declared folds, roles, and arms."""
    import audio_safety.evaluation.coast_r as coast_r_core

    n_items = 20
    layers = np.asarray([16, 18, 20], dtype=np.int16)
    hidden_dim = 8
    layer_scales = np.asarray([0.9, 1.0, 1.1], dtype=np.float64)
    item_coordinates = np.linspace(-0.95, 0.95, n_items)
    cells: list[dict] = []
    states: list[np.ndarray] = []
    score_rows: list[dict] = []

    for item_offset, item_coordinate in enumerate(item_coordinates):
        item_id = f"item-{item_offset:02d}"
        neutral = np.asarray(
            [
                item_coordinate,
                item_coordinate**2,
                item_coordinate**3,
                0.0,
                0.1 * item_coordinate,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        endpoint_origin = np.asarray(
            [
                0.05 * item_coordinate,
                -0.03 * item_coordinate,
                0.02 * item_coordinate**2,
                -0.01 * item_coordinate**3,
            ],
            dtype=np.float64,
        )
        for safety_label in ("harmful", "benign"):
            arm_offset = np.zeros(hidden_dim, dtype=np.float64)
            arm_offset[5] = 0.25 if safety_label == "benign" else -0.25
            for severity in (-1.0, 0.0, 1.0):
                delta = np.zeros(hidden_dim, dtype=np.float64)
                if severity != 0.0:
                    delta[:4] = (
                        severity,
                        0.5 * item_coordinate,
                        0.3 * item_coordinate**2,
                        0.2 * item_coordinate**3,
                    )
                activation_index = len(cells)
                behavior = "benign_answer"
                if safety_label == "harmful":
                    behavior = "harmful_compliance" if severity > 0.0 else "policy_refusal"
                cells.append(
                    {
                        "activation_index": activation_index,
                        "item_id": item_id,
                        "safety_label": safety_label,
                        "pitch_semitones": severity,
                        "reviewed_behavior_label": behavior,
                        "reference_text": f"{item_id}:{safety_label}:{severity:+.0f}",
                    }
                )
                states.append(
                    np.stack([neutral + arm_offset + scale * delta for scale in layer_scales])
                )
                score_rows.append(
                    {
                        "activation_index": activation_index,
                        "continuation_curve": (endpoint_origin + delta[:4]).tolist(),
                    }
                )

    cfg = _config().coast_r
    assert cfg is not None
    gate = cfg.model_copy(
        update={
            "max_continuation_tokens": 4,
            "reachable_rank_candidates": [4],
            "reachable_rank_cap": 4,
            "reachable_min_coverage": 0.0,
            "transport_ranks": [1, 2, 3, 4],
            "ridge_alphas": [1e-6],
        }
    )
    arrays = {
        "llm_layers": layers,
        "llm_p2": np.stack(states).astype(np.float32),
    }

    predictor_fit_rows: list[int] = []
    original_fit_predictor = coast_r_core.fit_natural_predictor

    def recording_fit_predictor(neutral_states, severity_features, deltas, reachable, **kwargs):
        predictor_fit_rows.append(len(deltas))
        return original_fit_predictor(
            neutral_states,
            severity_features,
            deltas,
            reachable,
            **kwargs,
        )

    monkeypatch.setattr(
        coast_r_core,
        "fit_natural_predictor",
        recording_fit_predictor,
    )
    result = _run_crossfit_fit(arrays, cells, score_rows, gate)

    manifest = result["intervention_manifest"]
    vectors = result["intervention_vectors"]
    selected_manifest = [row for row in manifest if row["selected_for_intervention"]]
    unavailable_manifest = [row for row in manifest if not row["selected_for_intervention"]]
    assert manifest
    assert [row["vector_index"] for row in selected_manifest] == list(range(len(selected_manifest)))
    assert all(row["transport_rank_status"] == "available" for row in selected_manifest)
    assert all(row["vector_index"] is None for row in unavailable_manifest)
    assert all(row["transport_rank_status"] == "not_available" for row in unavailable_manifest)
    assert set(vectors) == {"delta_u", "delta_perp", "delta_full", "delta_predicted"}
    assert {value.shape for value in vectors.values()} == {(len(selected_manifest), hidden_dim)}
    assert result["metrics"]["n_intervention_rows"] == len(manifest)
    assert result["metrics"]["n_intervention_vector_rows"] == len(selected_manifest)

    folds = result["metrics"]["layers"][str(gate.primary_layer)]["folds"]
    assert len(folds) == gate.outer_folds == 5
    manifest_keys: list[tuple[int, int]] = []
    for fold_row in folds:
        fold = int(fold_row["fold"])
        primary_rotation = fold_row["role_rotations"][0]
        available_ranks = set(primary_rotation["available_transport_ranks"])
        fold_manifest = [row for row in manifest if int(row["fold"]) == fold]
        assert fold_manifest
        assert {int(row["rank"]) for row in fold_manifest} == set(gate.transport_ranks)
        for activation_index in {int(row["transformed_activation_index"]) for row in fold_manifest}:
            rows = [
                row
                for row in fold_manifest
                if int(row["transformed_activation_index"]) == activation_index
            ]
            assert {int(row["rank"]) for row in rows} == set(gate.transport_ranks)
            selected_rows = [row for row in rows if row["selected_for_intervention"]]
            assert {int(row["rank"]) for row in selected_rows} == available_ranks
            assert len(selected_rows) == len(available_ranks)
            manifest_keys.extend((activation_index, int(row["rank"])) for row in selected_rows)

        for rotation in fold_row["role_rotations"]:
            assignment = rotation["role_assignment"]
            role_groups = [set(assignment[f"{role}_groups"]) for role in ("b", "r", "u", "f")]
            assert all(
                left.isdisjoint(right)
                for offset, left in enumerate(role_groups)
                for right in role_groups[offset + 1 :]
            )
            assert len(set().union(*role_groups)) == fold_row["n_train_items"]

    assert len(manifest_keys) == len(set(manifest_keys))
    harmful_transformed = {
        int(cell["activation_index"])
        for cell in cells
        if cell["safety_label"] == "harmful" and cell["pitch_semitones"] != 0.0
    }
    assert {activation_index for activation_index, _ in manifest_keys} == harmful_transformed

    # Four f-role items x two arms x two non-neutral severities. A harmful-only
    # predictor would see only eight rows, so this spies on the actual fit input.
    assert len(predictor_fit_rows) == len(layers) * gate.outer_folds * 2
    assert set(predictor_fit_rows) == {16}
