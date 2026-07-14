"""Run 7 Stage-A COAST-R orchestration.

The source Run 6 directory is immutable input.  This module reads its pooled
activations, reviewed cells, and already-rendered waveforms, then writes every
new artifact under a separate Run 7 directory.  Model-heavy imports stay inside
the GPU phase functions so fitting and tests remain CPU-only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from audio_safety.config.schema import CoastRConfig, ExperimentConfig
from audio_safety.utils.io import load_json, load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import ResolvedPaths


@dataclass(frozen=True)
class CoastRSource:
    """Validated, read-only pointers to the exposed Run 6 artifacts."""

    run_dir: Path
    activations_path: Path
    cells_path: Path
    cells: list[dict[str, Any]]
    audio_paths: dict[int, Path]
    activations_sha256: str
    cells_sha256: str

    @property
    def hashes(self) -> dict[str, str]:
        return {
            "activations_sha256": self.activations_sha256,
            "cells_sha256": self.cells_sha256,
        }


def require_coast_r_config(cfg: ExperimentConfig) -> CoastRConfig:
    gate = cfg.coast_r
    if gate is None or not gate.enabled:
        raise ValueError("config must contain an enabled `coast_r` block")
    return gate


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_hashes(
    actual: object,
    source: CoastRSource,
    *,
    context: str,
) -> None:
    """Fail closed when a resumable artifact belongs to another source run."""
    if not isinstance(actual, dict):
        raise RuntimeError(f"{context} has no source-hash mapping")
    mismatches = {
        name: {"expected": expected, "found": actual.get(name)}
        for name, expected in source.hashes.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{context} does not match the current Run 6 source hashes: {mismatches}"
        )


def _validate_resumable_rows(
    rows: list[dict[str, Any]],
    source: CoastRSource,
    *,
    id_field: str,
    context: str,
) -> None:
    """Validate both provenance and unique resume keys before appending JSONL."""
    seen: set[str] = set()
    for offset, row in enumerate(rows):
        if id_field not in row:
            raise RuntimeError(f"{context} row {offset} has no {id_field!r}")
        row_id = str(row[id_field])
        if row_id in seen:
            raise RuntimeError(f"{context} contains duplicate {id_field}={row_id!r}")
        seen.add(row_id)
        _require_source_hashes(row, source, context=f"{context} row {row_id}")


def _absolute_or_under(path_value: object, root: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def _cell_audio_path(
    cell: dict[str, Any],
    *,
    source_run_dir: Path,
    data_dir: Path,
) -> Path:
    variant = cell.get("variant_path")
    if variant:
        return _absolute_or_under(variant, source_run_dir)
    source = cell.get("source_path") or cell.get("path")
    if not source:
        raise KeyError(
            f"cell {cell.get('activation_index')} has neither variant_path nor source_path"
        )
    return _absolute_or_under(source, data_dir)


def load_coast_r_source(
    cfg: ExperimentConfig,
    paths: ResolvedPaths,
    *,
    require_audio: bool,
) -> CoastRSource:
    """Validate and hash the source run without changing it."""
    gate = require_coast_r_config(cfg)
    source_run_dir = paths.output_dir / gate.source_run_name
    activations_path = source_run_dir / gate.source_activations_file
    cells_path = source_run_dir / gate.source_cells_file
    missing_artifacts = [str(path) for path in (activations_path, cells_path) if not path.is_file()]
    if missing_artifacts:
        joined = "\n  - ".join(missing_artifacts)
        raise FileNotFoundError(f"Run 6 source artifacts are missing:\n  - {joined}")

    cells = load_jsonl(cells_path)
    indices = [int(cell["activation_index"]) for cell in cells]
    if indices != list(range(len(cells))):
        raise ValueError("source cells must be ordered by contiguous activation_index")

    audio_paths: dict[int, Path] = {}
    missing_audio: list[Path] = []
    for cell in cells:
        index = int(cell["activation_index"])
        audio_path = _cell_audio_path(
            cell,
            source_run_dir=source_run_dir,
            data_dir=paths.data_dir,
        )
        audio_paths[index] = audio_path
        if require_audio and not audio_path.is_file():
            missing_audio.append(audio_path)
    if require_audio and missing_audio:
        examples = "\n  - ".join(str(path) for path in missing_audio[:10])
        more = f"\n  ... and {len(missing_audio) - 10} more" if len(missing_audio) > 10 else ""
        raise FileNotFoundError(
            "COAST-R reuses the Run 6 waveforms, but "
            f"{len(missing_audio)}/{len(cells)} are missing:\n  - {examples}{more}"
        )

    return CoastRSource(
        run_dir=source_run_dir,
        activations_path=activations_path,
        cells_path=cells_path,
        cells=cells,
        audio_paths=audio_paths,
        activations_sha256=_sha256(activations_path),
        cells_sha256=_sha256(cells_path),
    )


def _score_key(row: dict[str, Any]) -> int:
    return int(row["activation_index"])


def score_continuation_bank(
    model: Any,
    processor: Any,
    cfg: ExperimentConfig,
    source: CoastRSource,
    run_dir: Path,
) -> dict[str, Any]:
    """GPU phase: score the frozen refusal/compliance continuation bank.

    The JSONL is resumable by ``activation_index``.  It stores the raw token
    log-probabilities as well as the aggregated curve so later analyses can
    change plots without rerunning the model.
    """
    from audio_safety.evaluation.continuation_scores import refusal_compliance_curve
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        resolve_audio_position_indices,
        score_audio_continuation,
    )

    gate = require_coast_r_config(cfg)
    output_path = run_dir / gate.continuation_scores_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gate.overwrite and output_path.exists():
        output_path.unlink()
    existing = load_jsonl(output_path) if output_path.exists() else []
    _validate_resumable_rows(
        existing,
        source,
        id_field="activation_index",
        context="continuation score artifact",
    )
    done = {_score_key(row) for row in existing}
    allowed = set(gate.score_safety_labels)
    selected = [cell for cell in source.cells if str(cell["safety_label"]) in allowed]

    written = 0
    with output_path.open("a") as handle:
        for cell in tqdm(selected, desc="COAST-R continuation scores", unit="cell"):
            activation_index = int(cell["activation_index"])
            audio_path = source.audio_paths[activation_index]
            conversation = build_audio_analysis_conversation(
                audio_path,
                cfg.dataset.target_generation.instruction,
                system_prompt=gate.system_prompt,
            )
            resolved_p2 = int(
                resolve_audio_position_indices(processor, conversation)["first_generation_prelogit"]
            )
            saved_p2 = cell.get("p2_index")
            if saved_p2 is None or int(saved_p2) != resolved_p2:
                raise RuntimeError(
                    "current official chat template does not reproduce the saved P2 index "
                    f"for activation_index={activation_index}: saved={saved_p2}, "
                    f"resolved={resolved_p2}"
                )
            if activation_index in done:
                continue
            refusal_rows = [
                score_audio_continuation(
                    model,
                    processor,
                    audio_path,
                    cfg.dataset.target_generation.instruction,
                    target,
                    max_tokens=gate.max_continuation_tokens,
                    system_prompt=gate.system_prompt,
                )
                for target in gate.refusal_continuations
            ]
            compliance_rows = [
                score_audio_continuation(
                    model,
                    processor,
                    audio_path,
                    cfg.dataset.target_generation.instruction,
                    target,
                    max_tokens=gate.max_continuation_tokens,
                    system_prompt=gate.system_prompt,
                )
                for target in gate.compliance_continuations
            ]
            curve = refusal_compliance_curve(
                refusal_rows,
                compliance_rows,
                steps=gate.max_continuation_tokens,
            )
            record = {
                "activation_index": activation_index,
                "item_id": str(cell["item_id"]),
                "safety_label": str(cell["safety_label"]),
                "operator_family": gate.operator_family,
                "operator_backend": gate.operator_backend,
                "severity": float(cell[gate.severity_field]),
                "audio_path": str(audio_path),
                "p1_index": cell.get("p1_index"),
                "p2_index": cell.get("p2_index"),
                "resolved_p2_index": resolved_p2,
                "resolved_position_name": "first_generation_prelogit",
                "prompt_length": cell.get("prompt_length"),
                "first_token_margin": cell.get("refusal_margin"),
                "refusal_targets": [
                    {"text": target, "token_log_probs": scores}
                    for target, scores in zip(gate.refusal_continuations, refusal_rows, strict=True)
                ],
                "compliance_targets": [
                    {"text": target, "token_log_probs": scores}
                    for target, scores in zip(
                        gate.compliance_continuations, compliance_rows, strict=True
                    )
                ],
                **curve,
                **source.hashes,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(activation_index)
            written += 1

    return {
        "output": str(output_path),
        "n_selected": len(selected),
        "n_preexisting": len(existing),
        "n_written": written,
        "n_complete": len(done.intersection({_score_key(cell) for cell in selected})),
        "source": source.hashes,
    }


def _layer_view(arrays: dict[str, np.ndarray], gate: CoastRConfig, layer: int) -> np.ndarray:
    """Map a configured decoder layer through the archive's explicit layer list."""
    layer_ids = [int(value) for value in np.asarray(arrays["llm_layers"]).reshape(-1)]
    if layer not in layer_ids:
        raise ValueError(f"configured layer {layer} is absent from source llm_layers={layer_ids}")
    # Select the one requested layer before widening to float64.  The source site
    # is roughly a gigabyte for n=150 x 32 decoder layers; widening the full cube
    # would transiently allocate another ~2 GB even though the fit uses one layer.
    values = np.asarray(arrays[gate.site])
    if values.ndim != 3 or values.shape[1] != len(layer_ids):
        raise ValueError(
            f"site {gate.site!r} must have shape (cell, layer, dim), got {values.shape}"
        )
    return np.asarray(values[:, layer_ids.index(layer), :], dtype=np.float64)


def _cell_index(
    cells: list[dict[str, Any]], gate: CoastRConfig
) -> dict[tuple[str, str, float], int]:
    index: dict[tuple[str, str, float], int] = {}
    for cell in cells:
        key = (
            str(cell["item_id"]),
            str(cell["safety_label"]),
            round(float(cell[gate.severity_field]), 8),
        )
        if key in index:
            raise ValueError(f"duplicate source cell {key}")
        index[key] = int(cell["activation_index"])
    return index


def _delta_metadata(cells: list[dict[str, Any]], gate: CoastRConfig) -> list[dict[str, Any]]:
    index = _cell_index(cells, gate)
    neutral = round(float(gate.neutral_severity), 8)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        severity = round(float(cell[gate.severity_field]), 8)
        if severity == neutral:
            continue
        item_id = str(cell["item_id"])
        safety_label = str(cell["safety_label"])
        neutral_index = index.get((item_id, safety_label, neutral))
        if neutral_index is None:
            raise ValueError(f"missing neutral cell for {(item_id, safety_label)}")
        rows.append(
            {
                "item_id": item_id,
                "safety_label": safety_label,
                "severity": severity,
                "transformed_activation_index": int(cell["activation_index"]),
                "neutral_activation_index": neutral_index,
            }
        )
    if not rows:
        raise ValueError("no non-neutral source cells were found")
    return rows


def _severity_features(rows: list[dict[str, Any]]) -> np.ndarray:
    severity = np.asarray([float(row["severity"]) for row in rows], dtype=np.float64)
    return np.column_stack((severity, np.abs(severity), np.square(severity)))


def _endpoint_by_activation(
    cells: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    gate: CoastRConfig,
) -> dict[int, np.ndarray]:
    if gate.endpoint_kind == "first_token_baseline":
        return {
            int(cell["activation_index"]): np.asarray(
                [float(cell["refusal_margin"])], dtype=np.float64
            )
            for cell in cells
        }
    endpoints: dict[int, np.ndarray] = {}
    for row in score_rows:
        curve = np.asarray(row["continuation_curve"], dtype=np.float64).reshape(-1)
        if len(curve) != gate.max_continuation_tokens or not np.isfinite(curve).all():
            raise ValueError(
                f"invalid continuation curve for activation {row.get('activation_index')}"
            )
        endpoints[int(row["activation_index"])] = curve
    return endpoints


def _endpoint_deltas(
    rows: list[dict[str, Any]],
    endpoints_by_index: dict[int, np.ndarray],
    *,
    arm_name: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        transformed = int(row["transformed_activation_index"])
        neutral = int(row["neutral_activation_index"])
        if transformed not in endpoints_by_index or neutral not in endpoints_by_index:
            raise ValueError(
                f"continuation score coverage is incomplete for {arm_name} delta row "
                f"{row['item_id']} severity={row['severity']}"
            )
        values.append(endpoints_by_index[transformed] - endpoints_by_index[neutral])
    if not values:
        raise ValueError(f"no {arm_name} endpoint delta rows were available")
    return np.stack(values)


def _prefix_artifacts(destination: dict[str, np.ndarray], prefix: str, fitted: Any) -> None:
    if not hasattr(fitted, "artifact_arrays"):
        return
    for name, value in fitted.artifact_arrays().items():
        destination[f"{prefix}_{name}"] = np.asarray(value)


def _json_metrics(fitted: Any) -> dict[str, Any]:
    return dict(fitted.metrics()) if hasattr(fitted, "metrics") else {}


def _operator_mean_prediction(
    train_delta: np.ndarray,
    train_severity: np.ndarray,
    test_severity: np.ndarray,
) -> np.ndarray:
    global_mean = train_delta.mean(axis=0)
    means: dict[tuple[float, ...], np.ndarray] = {}
    for key in {tuple(row.tolist()) for row in train_severity}:
        mask = np.all(train_severity == np.asarray(key), axis=1)
        means[key] = train_delta[mask].mean(axis=0)
    return np.stack([means.get(tuple(row.tolist()), global_mean) for row in test_severity])


def _predictor_gate(metrics: dict[str, Any], threshold: float) -> bool:
    for key in (
        "relative_improvement",
        "cv_relative_improvement",
        "cv_improvement_over_mean",
        "mse_improvement_over_mean",
    ):
        if metrics.get(key) is not None:
            # "Beats the operator/severity mean" is strict.  With the default
            # threshold 0, an exact tie must not unlock out-of-pair induction.
            return float(metrics[key]) > threshold
    raise KeyError(
        "NaturalCoordinatePredictor.metrics() must expose validation improvement "
        "relative to its mean baseline"
    )


def _run_crossfit_fit(
    arrays: dict[str, np.ndarray],
    cells: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    gate: CoastRConfig,
) -> dict[str, Any]:
    """Compose the CPU primitives in :mod:`audio_safety.evaluation.coast_r`."""
    from audio_safety.evaluation.coast_r import (
        basis_coordinates,
        deterministic_group_folds,
        fit_dim_refusal_basis,
        fit_natural_predictor,
        fit_reachable_basis,
        fit_reduced_rank_transport,
        make_disjoint_role_rotations,
        orthogonal_residual,
        project_onto_basis,
    )

    metadata = _delta_metadata(cells, gate)
    endpoints_by_index = _endpoint_by_activation(cells, score_rows, gate)
    harmful_positions = np.asarray(
        [i for i, row in enumerate(metadata) if row["safety_label"] == "harmful"],
        dtype=np.int64,
    )
    if len(harmful_positions) == 0:
        raise ValueError("COAST-R fit found no harmful non-neutral delta rows")
    harmful_rows = [metadata[int(i)] for i in harmful_positions]
    harmful_groups = np.asarray([row["item_id"] for row in harmful_rows], dtype=object)
    endpoint_delta = _endpoint_deltas(
        harmful_rows,
        endpoints_by_index,
        arm_name="harmful",
    )
    severity = _severity_features(harmful_rows)
    benign_positions = np.asarray(
        [i for i, row in enumerate(metadata) if row["safety_label"] == "benign"],
        dtype=np.int64,
    )
    benign_rows = [metadata[int(i)] for i in benign_positions]
    benign_endpoint_delta: np.ndarray | None = None
    benign_endpoint_requested = gate.endpoint_kind == "first_token_baseline" or "benign" in set(
        gate.score_safety_labels
    )
    if benign_endpoint_requested:
        benign_endpoint_delta = _endpoint_deltas(
            benign_rows,
            endpoints_by_index,
            arm_name="benign",
        )
    outer = deterministic_group_folds(harmful_groups, gate.outer_folds, seed=gate.seed)
    if len(outer) < 2:
        raise ValueError("COAST-R requires at least two non-empty outer folds")

    cells_by_index = {int(cell["activation_index"]): cell for cell in cells}
    layers = list(dict.fromkeys([gate.primary_layer, *gate.sensitivity_layers]))
    max_transport_rank = max(int(rank) for rank in gate.transport_ranks)
    artifact_arrays: dict[str, np.ndarray] = {}
    layer_metrics: dict[str, Any] = {}
    intervention_manifest: list[dict[str, Any]] = []
    vectors: dict[str, list[np.ndarray]] = {
        "delta_u": [],
        "delta_perp": [],
        "delta_full": [],
        "delta_predicted": [],
    }

    for layer in layers:
        states = _layer_view(arrays, gate, layer)
        if len(states) != len(cells):
            raise ValueError(
                f"source site {gate.site!r} has {len(states)} rows for {len(cells)} cells"
            )
        all_delta = np.stack(
            [
                states[int(row["transformed_activation_index"])]
                - states[int(row["neutral_activation_index"])]
                for row in metadata
            ]
        )
        all_groups = np.asarray([row["item_id"] for row in metadata], dtype=object)
        all_neutral_states = np.stack(
            [states[int(row["neutral_activation_index"])] for row in metadata]
        )
        all_severity = _severity_features(metadata)
        harmful_delta = all_delta[harmful_positions]
        benign_delta = all_delta[benign_positions]
        neutral_states = np.stack(
            [states[int(row["neutral_activation_index"])] for row in harmful_rows]
        )
        fold_metrics: list[dict[str, Any]] = []

        for fold, (train, test) in enumerate(outer):
            train_items = set(harmful_groups[train].tolist())
            test_items = set(harmful_groups[test].tolist())
            benign_test = np.asarray(
                [i for i, row in enumerate(benign_rows) if row["item_id"] in test_items],
                dtype=np.int64,
            )
            if benign_endpoint_delta is not None and len(benign_test) == 0:
                raise ValueError(f"outer fold {fold} has no matched held-out benign rows")
            rotations = make_disjoint_role_rotations(
                np.asarray(train, dtype=np.int64),
                harmful_groups,
                seed=gate.seed + fold,
            )
            rotation_metrics: list[dict[str, Any]] = []
            primary_bundle: dict[str, Any] | None = None
            for rotation_index, assignment in enumerate(rotations):
                b_indices = np.asarray(assignment.b_indices, dtype=np.int64)
                u_indices = np.asarray(assignment.u_indices, dtype=np.int64)
                r_indices = np.asarray(assignment.r_indices, dtype=np.int64)
                f_indices = np.asarray(assignment.f_indices, dtype=np.int64)
                role_indices = {
                    "b": b_indices,
                    "r": r_indices,
                    "u": u_indices,
                    "f": f_indices,
                }
                role_items = {
                    name: set(harmful_groups[indices].tolist())
                    for name, indices in role_indices.items()
                }
                role_names = list(role_items)
                for left_offset, left in enumerate(role_names):
                    for right in role_names[left_offset + 1 :]:
                        overlap = role_items[left].intersection(role_items[right])
                        if overlap:
                            raise RuntimeError(
                                f"COAST-R role leakage between {left}/{right}: "
                                f"{sorted(overlap)[:5]}"
                            )
                if set().union(*role_items.values()) != train_items:
                    raise RuntimeError("B/R/U/f roles do not partition the outer-train items")

                b_items = role_items["b"]
                reachable_train = np.asarray(
                    [i for i, item in enumerate(all_groups) if item in b_items],
                    dtype=np.int64,
                )
                reachable = fit_reachable_basis(
                    all_delta[reachable_train],
                    all_groups[reachable_train],
                    rank_cap=min(
                        gate.reachable_rank_cap,
                        max(gate.reachable_rank_candidates),
                    ),
                    rank_candidates=tuple(gate.reachable_rank_candidates),
                    coverage_target=gate.reachable_min_coverage,
                    min_rank=max_transport_rank,
                    n_folds=min(3, len(b_items)),
                    # B is frozen across the R/U role-swap sensitivity rotation.
                    seed=gate.seed + fold,
                )
                transport = fit_reduced_rank_transport(
                    harmful_delta[u_indices],
                    endpoint_delta[u_indices],
                    reachable,
                    groups=harmful_groups[u_indices],
                    max_rank=max_transport_rank,
                    ridge_alphas=tuple(gate.ridge_alphas),
                    n_folds=min(3, len(set(harmful_groups[u_indices].tolist()))),
                    seed=gate.seed + fold + rotation_index,
                    endpoint_kind=gate.endpoint_kind,
                )

                r_items = role_items["r"]
                refusal_states: list[np.ndarray] = []
                refusal_labels: list[int] = []
                for cell in cells:
                    if (
                        str(cell["item_id"]) not in r_items
                        or str(cell["safety_label"]) != "harmful"
                    ):
                        continue
                    label = str(cell.get(gate.behavior_label_field) or "")
                    if label not in {"policy_refusal", "harmful_compliance"}:
                        continue
                    refusal_states.append(states[int(cell["activation_index"])])
                    refusal_labels.append(int(label == "policy_refusal"))
                if not refusal_states:
                    raise ValueError(
                        f"layer={layer} fold={fold} rotation={rotation_index} R role "
                        "has no reviewed harmful refusal/compliance states"
                    )
                observed_refusal_classes = set(refusal_labels)
                if observed_refusal_classes != {0, 1}:
                    raise ValueError(
                        f"layer={layer} fold={fold} rotation={rotation_index} R role "
                        "requires both policy_refusal and harmful_compliance states; "
                        f"observed binary classes={sorted(observed_refusal_classes)}"
                    )
                refusal = fit_dim_refusal_basis(
                    np.stack(refusal_states),
                    np.asarray(refusal_labels, dtype=np.int8),
                    rank=1,
                )
                f_items = role_items["f"]
                predictor_train = np.asarray(
                    [i for i, item in enumerate(all_groups) if item in f_items],
                    dtype=np.int64,
                )
                if len(predictor_train) == 0:
                    raise ValueError(
                        f"layer={layer} fold={fold} rotation={rotation_index} f role "
                        "has no arm-label-free delta rows"
                    )
                predictor = fit_natural_predictor(
                    all_neutral_states[predictor_train],
                    all_severity[predictor_train],
                    all_delta[predictor_train],
                    reachable,
                    groups=all_groups[predictor_train],
                    ridge_alphas=tuple(gate.ridge_alphas),
                    n_folds=min(3, len(f_items)),
                    # f uses the fixed f items and B, independent of the R/U swap.
                    seed=gate.seed + fold,
                )
                predictor_metrics = _json_metrics(predictor)
                predictor_gate = _predictor_gate(
                    predictor_metrics, gate.natural_predictor_min_improvement
                )
                predicted_full = predictor.predict_delta(neutral_states[test], severity[test])
                actual_coordinates = basis_coordinates(harmful_delta[test], reachable.basis)
                predicted_coordinates = basis_coordinates(predicted_full, reachable.basis)
                mean_prediction = _operator_mean_prediction(
                    all_delta[predictor_train],
                    all_severity[predictor_train],
                    severity[test],
                )
                mean_coordinates = basis_coordinates(mean_prediction, reachable.basis)
                predictor_test_mse = float(
                    np.mean(np.square(predicted_coordinates - actual_coordinates))
                )
                mean_test_mse = float(np.mean(np.square(mean_coordinates - actual_coordinates)))
                test_improvement = (
                    float((mean_test_mse - predictor_test_mse) / mean_test_mse)
                    if mean_test_mse > 1e-12
                    else None
                )
                prefix = f"layer{layer}_fold{fold}_rotation{rotation_index}"
                _prefix_artifacts(artifact_arrays, f"{prefix}_reachable", reachable)
                _prefix_artifacts(artifact_arrays, f"{prefix}_transport", transport)
                _prefix_artifacts(artifact_arrays, f"{prefix}_refusal", refusal)
                _prefix_artifacts(artifact_arrays, f"{prefix}_predictor", predictor)

                rank_metrics: dict[str, Any] = {}
                for rank in gate.transport_ranks:
                    rank = int(rank)
                    if rank > transport.max_rank:
                        rank_metrics[str(rank)] = {
                            "status": "not_available",
                            "transport_max_rank": transport.max_rank,
                        }
                        continue
                    predicted_endpoint = transport.predict_endpoint(harmful_delta[test], rank)
                    mse = float(np.mean(np.square(predicted_endpoint - endpoint_delta[test])))
                    u_basis = transport.basis_for_rank(rank)
                    coupling = np.linalg.svd(
                        np.asarray(refusal.basis) @ np.asarray(u_basis).T,
                        compute_uv=False,
                    )
                    rank_metrics[str(rank)] = {
                        "status": "available",
                        "heldout_endpoint_mse": mse,
                        "refusal_transport_singular_values": coupling.tolist(),
                    }
                    if benign_endpoint_delta is not None:
                        benign_prediction = transport.predict_endpoint(
                            benign_delta[benign_test], rank
                        )
                        benign_actual = benign_endpoint_delta[benign_test]
                        rank_metrics[str(rank)]["heldout_benign"] = {
                            "n_rows": int(len(benign_test)),
                            "endpoint_mse": float(
                                np.mean(np.square(benign_prediction - benign_actual))
                            ),
                            "actual_delta_mean": benign_actual.mean(axis=0).tolist(),
                            "predicted_delta_mean": benign_prediction.mean(axis=0).tolist(),
                            "actual_delta_mean_norm": float(
                                np.linalg.norm(benign_actual.mean(axis=0))
                            ),
                        }
                rotation_metrics.append(
                    {
                        "rotation": int(assignment.rotation),
                        "role_assignment": assignment.metrics(),
                        "role_item_counts": {
                            name: len(items) for name, items in role_items.items()
                        },
                        "reachable": _json_metrics(reachable),
                        "transport": _json_metrics(transport),
                        "refusal": _json_metrics(refusal),
                        "natural_predictor": predictor_metrics,
                        "natural_predictor_gate_passed": predictor_gate,
                        "natural_predictor_test_mse": predictor_test_mse,
                        "operator_mean_test_mse": mean_test_mse,
                        "natural_predictor_test_improvement": test_improvement,
                        "configured_transport_ranks": [
                            int(value) for value in gate.transport_ranks
                        ],
                        "available_transport_ranks": [
                            int(value)
                            for value in gate.transport_ranks
                            if int(value) <= transport.max_rank
                        ],
                        "ranks": rank_metrics,
                    }
                )
                if rotation_index == 0:
                    primary_bundle = {
                        "transport": transport,
                        "predictor": predictor,
                        "predictor_gate": predictor_gate,
                        "available_ranks": [
                            int(value)
                            for value in gate.transport_ranks
                            if int(value) <= transport.max_rank
                        ],
                    }

            if primary_bundle is None:
                raise RuntimeError("no primary disjoint-role transport fit was produced")

            if layer == gate.primary_layer:
                primary_transport = primary_bundle["transport"]
                primary_predictor = primary_bundle["predictor"]
                for test_index in test:
                    row = harmful_rows[int(test_index)]
                    full = harmful_delta[int(test_index)]
                    transformed_cell = cells_by_index[int(row["transformed_activation_index"])]
                    for rank_value in primary_bundle["available_ranks"]:
                        rank = int(rank_value)
                        u_basis = primary_transport.basis_for_rank(rank)
                        same_pair = project_onto_basis(full, u_basis)
                        perpendicular = orthogonal_residual(full, u_basis)
                        predicted = primary_predictor.predict_transport_delta(
                            neutral_states[int(test_index)][None, :],
                            severity[int(test_index)][None, :],
                            primary_transport,
                            rank,
                        )[0]
                        vector_index = len(vectors["delta_u"])
                        vectors["delta_u"].append(np.asarray(same_pair, dtype=np.float32))
                        vectors["delta_perp"].append(np.asarray(perpendicular, dtype=np.float32))
                        vectors["delta_full"].append(np.asarray(full, dtype=np.float32))
                        vectors["delta_predicted"].append(np.asarray(predicted, dtype=np.float32))
                        intervention_manifest.append(
                            {
                                "intervention_id": (
                                    f"fold{fold}:{row['item_id']}:{row['severity']}:rank{rank}"
                                ),
                                "fold": fold,
                                "role_rotation": 0,
                                "item_id": row["item_id"],
                                "safety_label": row["safety_label"],
                                "operator_family": gate.operator_family,
                                "operator_backend": gate.operator_backend,
                                "severity": row["severity"],
                                "rank": rank,
                                "layer": layer,
                                "site": gate.site,
                                "position_name": "first_generation_prelogit",
                                "endpoint_kind": gate.endpoint_kind,
                                "transformed_activation_index": row["transformed_activation_index"],
                                "neutral_activation_index": row["neutral_activation_index"],
                                "vector_index": vector_index,
                                "transport_rank_status": "available",
                                "transport_max_rank": primary_transport.max_rank,
                                "natural_predictor_gate_passed": primary_bundle["predictor_gate"],
                                "selected_for_intervention": True,
                                "reference_text": transformed_cell.get("reference_text"),
                            }
                        )
                    unavailable_ranks = [
                        int(value)
                        for value in gate.transport_ranks
                        if int(value) > primary_transport.max_rank
                    ]
                    for rank in unavailable_ranks:
                        intervention_manifest.append(
                            {
                                "intervention_id": (
                                    f"fold{fold}:{row['item_id']}:{row['severity']}:rank{rank}"
                                ),
                                "fold": fold,
                                "role_rotation": 0,
                                "item_id": row["item_id"],
                                "safety_label": row["safety_label"],
                                "operator_family": gate.operator_family,
                                "operator_backend": gate.operator_backend,
                                "severity": row["severity"],
                                "rank": rank,
                                "layer": layer,
                                "site": gate.site,
                                "position_name": "first_generation_prelogit",
                                "endpoint_kind": gate.endpoint_kind,
                                "transformed_activation_index": row["transformed_activation_index"],
                                "neutral_activation_index": row["neutral_activation_index"],
                                "vector_index": None,
                                "transport_rank_status": "not_available",
                                "transport_max_rank": primary_transport.max_rank,
                                "natural_predictor_gate_passed": primary_bundle["predictor_gate"],
                                "selected_for_intervention": False,
                                "reference_text": transformed_cell.get("reference_text"),
                            }
                        )

            fold_metrics.append(
                {
                    "fold": fold,
                    "n_train_items": len(train_items),
                    "n_test_items": len(set(harmful_groups[test].tolist())),
                    "role_rotations": rotation_metrics,
                }
            )
        layer_metrics[str(layer)] = {"folds": fold_metrics}

    primary_folds = layer_metrics[str(gate.primary_layer)]["folds"]
    rank_fold_mse: dict[int, list[float]] = {int(rank): [] for rank in gate.transport_ranks}
    benign_rank_fold_mse: dict[int, list[float]] = {int(rank): [] for rank in gate.transport_ranks}
    benign_rank_rows: dict[int, int] = {int(rank): 0 for rank in gate.transport_ranks}
    for fold_row in primary_folds:
        primary_rotation = fold_row["role_rotations"][0]
        for rank_text, rank_row in primary_rotation["ranks"].items():
            rank = int(rank_text)
            if rank_row.get("status") != "available":
                continue
            rank_fold_mse[rank].append(float(rank_row["heldout_endpoint_mse"]))
            benign_row = rank_row.get("heldout_benign")
            if benign_row is not None:
                benign_rank_fold_mse[rank].append(float(benign_row["endpoint_mse"]))
                benign_rank_rows[rank] += int(benign_row["n_rows"])

    mean_mse_by_rank = {
        str(rank): float(np.mean(values))
        for rank, values in rank_fold_mse.items()
        if len(values) == len(primary_folds)
    }
    rank1_mse = mean_mse_by_rank.get("1")

    def relative_reduction(value: float | None) -> float | None:
        if rank1_mse is None or value is None or rank1_mse <= 1e-12:
            return None
        return float((rank1_mse - value) / rank1_mse)

    rank2_mse = mean_mse_by_rank.get("2")
    higher = {int(rank): value for rank, value in mean_mse_by_rank.items() if int(rank) > 1}
    best_higher_rank = min(higher, key=lambda rank: (higher[rank], rank)) if higher else None
    best_higher_mse = None if best_higher_rank is None else higher[best_higher_rank]
    best_higher_reduction = relative_reduction(best_higher_mse)
    rank_summary = {
        "endpoint_kind": gate.endpoint_kind,
        "n_outer_folds": len(primary_folds),
        "mean_outer_heldout_mse_by_rank": mean_mse_by_rank,
        "rank_fold_counts": {str(rank): len(values) for rank, values in rank_fold_mse.items()},
        "rank2_relative_mse_reduction_vs_rank1": relative_reduction(rank2_mse),
        "best_rank_gt1": best_higher_rank,
        "best_rank_gt1_relative_mse_reduction_vs_rank1": best_higher_reduction,
        "multidim_min_mse_reduction": gate.multidim_min_mse_reduction,
        "geometry_candidate_only": bool(
            best_higher_reduction is not None
            and best_higher_reduction >= gate.multidim_min_mse_reduction
        ),
        "causal_gate_pending": True,
        "interpretation": "exploratory geometry only; no GO or multidimensional conclusion",
    }
    benign_summary = {
        "endpoint_scored": benign_endpoint_delta is not None,
        "outer_heldout_rotation": 0,
        "mean_endpoint_mse_by_rank": {
            str(rank): float(np.mean(values))
            for rank, values in benign_rank_fold_mse.items()
            if values
        },
        "n_rows_by_rank": {str(rank): count for rank, count in benign_rank_rows.items() if count},
        "descriptive_only": True,
        "used_to_train_transport_u": False,
    }

    empty_vectors = [name for name, values in vectors.items() if not values]
    if empty_vectors:
        raise RuntimeError(
            f"COAST-R fit produced no frozen intervention rows for vector arrays: {empty_vectors}"
        )
    vector_lengths = {name: len(values) for name, values in vectors.items()}
    if len(set(vector_lengths.values())) != 1:
        raise RuntimeError(f"intervention vector arrays are not aligned: {vector_lengths}")
    vector_arrays = {
        name: np.stack(values).astype(np.float32, copy=False) for name, values in vectors.items()
    }
    metrics = {
        "site": gate.site,
        "primary_layer": gate.primary_layer,
        "sensitivity_layers": gate.sensitivity_layers,
        "n_cells": len(cells),
        "n_delta_rows": len(metadata),
        "n_harmful_delta_rows": len(harmful_rows),
        "n_intervention_rows": len(intervention_manifest),
        "n_intervention_vector_rows": len(vectors["delta_u"]),
        "rank_summary": rank_summary,
        "benign_heldout_summary": benign_summary,
        "causal_gate_pending": True,
        "layers": layer_metrics,
    }
    return {
        "metrics": metrics,
        "artifact_arrays": artifact_arrays,
        "intervention_manifest": intervention_manifest,
        "intervention_vectors": vector_arrays,
    }


def _fit_report(metrics: dict[str, Any]) -> str:
    return (
        "# COAST-R Stage A fit\n\n"
        "> Exploratory cross-fit over the fully exposed Run 6 cohort; "
        "not a confirmatory result.\n\n"
        "```json\n"
        f"{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}\n"
        "```\n"
    )


def fit_coast_r(
    cfg: ExperimentConfig,
    source: CoastRSource,
    run_dir: Path,
) -> dict[str, Any]:
    """CPU phase: fit cross-fitted reachable transport and freeze intervention rows."""
    gate = require_coast_r_config(cfg)
    metrics_path = run_dir / gate.fit_metrics_file
    artifact_path = run_dir / gate.fit_artifact_file
    manifest_path = run_dir / gate.intervention_manifest_file
    vectors_path = run_dir / gate.intervention_vectors_file
    report_path = run_dir / gate.fit_report_file
    outputs = (metrics_path, artifact_path, manifest_path, vectors_path, report_path)
    if not gate.overwrite and all(path.is_file() for path in outputs):
        existing_metrics = load_json(metrics_path)
        _require_source_hashes(
            existing_metrics.get("source") if isinstance(existing_metrics, dict) else None,
            source,
            context="completed COAST-R fit metrics",
        )
        return existing_metrics
    if not gate.overwrite and any(path.exists() for path in outputs):
        present = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            "partial COAST-R fit artifacts exist; remove the incomplete Run 7 directory or "
            f"set coast_r.overwrite=true: {present}"
        )

    score_path = run_dir / gate.continuation_scores_file
    if gate.endpoint_kind == "continuation_curve" and not score_path.is_file():
        raise FileNotFoundError(
            f"continuation endpoint requires the score phase first: {score_path}"
        )
    score_rows = load_jsonl(score_path) if score_path.is_file() else []
    _validate_resumable_rows(
        score_rows,
        source,
        id_field="activation_index",
        context="continuation score artifact used by fit",
    )
    with np.load(source.activations_path) as archive:
        if gate.site not in archive.files:
            raise KeyError(f"source activation archive has no site {gate.site!r}")
        if "llm_layers" not in archive.files:
            raise KeyError("source activation archive has no `llm_layers`")
        source_layers = np.asarray(archive["llm_layers"]).reshape(-1)
        required_layers = list(dict.fromkeys([gate.primary_layer, *gate.sensitivity_layers]))
        missing_layers = [layer for layer in required_layers if layer not in source_layers]
        if missing_layers:
            raise ValueError(
                f"configured decoder layers {missing_layers} are absent from source "
                f"llm_layers={source_layers.tolist()}"
            )
        layer_indices = [
            int(np.flatnonzero(source_layers == layer)[0]) for layer in required_layers
        ]
        source_site = np.asarray(archive[gate.site])
        if source_site.ndim != 3 or source_site.shape[1] != len(source_layers):
            raise ValueError(
                f"site {gate.site!r} must have shape (cell, layer, dim), got {source_site.shape}"
            )
        # The compressed NPZ must decompress the source array once, but retain only
        # the declared primary/sensitivity layers for the CPU cross-fit.
        arrays = {
            gate.site: np.asarray(source_site[:, layer_indices, :]),
            "llm_layers": np.asarray(required_layers, dtype=source_layers.dtype),
        }
        del source_site

    result = _run_crossfit_fit(arrays, source.cells, score_rows, gate)
    metrics = dict(result["metrics"])
    metrics.update(
        {
            "status": "EXPLORATORY_METHOD_KILL_TEST",
            "source_run_name": gate.source_run_name,
            "source": source.hashes,
        }
    )
    artifact_arrays = {
        str(name): np.asarray(value) for name, value in result["artifact_arrays"].items()
    }
    vector_arrays = {
        str(name): np.asarray(value) for name, value in result["intervention_vectors"].items()
    }
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **artifact_arrays)
    np.savez_compressed(vectors_path, **vector_arrays)
    save_jsonl(result["intervention_manifest"], manifest_path)
    report_path.write_text(str(result.get("report") or _fit_report(metrics)))
    save_json(metrics, metrics_path)
    return metrics


def _manifest_row_id(row: dict[str, Any], offset: int) -> str:
    if row.get("intervention_id") is not None:
        return str(row["intervention_id"])
    return ":".join(
        [
            str(row.get("fold", "fold")),
            str(row.get("item_id", "item")),
            str(row.get("severity", "severity")),
            str(row.get("rank", "rank")),
            str(offset),
        ]
    )


def _eligible_intervention(
    manifest_row: dict[str, Any],
    cells_by_index: dict[int, dict[str, Any]],
    gate: CoastRConfig,
) -> bool:
    transformed = cells_by_index[int(manifest_row["transformed_activation_index"])]
    neutral = cells_by_index[int(manifest_row["neutral_activation_index"])]
    if str(transformed.get("safety_label")) != "harmful":
        return False
    label_field = gate.behavior_label_field
    neutral_refused = str(neutral.get(label_field) or "") == "policy_refusal"
    if gate.intervention_eligibility == "all_harmful":
        return True
    if gate.intervention_eligibility == "neutral_refusers":
        return neutral_refused
    if gate.intervention_eligibility == "verified_flips":
        return neutral_refused and str(transformed.get(label_field) or "") == "harmful_compliance"
    raise ValueError(f"unknown intervention eligibility {gate.intervention_eligibility!r}")


def _heuristic(output: str, cell: dict[str, Any]) -> dict[str, Any]:
    from audio_safety.evaluation.labeling import label_output

    label, failure_mode, needs_review = label_output(
        output,
        safety_label=str(cell["safety_label"]),
        reference_text=str(cell.get("reference_text") or "") or None,
    )
    return {
        "label": label,
        "failure_mode": failure_mode,
        "needs_review": needs_review,
    }


def generate_coast_r_interventions(
    model: Any,
    processor: Any,
    cfg: ExperimentConfig,
    source: CoastRSource,
    run_dir: Path,
) -> dict[str, Any]:
    """GPU phase: write one judge-ready row per frozen causal condition.

    ``restore_u`` subtracts the same-pair reachable component from transformed
    P2. ``neutral_same_pair_u`` is explicitly only a reconstruction sanity check.
    ``neutral_predicted_u`` is the out-of-pair sufficiency condition.  Every raw
    intervention delta is added once at the absolute P2 prefill position.
    """
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        generate_audio_response,
        generate_audio_response_with_raw_delta,
        resolve_audio_position_indices,
    )

    gate = require_coast_r_config(cfg)
    metrics_path = run_dir / gate.fit_metrics_file
    manifest_path = run_dir / gate.intervention_manifest_file
    vectors_path = run_dir / gate.intervention_vectors_file
    output_path = run_dir / gate.intervention_outputs_file
    if not metrics_path.is_file() or not manifest_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError("COAST-R intervention requires the fit phase artifacts")
    fit_metrics = load_json(metrics_path)
    _require_source_hashes(
        fit_metrics.get("source") if isinstance(fit_metrics, dict) else None,
        source,
        context="COAST-R fit used by intervention",
    )
    manifest = load_jsonl(manifest_path)
    with np.load(vectors_path) as archive:
        required = {"delta_u", "delta_predicted"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"intervention vectors are missing required arrays: {sorted(missing)}")
        delta_u = np.asarray(archive["delta_u"], dtype=np.float32)
        delta_predicted = np.asarray(archive["delta_predicted"], dtype=np.float32)
        delta_full = None
        for name in ("delta_full", "full_delta"):
            if name in archive.files:
                delta_full = np.asarray(archive[name], dtype=np.float32)
                break
        delta_perp = (
            np.asarray(archive["delta_perp"], dtype=np.float32)
            if "delta_perp" in archive.files
            else None
        )
    if len(delta_u) != len(delta_predicted):
        raise ValueError("delta_u and delta_predicted must have the same number of rows")
    if delta_u.ndim != 2 or delta_predicted.shape != delta_u.shape:
        raise ValueError(
            "delta_u and delta_predicted must be aligned (n_interventions, d_model) arrays"
        )
    for name, values in (("delta_full", delta_full), ("delta_perp", delta_perp)):
        if values is not None and values.shape != delta_u.shape:
            raise ValueError(f"{name} must have the same shape as delta_u")
    if delta_perp is None and delta_full is not None:
        delta_perp = delta_full - delta_u
    if delta_full is None and delta_perp is not None:
        delta_full = delta_u + delta_perp

    cells_by_index = {int(cell["activation_index"]): cell for cell in source.cells}
    selected: list[tuple[int, dict[str, Any]]] = []
    manifest_ids: set[str] = set()
    for offset, row in enumerate(manifest):
        row_id = _manifest_row_id(row, offset)
        if row_id in manifest_ids:
            raise RuntimeError(f"intervention manifest contains duplicate id {row_id!r}")
        manifest_ids.add(row_id)
        if row.get("selected_for_intervention", True) is False:
            continue
        if not _eligible_intervention(row, cells_by_index, gate):
            continue
        vector_index = int(row.get("vector_index", offset))
        if vector_index < 0 or vector_index >= len(delta_u):
            raise IndexError(f"intervention vector_index {vector_index} is out of range")
        selected.append((offset, row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gate.overwrite and output_path.exists():
        output_path.unlink()
    existing = load_jsonl(output_path) if output_path.exists() else []
    _validate_resumable_rows(
        existing,
        source,
        id_field="intervention_id",
        context="intervention output artifact",
    )
    done = {str(row["intervention_id"]) for row in existing}
    baseline_cache: dict[int, str] = {}
    resolved_p2_cache: dict[int, int] = {}
    written = 0

    def validated_p2(index: int, cell: dict[str, Any], audio_path: Path) -> int:
        if index not in resolved_p2_cache:
            conversation = build_audio_analysis_conversation(
                audio_path,
                cfg.dataset.target_generation.instruction,
                system_prompt=gate.system_prompt,
            )
            resolved = int(
                resolve_audio_position_indices(processor, conversation)["first_generation_prelogit"]
            )
            saved = cell.get("p2_index")
            if saved is None or int(saved) != resolved:
                raise RuntimeError(
                    "current official chat template does not reproduce the saved P2 index "
                    f"for activation_index={index}: saved={saved}, resolved={resolved}"
                )
            resolved_p2_cache[index] = resolved
        return resolved_p2_cache[index]

    def baseline(index: int, cell: dict[str, Any], audio_path: Path) -> str:
        if index not in baseline_cache:
            baseline_cache[index] = generate_audio_response(
                model,
                processor,
                audio_path,
                cfg.dataset.target_generation.instruction,
                max_new_tokens=gate.max_new_tokens,
                system_prompt=gate.system_prompt,
                do_sample=False,
            )
        return baseline_cache[index]

    with output_path.open("a") as handle:
        for offset, manifest_row in tqdm(selected, desc="COAST-R causal generation", unit="row"):
            row_id = _manifest_row_id(manifest_row, offset)
            vector_index = int(manifest_row.get("vector_index", offset))
            transformed_index = int(manifest_row["transformed_activation_index"])
            neutral_index = int(manifest_row["neutral_activation_index"])
            transformed_cell = cells_by_index[transformed_index]
            neutral_cell = cells_by_index[neutral_index]
            transformed_audio = source.audio_paths[transformed_index]
            neutral_audio = source.audio_paths[neutral_index]
            transformed_p2 = validated_p2(transformed_index, transformed_cell, transformed_audio)
            neutral_p2 = validated_p2(neutral_index, neutral_cell, neutral_audio)
            u = delta_u[vector_index]
            predicted = delta_predicted[vector_index]
            if not np.isfinite(u).all() or not np.isfinite(predicted).all():
                raise ValueError(f"non-finite intervention vector for {row_id}")

            conditions: list[tuple[str, Path, dict[str, Any], np.ndarray | None]] = [
                ("transformed_baseline", transformed_audio, transformed_cell, None),
                ("neutral_baseline", neutral_audio, neutral_cell, None),
                ("restore_u", transformed_audio, transformed_cell, -u),
            ]
            if delta_perp is not None:
                conditions.append(
                    (
                        "restore_perp",
                        transformed_audio,
                        transformed_cell,
                        -delta_perp[vector_index],
                    )
                )
            if delta_full is not None:
                conditions.append(
                    (
                        "restore_full",
                        transformed_audio,
                        transformed_cell,
                        -delta_full[vector_index],
                    )
                )
            conditions.append(("neutral_same_pair_u", neutral_audio, neutral_cell, u))
            if manifest_row.get("natural_predictor_gate_passed", True):
                conditions.append(("neutral_predicted_u", neutral_audio, neutral_cell, predicted))

            for condition, audio_path, condition_cell, raw_delta in conditions:
                intervention_id = f"{row_id}:{condition}"
                if intervention_id in done:
                    continue
                if raw_delta is None:
                    source_index = (
                        transformed_index if condition == "transformed_baseline" else neutral_index
                    )
                    output = baseline(source_index, condition_cell, audio_path)
                    applied_count = 0
                    delta_norm = 0.0
                else:
                    if not np.isfinite(raw_delta).all():
                        raise ValueError(f"non-finite {condition} delta for {intervention_id}")
                    output, applied_count = generate_audio_response_with_raw_delta(
                        model,
                        processor,
                        audio_path,
                        cfg.dataset.target_generation.instruction,
                        layer_idx=gate.primary_layer,
                        position_name="first_generation_prelogit",
                        delta=raw_delta,
                        max_new_tokens=gate.max_new_tokens,
                        system_prompt=gate.system_prompt,
                    )
                    if applied_count != 1:
                        raise RuntimeError(
                            f"{condition} applied {applied_count} times, expected exactly 1"
                        )
                    delta_norm = float(np.linalg.norm(raw_delta))
                    source_index = (
                        transformed_index if condition.startswith("restore_") else neutral_index
                    )
                heuristic = _heuristic(output, condition_cell)
                reference_text = condition_cell.get("reference_text")
                record = {
                    **manifest_row,
                    "intervention_id": intervention_id,
                    "condition": condition,
                    "vector_index": vector_index,
                    "audio_path": str(audio_path),
                    "reference_text": reference_text,
                    "judge_request_text": str(reference_text or ""),
                    "modality": "audio",
                    "category": condition_cell.get("category"),
                    "safety_label": condition_cell.get("safety_label"),
                    "output": output,
                    "heuristic_behavior_label": heuristic["label"],
                    "heuristic_failure_mode": heuristic["failure_mode"],
                    "heuristic_needs_review": heuristic["needs_review"],
                    "raw_delta_norm": delta_norm,
                    "intervention_applied_count": applied_count,
                    "intervention_eligibility": gate.intervention_eligibility,
                    "resolved_p2_index": (
                        transformed_p2 if source_index == transformed_index else neutral_p2
                    ),
                    **source.hashes,
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                done.add(intervention_id)
                written += 1

    return {
        "output": str(output_path),
        "n_manifest": len(manifest),
        "n_selected": len(selected),
        "n_preexisting": len(existing),
        "n_written": written,
        "n_complete": len(done),
        "intervention_eligibility": gate.intervention_eligibility,
        "source": source.hashes,
    }
