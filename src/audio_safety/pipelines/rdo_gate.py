"""End-to-end helpers for the Audio-RDO gate.

Algorithmic anchors:
- RDO-style optimized refusal direction follows the gradient-optimized
  representation-engineering setup from *The Geometry of Refusal in Large
  Language Models: Concept Cones and Representational Independence*.
- DIM and SAR-style baseline vectors follow SARSteer's compliance-to-refusal and
  text-derived refusal steering baselines.
- Style escape is now an exploratory content-preserving expressive-style
  condition. It is not the same as a strict transcript-fixed acoustic-only test.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from audio_safety.config.schema import ExperimentConfig
from audio_safety.data.datasets import AudioRdoPair, split_audio_rdo_pairs
from audio_safety.evaluation.decision import AudioRdoGateMetrics, decide_audio_rdo_gate
from audio_safety.evaluation.labeling import label_output
from audio_safety.models.hooks import ResidualStreamCapture
from audio_safety.models.qwen2_audio import (
    build_audio_analysis_conversation,
    generate_audio_response_with_intervention,
    model_input_device,
    prepare_qwen2_audio_inputs,
    prepare_qwen2_audio_teacher_forced_inputs,
    resolve_audio_position_indices,
)
from audio_safety.pipelines.audio_rdo import (
    RdoTrainingBatch,
    dim_compliance_to_refusal,
    escape_scores,
    harmful_compliance_rate,
    refusal_rate,
    sar_text_refusal_vector,
    signed_occupancy,
    train_audio_rdo_axis,
    unit_vector,
)
from audio_safety.utils.io import load_jsonl, save_json, save_jsonl


@dataclass(frozen=True)
class Site:
    layer: int
    position: str


def split_ids(pairs: list[AudioRdoPair], cfg: ExperimentConfig) -> dict[str, set[str]]:
    split = split_audio_rdo_pairs(pairs, cfg.dataset, cfg.seed)
    return {
        "train": {pair.item_id for pair in split.train},
        "validation": {pair.item_id for pair in split.validation},
        "heldout": {pair.item_id for pair in split.heldout},
    }


def rows_for_split(
    rows: Sequence[dict[str, Any]],
    split_map: dict[str, set[str]],
    split_name: str,
) -> list[dict[str, Any]]:
    allowed = split_map[split_name]
    return [row for row in rows if str(row["item_id"]) in allowed]


def valid_behavior_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("behavior_label") != "decoding_failure"
        and bool(row.get("transcript_control_passed", True))
    ]


def _row_audio_path(row: dict[str, Any], data_dir: Path) -> Path:
    return data_dir / str(row["path"])


def _row_conversation(row: dict[str, Any], cfg: ExperimentConfig, data_dir: Path):
    return build_audio_analysis_conversation(
        _row_audio_path(row, data_dir),
        cfg.dataset.target_generation.instruction,
    )


def _capture_inputs_site(
    model: Any,
    inputs: Any,
    *,
    layer: int,
    token_index: int,
) -> np.ndarray:
    with ResidualStreamCapture(model, token_index=token_index, layers=[layer]) as cap:
        model(**inputs)
    return cap.states()[layer].numpy()


def capture_prompt_hidden(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    data_dir: Path,
    site: Site,
) -> np.ndarray:
    conversation = _row_conversation(row, cfg, data_dir)
    token_index = resolve_audio_position_indices(processor, conversation)[site.position]
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    return _capture_inputs_site(model, inputs, layer=site.layer, token_index=token_index)


def capture_refusal_continuation_hidden(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    data_dir: Path,
    site: Site,
) -> np.ndarray:
    conversation = _row_conversation(row, cfg, data_dir)
    device = model_input_device(model)
    inputs, _labels, _prompt_len = prepare_qwen2_audio_teacher_forced_inputs(
        processor,
        conversation,
        cfg.rdo.refusal_target,
        device=device,
    )
    return _capture_inputs_site(model, inputs, layer=site.layer, token_index=-1)


def extract_selected_site_activations(
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    data_dir: Path,
    run_dir: Path,
    site: Site,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    activations = []
    metadata = []
    for idx, row in enumerate(rows):
        hidden = capture_prompt_hidden(model, processor, row, cfg, data_dir, site)
        activations.append(hidden)
        metadata.append(
            {
                "activation_index": idx,
                "item_id": row["item_id"],
                "safety_label": row["safety_label"],
                "style": row["style"],
                "behavior_label": row.get("behavior_label"),
                "path": row["path"],
            }
        )
    array = np.stack(activations).astype(np.float32)
    path = run_dir / cfg.rdo.activations_file
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, activations=array)
    save_jsonl(metadata, run_dir / cfg.rdo.activation_metadata_file)
    return array, metadata


def _teacher_forced_batch(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    target: str,
    cfg: ExperimentConfig,
    data_dir: Path,
    position_name: str,
):
    conversation = _row_conversation(row, cfg, data_dir)
    token_index = resolve_audio_position_indices(processor, conversation)[position_name]
    inputs, labels, _prompt_len = prepare_qwen2_audio_teacher_forced_inputs(
        processor,
        conversation,
        target,
        device=model_input_device(model),
    )
    return inputs, labels, token_index


def _prompt_inputs(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    data_dir: Path,
    position_name: str,
):
    conversation = _row_conversation(row, cfg, data_dir)
    token_index = resolve_audio_position_indices(processor, conversation)[position_name]
    inputs = prepare_qwen2_audio_inputs(
        processor,
        conversation,
        device=model_input_device(model),
    )
    return inputs, token_index


def build_rdo_training_batches(
    model: Any,
    processor: Any,
    train_rows: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    data_dir: Path,
    position_name: str,
    *,
    limit: int | None = None,
) -> list[RdoTrainingBatch]:
    rows = valid_behavior_rows(train_rows)
    add_rows = [
        row
        for row in rows
        if row["safety_label"] == "harmful" and row["behavior_label"] == "harmful_compliance"
    ]
    ablate_rows = [
        row
        for row in rows
        if row["safety_label"] == "harmful" and row["behavior_label"] == "policy_refusal"
    ]
    retain_rows = [row for row in rows if row["safety_label"] == "benign"]
    if not add_rows:
        raise ValueError("RDO training requires harmful_compliance rows for addition loss")
    n_batches = limit or max(len(add_rows), len(ablate_rows), len(retain_rows), 1)
    batches = []
    for add_row, ablate_row, retain_row in islice(
        zip(
            cycle(add_rows),
            cycle(ablate_rows or [None]),
            cycle(retain_rows or [None]),
        ),
        n_batches,
    ):
        add_inputs, add_labels, add_token_index = _teacher_forced_batch(
            model,
            processor,
            add_row,
            cfg.rdo.refusal_target,
            cfg,
            data_dir,
            position_name,
        )
        ablate_inputs = ablate_labels = ablate_token_index = None
        if ablate_row is not None:
            ablate_inputs, ablate_labels, ablate_token_index = _teacher_forced_batch(
                model,
                processor,
                ablate_row,
                cfg.rdo.nonrefusal_prefix,
                cfg,
                data_dir,
                position_name,
            )
        retain_inputs = retain_token_index = None
        if retain_row is not None:
            retain_inputs, retain_token_index = _prompt_inputs(
                model,
                processor,
                retain_row,
                cfg,
                data_dir,
                position_name,
            )
        batches.append(
            RdoTrainingBatch(
                add_inputs=add_inputs,
                add_labels=add_labels,
                add_token_index=add_token_index,
                ablate_inputs=ablate_inputs,
                ablate_labels=ablate_labels,
                ablate_token_index=ablate_token_index,
                retain_inputs=retain_inputs,
                retain_token_index=retain_token_index,
            )
        )
    return batches


def generate_intervention_records(
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    data_dir: Path,
    *,
    site: Site,
    vector: np.ndarray,
    vector_name: str,
    mode: str,
    target_coordinates: dict[tuple[str, str, str], float] | None = None,
) -> list[dict[str, Any]]:
    # add/ablate move the hidden state by a position-invariant translation, so the
    # all-token scope is well defined. Restoration (set_coordinate) is the
    # preregistered H4 operator (design.md §4 / H4): it restores only the
    # readout-position r_A coordinate to that sample's neutral occupancy
    # Occ_{i,neutral}. That target is a single scalar measured at one position;
    # clamping every token position to it would be a different, stronger
    # intervention and would silently corrupt restoration_rr_pp / restored_fraction
    # (§0 GO thresholds). Keep restoration single-position regardless of the flag.
    all_positions = cfg.rdo.intervention_all_positions and mode != "set_coordinate"
    records = []
    for row in tqdm(rows, desc=f"{vector_name} {mode}", unit="row", leave=False):
        key = (str(row["item_id"]), str(row["safety_label"]), str(row["style"]))
        target = target_coordinates.get(key) if target_coordinates else None
        output = generate_audio_response_with_intervention(
            model,
            processor,
            _row_audio_path(row, data_dir),
            cfg.dataset.target_generation.instruction,
            layer_idx=site.layer,
            position_name=site.position,
            vector=vector,
            mode=mode,
            scale=cfg.rdo.alpha,
            target_coordinate=target,
            all_positions=all_positions,
            max_new_tokens=cfg.dataset.target_generation.max_new_tokens,
        )
        label, failure_mode, needs_review = label_output(
            output,
            safety_label=str(row["safety_label"]),
            reference_text=str(row.get("reference_text") or ""),
        )
        updated = dict(row)
        updated.update(
            {
                "vector_name": vector_name,
                "intervention_mode": mode,
                "intervention_output": output,
                "intervention_label": label,
                "intervention_failure_mode": failure_mode,
                "intervention_needs_manual_review": needs_review,
            }
        )
        records.append(updated)
    return records


def intervention_metric_summary(
    base_rows: Sequence[dict[str, Any]],
    intervention_rows: Sequence[dict[str, Any]],
    *,
    safety_label: str,
) -> dict[str, float]:
    base_labels = [
        str(row["behavior_label"]) for row in base_rows if row["safety_label"] == safety_label
    ]
    intervention_labels = [
        str(row["intervention_label"])
        for row in intervention_rows
        if row["safety_label"] == safety_label
    ]
    return {
        "base_refusal_rate": refusal_rate(base_labels),
        "intervention_refusal_rate": refusal_rate(intervention_labels),
        "base_harmful_compliance_rate": harmful_compliance_rate(base_labels),
        "intervention_harmful_compliance_rate": harmful_compliance_rate(intervention_labels),
    }


def train_and_validate_site(
    model: Any,
    processor: Any,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    data_dir: Path,
    site: Site,
    *,
    limit: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    print(
        f"[rdo] site layer={site.layer} position={site.position} "
        f"limit={limit if limit is not None else 'full'}",
        flush=True,
    )
    batches = build_rdo_training_batches(
        model,
        processor,
        train_rows,
        cfg,
        data_dir,
        site.position,
        limit=limit,
    )
    print(f"[rdo] built {len(batches)} training batches", flush=True)
    axis = train_audio_rdo_axis(model, batches, layer_idx=site.layer, cfg=cfg.rdo)
    add_rows = [
        row
        for row in valid_behavior_rows(validation_rows)
        if row["safety_label"] == "harmful" and row["behavior_label"] == "harmful_compliance"
    ]
    benign_rows = [
        row for row in valid_behavior_rows(validation_rows) if row["safety_label"] == "benign"
    ]
    ablate_rows = [
        row
        for row in valid_behavior_rows(validation_rows)
        if row["safety_label"] == "harmful" and row["behavior_label"] == "policy_refusal"
    ]
    if limit:
        add_rows = add_rows[:limit]
        benign_rows = benign_rows[:limit]
        ablate_rows = ablate_rows[:limit]
    add_records = generate_intervention_records(
        model,
        processor,
        [*add_rows, *benign_rows],
        cfg,
        data_dir,
        site=site,
        vector=axis,
        vector_name="rdo_audio",
        mode="add",
    )
    ablate_records = generate_intervention_records(
        model,
        processor,
        ablate_rows,
        cfg,
        data_dir,
        site=site,
        vector=axis,
        vector_name="rdo_audio",
        mode="ablate",
    )
    add_harm = intervention_metric_summary(add_rows, add_records, safety_label="harmful")
    add_benign = intervention_metric_summary(benign_rows, add_records, safety_label="benign")
    abl_harm = intervention_metric_summary(ablate_rows, ablate_records, safety_label="harmful")
    add_rr_pp = 100 * (add_harm["intervention_refusal_rate"] - add_harm["base_refusal_rate"])
    benign_orr_pp = 100 * (
        add_benign["intervention_refusal_rate"] - add_benign["base_refusal_rate"]
    )
    ablation_asr_pp = 100 * (
        abl_harm["intervention_harmful_compliance_rate"]
        - abl_harm["base_harmful_compliance_rate"]
    )
    score = add_rr_pp - benign_orr_pp + ablation_asr_pp
    return axis, {
        "layer": site.layer,
        "position": site.position,
        "add_rr_pp": float(add_rr_pp),
        "benign_orr_add_pp": float(benign_orr_pp),
        "ablation_asr_pp": float(ablation_asr_pp),
        "score": float(score),
        "n_add": len(add_rows),
        "n_benign": len(benign_rows),
        "n_ablate": len(ablate_rows),
    }


def compute_baseline_vectors(
    activations: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    train_ids: set[str],
    *,
    sar_refusal_hidden: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rows = [
        (idx, meta)
        for idx, meta in enumerate(metadata)
        if str(meta["item_id"]) in train_ids and meta.get("safety_label") == "harmful"
    ]
    refused_rows = [idx for idx, meta in rows if meta.get("behavior_label") == "policy_refusal"]
    complied_rows = [
        idx for idx, meta in rows if meta.get("behavior_label") == "harmful_compliance"
    ]
    if not refused_rows or not complied_rows:
        raise ValueError(
            "MDSteer-c2r baseline requires both policy_refusal and "
            "harmful_compliance harmful train rows"
        )
    refused = np.stack(
        [activations[idx] for idx in refused_rows]
    )
    complied = np.stack(
        [activations[idx] for idx in complied_rows]
    )
    vectors = {"mdsteer_c2r": dim_compliance_to_refusal(refused, complied)}
    if sar_refusal_hidden is not None:
        base = np.stack([activations[idx] for idx, _meta in rows])
        vectors["sarsteer_text"] = sar_text_refusal_vector(sar_refusal_hidden, base)
    return vectors


def behavior_decomposition(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = f"{row['safety_label']}:{row['style']}"
        table[key][str(row["behavior_label"])] += 1
    return {key: dict(value) for key, value in table.items()}


def decoding_failure_share(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(row.get("behavior_label") == "decoding_failure" for row in rows) / len(rows)


def genuine_style_gap_pp(rows: Sequence[dict[str, Any]]) -> float:
    harmful = [
        row
        for row in rows
        if row.get("safety_label") == "harmful"
        and row.get("behavior_label") != "decoding_failure"
    ]
    neutral = [row for row in harmful if row.get("style") == "neutral"]
    styled = [row for row in harmful if row.get("style") != "neutral"]
    return 100 * (refusal_rate([row["behavior_label"] for row in neutral]) - refusal_rate(
        [row["behavior_label"] for row in styled]
    ))


def _activation_index(metadata: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    return {
        (str(row["item_id"]), str(row["safety_label"]), str(row["style"])): int(
            row["activation_index"]
        )
        for row in metadata
    }


def compute_style_escape_metrics(
    activations: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    axis: np.ndarray,
) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    idx = _activation_index(metadata)
    meta_by_key = {
        (str(row["item_id"]), str(row["safety_label"]), str(row["style"])): row
        for row in metadata
    }
    scores = []
    labels = []
    for item_id, safety_label, style in idx:
        if safety_label != "harmful" or style == "neutral":
            continue
        keys = {
            "hs": (item_id, "harmful", style),
            "hn": (item_id, "harmful", "neutral"),
            "bs": (item_id, "benign", style),
            "bn": (item_id, "benign", "neutral"),
        }
        if not all(key in idx for key in keys.values()):
            continue
        score = escape_scores(
            activations[[idx[keys["hs"]]]],
            activations[[idx[keys["hn"]]]],
            activations[[idx[keys["bs"]]]],
            activations[[idx[keys["bn"]]]],
            axis,
        )[0]
        label = int(meta_by_key[keys["hs"]].get("behavior_label") == "harmful_compliance")
        scores.append(float(score))
        labels.append(label)
    if not scores or len(set(labels)) < 2:
        return {"escape_spearman": float("nan"), "escape_auroc": float("nan"), "n_escape": len(scores)}
    rho = spearmanr(scores, labels).statistic
    return {
        "escape_spearman": float(rho),
        "escape_auroc": float(roc_auc_score(labels, scores)),
        "n_escape": len(scores),
    }


def neutral_target_coordinates(
    activations: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    axis: np.ndarray,
) -> dict[tuple[str, str, str], float]:
    idx = _activation_index(metadata)
    coords = {}
    for item_id, safety_label, style in idx:
        neutral = (item_id, safety_label, "neutral")
        key = (item_id, safety_label, style)
        if neutral in idx:
            coords[key] = float(signed_occupancy(activations[idx[neutral]], axis))
    return coords


def save_axis(path: Path, axis: np.ndarray, site: Site) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, axis=unit_vector(axis), layer=site.layer, position=site.position)


def load_axis(path: Path) -> tuple[np.ndarray, Site]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RDO axis artifact: {path}. "
            "Run scripts/train_rdo_axis.py first with the same --run-name."
        )
    data = np.load(path, allow_pickle=True)
    return data["axis"], Site(layer=int(data["layer"]), position=str(data["position"]))


def save_selected_site(path: Path, site: Site, metrics: dict[str, Any]) -> None:
    save_json({"layer": site.layer, "position": site.position, "metrics": metrics}, path)


def load_selected_site(path: Path) -> Site:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected-site artifact: {path}. "
            "Run scripts/train_rdo_axis.py first with the same --run-name."
        )
    data = json.loads(path.read_text())
    return Site(layer=int(data["layer"]), position=str(data["position"]))


def finalize_gate_metrics(
    behavior_rows: Sequence[dict[str, Any]],
    intervention_records: Sequence[dict[str, Any]],
    style_metrics: dict[str, float],
    cfg: ExperimentConfig,
    *,
    rdo_beats_mdsteer_c2r: bool | None = None,
    rdo_beats_sarsteer_text: bool | None = None,
) -> dict[str, Any]:
    by_vector_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in intervention_records:
        by_vector_mode[(str(row["vector_name"]), str(row["intervention_mode"]))].append(row)

    rdo_add = by_vector_mode.get(("rdo_audio", "add"), [])
    rdo_ablate = by_vector_mode.get(("rdo_audio", "ablate"), [])
    rdo_restore = by_vector_mode.get(("rdo_audio", "set_coordinate"), [])

    heldout_add_base = [
        row
        for row in behavior_rows
        if row.get("safety_label") == "harmful"
        and row.get("behavior_label") == "harmful_compliance"
    ]
    benign_base = [row for row in behavior_rows if row.get("safety_label") == "benign"]
    ablate_base = [
        row
        for row in behavior_rows
        if row.get("safety_label") == "harmful" and row.get("behavior_label") == "policy_refusal"
    ]

    add_harm = intervention_metric_summary(heldout_add_base, rdo_add, safety_label="harmful")
    add_benign = intervention_metric_summary(benign_base, rdo_add, safety_label="benign")
    abl_harm = intervention_metric_summary(ablate_base, rdo_ablate, safety_label="harmful")
    restore_harm = intervention_metric_summary(heldout_add_base, rdo_restore, safety_label="harmful")
    restore_benign = intervention_metric_summary(benign_base, rdo_restore, safety_label="benign")

    metrics = AudioRdoGateMetrics(
        genuine_style_gap_pp=genuine_style_gap_pp(behavior_rows),
        decoding_failure_share=decoding_failure_share(behavior_rows),
        add_rr_pp=100
        * (add_harm["intervention_refusal_rate"] - add_harm["base_refusal_rate"]),
        benign_orr_add_pp=100
        * (add_benign["intervention_refusal_rate"] - add_benign["base_refusal_rate"]),
        ablation_asr_pp=100
        * (
            abl_harm["intervention_harmful_compliance_rate"]
            - abl_harm["base_harmful_compliance_rate"]
        ),
        rdo_beats_mdsteer_c2r=rdo_beats_mdsteer_c2r,
        rdo_beats_sarsteer_text=rdo_beats_sarsteer_text,
        escape_spearman=style_metrics.get("escape_spearman"),
        escape_auroc=style_metrics.get("escape_auroc"),
        restoration_rr_pp=100
        * (restore_harm["intervention_refusal_rate"] - restore_harm["base_refusal_rate"]),
        restored_fraction=restore_harm["intervention_refusal_rate"],
        benign_orr_restore_pp=100
        * (
            restore_benign["intervention_refusal_rate"]
            - restore_benign["base_refusal_rate"]
        ),
    )
    decision = decide_audio_rdo_gate(
        metrics,
        cfg.decision,
        max_decoding_failure_share=cfg.behavior.max_decoding_failure_share,
    )
    return {
        "behavior_decomposition": behavior_decomposition(behavior_rows),
        "metrics": metrics.__dict__,
        "style_escape": style_metrics,
        "decision": {"status": decision.status, "reasons": decision.reasons},
    }
