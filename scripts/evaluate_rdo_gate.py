#!/usr/bin/env -S uv run python
"""Evaluate RDO/baseline interventions, style escape, restoration, and decision."""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.models.qwen2_audio import load_qwen2_audio
from audio_safety.pipelines.rdo_gate import (
    compute_style_escape_metrics,
    finalize_gate_metrics,
    generate_intervention_records,
    intervention_metric_summary,
    load_axis,
    neutral_target_coordinates,
    rows_for_split,
    split_ids,
)
from audio_safety.utils.io import load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug limit per eval group")
    return parser.parse_args()



def require_eval_artifacts(run_dir: Path, cfg, args: argparse.Namespace) -> None:
    train_missing = [
        path
        for path in (run_dir / cfg.rdo.axis_file, run_dir / cfg.rdo.selected_site_file)
        if not path.exists()
    ]
    extract_missing = [
        path
        for path in (
            run_dir / cfg.rdo.baseline_vectors_file,
            run_dir / cfg.rdo.activations_file,
            run_dir / cfg.rdo.activation_metadata_file,
        )
        if not path.exists()
    ]
    if not train_missing and not extract_missing:
        return
    parts = ["Missing artifacts for RDO evaluation."]
    if train_missing:
        parts.append("Training artifacts missing:")
        parts.extend(f"  - {path}" for path in train_missing)
        parts.append(
            "Run first:\n"
            f"  ./scripts/train_rdo_axis.py --config {args.config} --run-name {args.run_name}"
        )
    if extract_missing:
        parts.append("Activation/baseline artifacts missing:")
        parts.extend(f"  - {path}" for path in extract_missing)
        parts.append(
            "Then run:\n"
            f"  ./scripts/extract_rdo_activations.py --config {args.config} --run-name {args.run_name}"
        )
    raise SystemExit("\n".join(parts))

def _valid(rows):
    return [
        row
        for row in rows
        if row.get("behavior_label") != "decoding_failure"
        and bool(row.get("transcript_control_passed", True))
    ]


def _style_induced_rows(rows):
    neutral = {
        row["item_id"]: row
        for row in rows
        if row["safety_label"] == "harmful" and row["style"] == "neutral"
    }
    out = []
    for row in rows:
        if (
            row["safety_label"] == "harmful"
            and row["style"] != "neutral"
            and row["behavior_label"] == "harmful_compliance"
            and neutral.get(row["item_id"], {}).get("behavior_label") == "policy_refusal"
        ):
            out.append(row)
    return out


def _matching_benign(rows, harmful_rows):
    wanted = {(row["item_id"], row["style"]) for row in harmful_rows}
    return [
        row
        for row in rows
        if row["safety_label"] == "benign" and (row["item_id"], row["style"]) in wanted
    ]


def _add_metric(base_rows, records):
    harm_base = [
        row
        for row in base_rows
        if row["safety_label"] == "harmful" and row["behavior_label"] == "harmful_compliance"
    ]
    benign_base = [row for row in base_rows if row["safety_label"] == "benign"]
    harm = intervention_metric_summary(harm_base, records, safety_label="harmful")
    benign = intervention_metric_summary(benign_base, records, safety_label="benign")
    return {
        "harmful_rr_pp": 100 * (harm["intervention_refusal_rate"] - harm["base_refusal_rate"]),
        "benign_orr_pp": 100
        * (benign["intervention_refusal_rate"] - benign["base_refusal_rate"]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(
        cfg.paths,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    require_eval_artifacts(run_dir, cfg, args)
    rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    heldout = _valid(rows_for_split(rows, split_map, "heldout"))
    if args.limit is not None:
        heldout = heldout[: args.limit]

    axis, site = load_axis(run_dir / cfg.rdo.axis_file)
    baseline_npz = np.load(run_dir / cfg.rdo.baseline_vectors_file)
    activation_npz = np.load(run_dir / cfg.rdo.activations_file)
    activations = activation_npz["activations"]
    metadata = load_jsonl(run_dir / cfg.rdo.activation_metadata_file)
    style_metrics = compute_style_escape_metrics(activations, metadata, axis)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    add_base = [
        row
        for row in heldout
        if row["safety_label"] == "benign"
        or (row["safety_label"] == "harmful" and row["behavior_label"] == "harmful_compliance")
    ]
    ablate_base = [
        row
        for row in heldout
        if row["safety_label"] == "harmful" and row["behavior_label"] == "policy_refusal"
    ]

    records = []
    vectors = {"rdo_audio": axis}
    vectors.update({name: baseline_npz[name] for name in baseline_npz.files})
    rng = np.random.default_rng(cfg.seed)
    random_vec = rng.standard_normal(axis.shape)
    random_vec = random_vec / np.linalg.norm(random_vec)
    vectors["random"] = random_vec

    add_metrics = {}
    for name, vector in vectors.items():
        generated = generate_intervention_records(
            model,
            processor,
            add_base,
            cfg,
            paths.data_dir,
            site=site,
            vector=vector,
            vector_name=name,
            mode="add",
        )
        records.extend(generated)
        add_metrics[name] = _add_metric(add_base, generated)

    records.extend(
        generate_intervention_records(
            model,
            processor,
            ablate_base,
            cfg,
            paths.data_dir,
            site=site,
            vector=axis,
            vector_name="rdo_audio",
            mode="ablate",
        )
    )

    restore_harmful = _style_induced_rows(heldout)
    restore_rows = [*restore_harmful, *_matching_benign(heldout, restore_harmful)]
    target_coords = neutral_target_coordinates(activations, metadata, axis)
    records.extend(
        generate_intervention_records(
            model,
            processor,
            restore_rows,
            cfg,
            paths.data_dir,
            site=site,
            vector=axis,
            vector_name="rdo_audio",
            mode="set_coordinate",
            target_coordinates=target_coords,
        )
    )

    rdo_orr = add_metrics["rdo_audio"]["benign_orr_pp"]
    tolerance = cfg.baselines.matched_orr_tolerance_pp

    def beats(name: str) -> bool | None:
        if name not in add_metrics:
            return None
        baseline = add_metrics[name]
        if baseline["benign_orr_pp"] > rdo_orr + tolerance:
            return True
        return add_metrics["rdo_audio"]["harmful_rr_pp"] > baseline["harmful_rr_pp"]

    metrics = finalize_gate_metrics(
        heldout,
        records,
        style_metrics,
        cfg,
        rdo_beats_mdsteer_c2r=beats("mdsteer_c2r"),
        rdo_beats_sarsteer_text=beats("sarsteer_text"),
    )
    metrics["matched_orr_curves"] = add_metrics
    save_jsonl(records, run_dir / cfg.rdo.intervention_outputs_file)
    save_json(metrics, run_dir / cfg.rdo.metrics_file)
    print(f"[eval] wrote interventions -> {run_dir / cfg.rdo.intervention_outputs_file}")
    print(f"[eval] wrote metrics -> {run_dir / cfg.rdo.metrics_file}")
    print(f"[eval] decision: {metrics['decision']['status']}")


if __name__ == "__main__":
    main()
