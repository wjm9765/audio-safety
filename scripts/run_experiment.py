#!/usr/bin/env -S uv run python
"""CLI entry point for experiment runs. Thin by design (AGENTS.md): parses args,
resolves config/paths/seed, snapshots reproducibility info, dispatches to pipelines.

Example:
    ./scripts/run_experiment.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --run-name exp1_20260704_1200_audio_rdo_gate \
        --override stats.n_permutations=1000
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.utils.io import snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir
from audio_safety.utils.seed import set_seed

STAGES = (
    "pairs",
    "style_variants",
    "render_audio",
    "score_transcripts",
    "behavior",
    "rdo",
    "baselines",
    "extract_activations",
    "style_escape",
    "restoration",
    "stats",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. stats.n_permutations=1000)",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--output-dir", type=Path, default=None, help="override output root")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override cache root")
    parser.add_argument("--limit", type=int, default=None, help="limit rows for supported stages")
    parser.add_argument("--dry-run", action="store_true", help="supported by render_audio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(
        cfg.paths,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    set_seed(cfg.seed)

    run_name = args.run_name or f"{cfg.name}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)
    print(f"[run] {run_name}")
    print(f"[run] config snapshot -> {run_dir / 'config_snapshot.yaml'}")

    if args.stage == "pairs":
        from audio_safety.data import generate_pair_manifest, load_harmful_seed_rows

        seed_rows = load_harmful_seed_rows(
            paths.data_dir / cfg.dataset.seed_file,
            source=cfg.dataset.harmful_source,
        )
        output_path = paths.data_dir / cfg.dataset.source_file
        pairs = generate_pair_manifest(
            seed_rows,
            cfg.dataset.pair_generation,
            output_path,
            limit=args.limit or cfg.dataset.n_pairs,
        )
        print(f"[pairs] wrote {len(pairs)} draft pairs -> {output_path}")
        return

    if args.stage == "style_variants":
        from audio_safety.data import (
            generate_style_variant_manifest,
            load_audio_rdo_pairs,
            style_rows_from_pairs,
        )

        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        if args.limit is not None:
            selected_pairs = pairs[: args.limit]
        else:
            selected_pairs = pairs[: cfg.dataset.n_pairs]
        rows = style_rows_from_pairs(selected_pairs, safety_label="both")
        output_path = paths.data_dir / cfg.dataset.style_variant_generation.output_file
        records = generate_style_variant_manifest(
            rows,
            cfg.dataset.style_variant_generation,
            output_path,
        )
        print(f"[style] wrote {len(records)} style variant records -> {output_path}")
        return

    if args.stage == "render_audio":
        from audio_safety.data import load_audio_rdo_pairs, render_audio_records

        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        records = render_audio_records(
            pairs[: args.limit] if args.limit else pairs,
            cfg.dataset,
            cfg.dataset.tts,
            paths.data_dir,
            dry_run=args.dry_run,
        )
        print(f"[render] wrote {len(records)} render records")
        return

    if args.stage == "score_transcripts":
        from audio_safety.data import score_transcript_manifest

        scored = score_transcript_manifest(paths.data_dir, cfg.dataset)
        passed = sum(bool(row.get("transcript_control_passed")) for row in scored)
        print(f"[asr] transcript-control passed: {passed}/{len(scored)}")
        return

    if args.stage == "behavior":
        from audio_safety.evaluation import label_behavior_records
        from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
        from audio_safety.utils.io import load_jsonl, save_jsonl

        rows = [
            row
            for row in load_jsonl(paths.data_dir / cfg.dataset.asr.scored_manifest_file)
            if bool(row.get("transcript_control_passed"))
        ]
        if args.limit is not None:
            rows = rows[: args.limit]
        model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
        outputs = []
        for row in rows:
            updated = dict(row)
            updated["output"] = generate_audio_response(
                model,
                processor,
                paths.data_dir / str(row["path"]),
                cfg.dataset.target_generation.instruction,
                max_new_tokens=cfg.dataset.target_generation.max_new_tokens,
            )
            outputs.append(updated)
        labeled = label_behavior_records(outputs)
        output_path = paths.data_dir / cfg.dataset.target_generation.outputs_file
        save_jsonl(labeled, output_path)
        print(f"[behavior] wrote {len(labeled)} outputs -> {output_path}")
        return

    if args.stage == "rdo":
        from audio_safety.data import load_audio_rdo_pairs
        from audio_safety.models.qwen2_audio import load_qwen2_audio
        from audio_safety.pipelines.rdo_gate import (
            Site,
            rows_for_split,
            save_axis,
            save_selected_site,
            split_ids,
            train_and_validate_site,
        )
        from audio_safety.utils.io import load_jsonl, save_json

        limit = args.limit if args.limit is not None else cfg.rdo.limit_per_site
        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        split_map = split_ids(pairs, cfg)
        rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
        train_rows = rows_for_split(rows, split_map, "train")
        validation_rows = rows_for_split(rows, split_map, "validation")
        model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
        best_axis = best_site = best_metrics = None
        all_metrics = []
        for site in [
            Site(layer=layer, position=position)
            for layer in cfg.hidden.layers
            for position in cfg.hidden.positions
        ]:
            axis, metrics = train_and_validate_site(
                model,
                processor,
                train_rows,
                validation_rows,
                cfg,
                paths.data_dir,
                site,
                limit=limit,
            )
            all_metrics.append(metrics)
            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_axis, best_site, best_metrics = axis, site, metrics
        if best_axis is None or best_site is None or best_metrics is None:
            raise RuntimeError("no RDO site was trained")
        save_axis(run_dir / cfg.rdo.axis_file, best_axis, best_site)
        save_selected_site(run_dir / cfg.rdo.selected_site_file, best_site, best_metrics)
        save_json(
            {"candidates": all_metrics, "selected": best_metrics},
            run_dir / cfg.rdo.validation_metrics_file,
        )
        print(f"[rdo] selected layer={best_site.layer} position={best_site.position}")
        return

    if args.stage in {"baselines", "extract_activations"}:
        import numpy as np

        from audio_safety.data import load_audio_rdo_pairs
        from audio_safety.models.qwen2_audio import load_qwen2_audio
        from audio_safety.pipelines.rdo_gate import (
            capture_refusal_continuation_hidden,
            compute_baseline_vectors,
            extract_selected_site_activations,
            load_selected_site,
            split_ids,
        )
        from audio_safety.utils.io import load_jsonl

        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        split_map = split_ids(pairs, cfg)
        rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
        site = load_selected_site(run_dir / cfg.rdo.selected_site_file)
        model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
        activations, metadata = extract_selected_site_activations(
            model, processor, rows, cfg, paths.data_dir, run_dir, site
        )
        train_harmful = [
            row
            for row in rows
            if row["item_id"] in split_map["train"] and row["safety_label"] == "harmful"
        ]
        sar_hidden = np.stack(
            [
                capture_refusal_continuation_hidden(
                    model, processor, row, cfg, paths.data_dir, site
                )
                for row in train_harmful
            ]
        )
        vectors = compute_baseline_vectors(
            activations,
            metadata,
            split_map["train"],
            sar_refusal_hidden=sar_hidden,
        )
        baseline_path = run_dir / cfg.rdo.baseline_vectors_file
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(baseline_path, **vectors)
        print(f"[baselines] saved {list(vectors)} -> {baseline_path}")
        return

    if args.stage in {"style_escape", "restoration", "stats"}:
        import numpy as np

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

        def valid(rows):
            return [
                row
                for row in rows
                if row.get("behavior_label") != "decoding_failure"
                and bool(row.get("transcript_control_passed", True))
            ]

        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        split_map = split_ids(pairs, cfg)
        rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
        heldout = valid(rows_for_split(rows, split_map, "heldout"))
        if args.limit is not None:
            heldout = heldout[: args.limit]
        axis, site = load_axis(run_dir / cfg.rdo.axis_file)
        baseline_npz = np.load(run_dir / cfg.rdo.baseline_vectors_file)
        activations = np.load(run_dir / cfg.rdo.activations_file)["activations"]
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
            add_metrics[name] = {
                "harmful": intervention_metric_summary(
                    [r for r in add_base if r["safety_label"] == "harmful"],
                    generated,
                    safety_label="harmful",
                ),
                "benign": intervention_metric_summary(
                    [r for r in add_base if r["safety_label"] == "benign"],
                    generated,
                    safety_label="benign",
                ),
            }
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
        target_coords = neutral_target_coordinates(activations, metadata, axis)
        records.extend(
            generate_intervention_records(
                model,
                processor,
                add_base,
                cfg,
                paths.data_dir,
                site=site,
                vector=axis,
                vector_name="rdo_audio",
                mode="set_coordinate",
                target_coordinates=target_coords,
            )
        )

        def harmful_rr_pp(name: str) -> float:
            m = add_metrics[name]["harmful"]
            return 100 * (m["intervention_refusal_rate"] - m["base_refusal_rate"])

        def benign_orr_pp(name: str) -> float:
            m = add_metrics[name]["benign"]
            return 100 * (m["intervention_refusal_rate"] - m["base_refusal_rate"])

        def beats(name: str) -> bool | None:
            if name not in add_metrics:
                return None
            if benign_orr_pp(name) > benign_orr_pp("rdo_audio") + cfg.baselines.matched_orr_tolerance_pp:
                return True
            return harmful_rr_pp("rdo_audio") > harmful_rr_pp(name)

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
        print(f"[stats] decision: {metrics['decision']['status']}")
        return

    if args.stage == "all":
        raise NotImplementedError("run all stages explicitly so GPU failures are resumable")

    raise NotImplementedError(
        f"stage {args.stage!r} not wired yet — implement per design.md §6 skeleton"
    )


if __name__ == "__main__":
    main()
