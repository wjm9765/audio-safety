#!/usr/bin/env -S uv run python
"""Run the config-driven Exp3 Qwen refusal-instability mechanism pipeline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from audio_safety.config import load_exp3_config
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.pipelines.exp3_qwen_mechanism import (
    analyze_all,
    assert_input_dose_gate,
    assert_span_patch_gate,
    build_pair_manifest,
    collect_behavior,
    readout_token_ids,
    refusal_matcher_sha,
    run_content_audit,
    run_content_patch_audit,
    run_escape_reset,
    run_input_dose,
    run_readout_patch,
    run_representation_capture,
    run_span_patch,
)
from audio_safety.utils.io import (
    get_git_commit,
    get_git_dirty,
    load_jsonl,
    save_json,
    snapshot_config,
)
from audio_safety.utils.paths import resolve_paths, run_output_dir

GPU_STAGE_ORDER = (
    "behavior",
    "content-audit",
    "input-dose",
    "capture",
    "span-patch",
    "escape-reset",
    "readout-patch",
)
GPU_STAGES = frozenset(GPU_STAGE_ORDER)
STAGES = [
    "preflight",
    *GPU_STAGE_ORDER,
    "analyze",
    "gpu-all",
    "all",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml"),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="exp3_YYYYMMDD_HHMM_tag; reused across resumable stage invocations",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _load_model(cfg, paths):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Exp3 GPU stages require CUDA; run preflight/analyze on CPU")
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    return model, processor


def _run_gpu_stage(stage, cfg, paths, run_dir, model, processor):
    if stage == "behavior":
        _, pending = collect_behavior(
            cfg,
            paths,
            run_dir,
            model=model,
            processor=processor,
            generate_missing=True,
        )
        if pending:
            raise RuntimeError(f"behavior still has {pending} pending rows")
    elif stage == "content-audit":
        run_content_audit(cfg, run_dir, model, processor)
        run_content_patch_audit(cfg, run_dir, model, processor)
    elif stage == "input-dose":
        run_input_dose(cfg, run_dir, model, processor)
    elif stage == "capture":
        assert_input_dose_gate(cfg, run_dir)
        run_representation_capture(cfg, run_dir, model, processor)
    elif stage == "span-patch":
        assert_input_dose_gate(cfg, run_dir)
        run_span_patch(cfg, run_dir, model, processor)
    elif stage == "escape-reset":
        assert_span_patch_gate(cfg, run_dir)
        run_escape_reset(cfg, run_dir, model, processor)
    elif stage == "readout-patch":
        run_readout_patch(cfg, run_dir, model, processor)
    else:
        raise ValueError(f"unknown GPU stage {stage!r}")


def main() -> None:
    args = _parse_args()
    if re.fullmatch(r"exp3_\d{8}_\d{4}_[a-z0-9_]+", args.run_name) is None:
        raise ValueError("--run-name must match exp3_YYYYMMDD_HHMM_tag")
    cfg = load_exp3_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    requested = args.stage
    will_run_gpu = requested in GPU_STAGES or requested in {"gpu-all", "all"}
    if will_run_gpu:
        dirty = get_git_dirty()
        if dirty is not False:
            state = "not a git checkout" if dirty is None else "has uncommitted changes"
            raise RuntimeError(
                f"Exp3 GPU runs require an immutable clean git checkout; worktree {state}"
            )
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot_config(cfg, run_dir)
    else:
        frozen_snapshot = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        frozen_config = frozen_snapshot["config"]
        current_config = cfg.model_dump(mode="json")
        if frozen_config != current_config:
            raise RuntimeError(
                "resolved config differs from this run's frozen config_snapshot.yaml; "
                "use a new --run-name"
            )
        frozen_commit = frozen_snapshot.get("git_commit")
        current_commit = get_git_commit()
        if frozen_commit and current_commit and frozen_commit != current_commit:
            raise RuntimeError(
                "git commit differs from this run's config snapshot; use a new --run-name"
            )

    if requested in {"preflight", "all"}:
        pairs = build_pair_manifest(cfg, paths, run_dir)
        rows, pending = collect_behavior(
            cfg,
            paths,
            run_dir,
            generate_missing=False,
        )
        print(
            f"[exp3] preflight pairs={len(pairs)} imported_behavior={len(rows)} "
            f"pending_behavior={pending}",
            flush=True,
        )

    gpu_stages = list(GPU_STAGE_ORDER) if requested in {"gpu-all", "all"} else [requested]
    gpu_stages = [stage for stage in gpu_stages if stage in GPU_STAGES]
    if gpu_stages:
        pair_path = run_dir / cfg.exp3.artifacts.pairs_file
        if not pair_path.is_file():
            build_pair_manifest(cfg, paths, run_dir)
        model, processor = _load_model(cfg, paths)
        import torch
        import transformers

        refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
        tokenizer = processor.tokenizer
        provenance = {
            "git_commit": get_git_commit(),
            "model_id": cfg.model.model_id,
            "model_revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attention_implementation": cfg.model.attn_implementation,
            "instruction": cfg.exp3.instruction,
            "system_prompt": cfg.exp3.system_prompt,
            "decoding": "greedy",
            "matcher_sha256": refusal_matcher_sha(cfg.exp3.readout.refusal_patterns),
            "refusal_first_token_ids": refusal_ids,
            "nonrefusal_first_token_ids": nonrefusal_ids,
            "refusal_first_token_text": [tokenizer.decode([value]) for value in refusal_ids],
            "nonrefusal_first_token_text": [tokenizer.decode([value]) for value in nonrefusal_ids],
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
            "resolved_tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "git_dirty": get_git_dirty(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        provenance_path = run_dir / "provenance.json"
        if provenance_path.is_file():
            if yaml.safe_load(provenance_path.read_text(encoding="utf-8")) != provenance:
                raise RuntimeError("runtime provenance changed; use a new --run-name")
        else:
            save_json(provenance, provenance_path)
        for stage in gpu_stages:
            print(f"[exp3] stage={stage}", flush=True)
            try:
                _run_gpu_stage(stage, cfg, paths, run_dir, model, processor)
            except Exception as exc:
                error_path = run_dir / cfg.exp3.artifacts.errors_file
                errors = load_jsonl(error_path) if error_path.is_file() else []
                errors.append(
                    {
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                atomic_save_jsonl(errors, error_path)
                raise
        del model, processor
        torch.cuda.empty_cache()

    if requested in {"analyze", "all"}:
        metrics = analyze_all(cfg, run_dir)
        print(f"[exp3] analysis sections={sorted(metrics)}", flush=True)
    print(f"[exp3] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
