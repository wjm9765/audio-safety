#!/usr/bin/env -S uv run python
"""Run Exp4 post-prefill audio/t_AB KV-cache routing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from audio_safety.config import load_exp4_config
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.pipelines.exp3_qwen_mechanism import refusal_matcher_sha
from audio_safety.pipelines.exp4_audio_kv_routing import (
    analyze_cache_routing,
    freeze_source_inputs,
    merge_shard_records,
    run_cache_routing,
)
from audio_safety.utils.io import (
    get_git_commit,
    get_git_dirty,
    load_jsonl,
    save_json,
    snapshot_config,
)
from audio_safety.utils.paths import resolve_paths, run_output_dir

STAGES = ("preflight", "run", "merge", "analyze", "all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp4_audio_kv_routing.yaml"),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="exp4_YYYYMMDD_HHMM_tag; reuse for resumable stage invocations",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of concurrent worker processes covering the frozen cohort",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this worker; cohort pairs are assigned round-robin",
    )
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must lie in [0, --shard-count)")
    if args.shard_count > 1 and args.stage == "all":
        parser.error(
            "sharded execution must be staged explicitly: run 'preflight' once, then one "
            "'run' per shard, then 'merge', then 'analyze'"
        )
    return args


def _shard_errors_file(cfg, args) -> Path:
    """Keep concurrent shards from overwriting each other's failure record."""
    errors = Path(cfg.exp4.artifacts.errors_file)
    if args.shard_count == 1:
        return errors
    return errors.with_name(
        f"{errors.stem}.shard{args.shard_index:02d}_of_{args.shard_count:02d}{errors.suffix}"
    )


def _load_model(cfg, paths):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Exp4 run/all stages require CUDA; preflight/analyze are CPU-only")
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    # `device_map: auto` allocates against free memory, so a co-resident shard
    # can silently push layers to CPU. That changes kernels and would break the
    # tolerance-zero reproduction gates, so fail instead of running degraded.
    offloaded = sorted(
        {
            name.rsplit(".", 1)[0]
            for name, parameter in model.named_parameters()
            if parameter.device.type != "cuda"
        }
    )
    if offloaded:
        raise RuntimeError(
            f"Exp4 requires the whole model resident on GPU; {len(offloaded)} module(s) "
            f"are not on CUDA, first={offloaded[:3]}"
        )
    return model, processor


def _assert_or_write_snapshot(cfg, run_dir: Path) -> None:
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.is_file():
        snapshot_config(cfg, run_dir)
        return
    frozen = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    if frozen.get("config") != cfg.model_dump(mode="json"):
        raise RuntimeError("resolved config differs from this Exp4 run's frozen snapshot")
    frozen_commit, current_commit = frozen.get("git_commit"), get_git_commit()
    if frozen_commit and current_commit and frozen_commit != current_commit:
        raise RuntimeError("git commit differs from this Exp4 run's frozen snapshot")


def _runtime_provenance(cfg, model, processor) -> dict:
    import torch
    import transformers

    tokenizer = processor.tokenizer
    return {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "model_id": cfg.model.model_id,
        "model_revision": cfg.model.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "dtype": cfg.model.dtype,
        "attention_implementation": cfg.model.attn_implementation,
        "decoding": "manual greedy after fixed y1",
        "max_new_tokens": cfg.exp4.max_new_tokens,
        "instruction": cfg.exp3.instruction,
        "system_prompt": cfg.exp3.system_prompt,
        "matcher_sha256": refusal_matcher_sha(cfg.exp3.readout.refusal_patterns),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> None:
    args = _parse_args()
    if re.fullmatch(r"exp4_\d{8}_\d{4}_[a-z0-9_]+", args.run_name) is None:
        raise ValueError("--run-name must match exp4_YYYYMMDD_HHMM_tag")
    cfg = load_exp4_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    will_run_gpu = args.stage in {"run", "all"}
    if will_run_gpu:
        dirty = get_git_dirty()
        if dirty is not False:
            state = "not a git checkout" if dirty is None else "has uncommitted changes"
            raise RuntimeError(
                f"Exp4 GPU runs require a clean immutable checkout; worktree {state}"
            )
    _assert_or_write_snapshot(cfg, run_dir)

    if args.stage in {"preflight", "all"}:
        cohort = freeze_source_inputs(cfg, paths, run_dir)
        discordant = sum(row.get("selection_role") == "discordant" for row in cohort)
        print(
            f"[exp4] preflight pairs={len(cohort)} discordant={discordant} "
            f"source={cfg.exp4.source_exp3_run}",
            flush=True,
        )

    if will_run_gpu:
        # Re-hash at the execution boundary as well as preflight. A source run
        # mutated between separate invocations must not enter GPU inference.
        freeze_source_inputs(cfg, paths, run_dir)
        model, processor = _load_model(cfg, paths)
        provenance = _runtime_provenance(cfg, model, processor)
        provenance_path = run_dir / "provenance.json"
        if provenance_path.is_file():
            frozen_provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
            if frozen_provenance != provenance:
                raise RuntimeError("runtime provenance changed; use a new Exp4 run name")
        else:
            save_json(provenance, provenance_path)
        try:
            rows = run_cache_routing(
                cfg,
                paths,
                run_dir,
                model,
                processor,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            print(
                f"[exp4] shard {args.shard_index}/{args.shard_count} "
                f"cache-routing cells={len(rows)}",
                flush=True,
            )
        except Exception as exc:
            error_path = run_dir / _shard_errors_file(cfg, args)
            errors = load_jsonl(error_path) if error_path.is_file() else []
            errors.append(
                {
                    "stage": "run",
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_save_jsonl(errors, error_path)
            raise
        finally:
            del model, processor
            import torch

            torch.cuda.empty_cache()

    if args.stage == "merge":
        rows = merge_shard_records(cfg, run_dir, shard_count=args.shard_count)
        print(f"[exp4] merged shards={args.shard_count} cells={len(rows)}", flush=True)

    if args.stage in {"analyze", "all"}:
        metrics = analyze_cache_routing(cfg, run_dir)
        decisions = {name: values.get("decision") for name, values in metrics["contrasts"].items()}
        print(f"[exp4] decisions={decisions}", flush=True)
    print(f"[exp4] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
