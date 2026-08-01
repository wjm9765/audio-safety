#!/usr/bin/env -S uv run python
"""Exp6: cut the prefill audio -> t_AB attention edge and see if y1 reverts.

Prefill only. See docs/experiments/exp6_audio_tab_edge/design.md for the frozen
decision table and integrity gates.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch

from audio_safety.config import load_exp4_config
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.models.hooks import (
    AttentionInputCapture,
    AudioEdgeIntervention,
    SpanStateIntervention,
    get_decoder_layers,
)
from audio_safety.models.qwen2_audio import load_qwen2_audio
from audio_safety.pipelines.exp3_qwen_mechanism import _prepare_aligned_pair, _prepare_arm
from audio_safety.pipelines.exp4_audio_kv_routing import (
    _physical_prefill,
    endpoint_index,
    greedy_logits_processors,
    select_next_token,
    shard_assignment,
)
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import resolve_paths, run_output_dir

CONDITIONS = ("no_patch", "identity", "host_edge", "wrong_edge")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/exp4_audio_kv_routing.yaml")
    )
    parser.add_argument("--source-exp4-run", default="exp4_20260731_2010_audio_kv")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_exp4_config(args.config)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    source_dir = paths.output_dir / cfg.exp4.source_exp3_run
    exp4_dir = paths.output_dir / args.source_exp4_run

    cohort = load_jsonl(source_dir / cfg.exp4.source_cohort_file)
    if args.max_pairs:
        cohort = cohort[: args.max_pairs]

    # Inherit the exact-length wrong-item assignment already used by Exp3.
    span_rows = load_jsonl(source_dir / cfg.exp4.source_span_patch_file)
    wrong_of = {
        str(r["pair_id"]): str(r["wrong_item_id"])
        for r in span_rows
        if int(r.get("layer", -1)) == cfg.exp4.inject_layer and r.get("condition") == "wrong_item"
    }
    # Wrong-item partners are drawn from the full pairs file, not the cohort:
    # 85 of the 113 Exp3 wrong-item partners lie outside the mechanism cohort.
    item_path = {}
    for row in load_jsonl(source_dir / cfg.exp3.artifacts.pairs_file):
        item_path.setdefault(str(row["item_id"]), (row["source_path"], str(row["source_arm"])))

    # Exp4's recorded y1 values are the gate-3 reference.
    exp4_rows = load_jsonl(exp4_dir / cfg.exp4.artifacts.records_file)
    exp4_y1 = {
        (str(r["pair_id"]), str(r["direction"])): (int(r["fixed_y1_id"]), int(r["host_y1_id"]))
        for r in exp4_rows
    }
    endpoints, roles = endpoint_index(cfg, exp4_dir)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    n_layers = len(get_decoder_layers(model))
    clamp_layers = list(range(cfg.exp4.relay_start_layer, n_layers))
    processors = greedy_logits_processors(model)

    # Per-shard checkpoint: concurrent shards sharing one path would clobber
    # each other, since each rewrites the whole file from its own state.
    suffix = (
        "" if args.shard_count == 1 else f".shard{args.shard_index:02d}_of_{args.shard_count:02d}"
    )
    records_path = run_dir / f"edge/records{suffix}.jsonl"
    existing = load_jsonl(records_path) if records_path.is_file() else []
    state = {(str(r["pair_id"]), str(r["direction"]), str(r["condition"])): r for r in existing}

    assigned = shard_assignment(cohort, shard_index=args.shard_index, shard_count=args.shard_count)
    wrong_cache: dict[str, dict] = {}
    wrong_native_len: dict[str, int] = {}
    done = 0

    for index, pair in assigned:
        pid = str(pair["pair_id"])
        if all((pid, d, c) in state for d in cfg.exp4.directions for c in CONDITIONS):
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        audio_positions = [int(v) for v in source["audio_positions"]]
        t_ab = int(source["t_ab"])
        relay_positions = list(range(audio_positions[-1] + 1, t_ab))
        bundles = {
            "source": _physical_prefill(
                cfg, model, source, relay_positions=relay_positions, clamp_layers=clamp_layers
            ),
            "target": _physical_prefill(
                cfg, model, target, relay_positions=relay_positions, clamp_layers=clamp_layers
            ),
        }

        # Length-matched wrong-item attention inputs, computed once per item.
        wrong_id = wrong_of.get(pid)
        wrong_states = None
        if wrong_id and wrong_id in item_path:
            key = f"{wrong_id}:{len(audio_positions)}"
            if key not in wrong_cache:
                arm = _prepare_arm(cfg, model, processor, item_path[wrong_id][0])
                w_positions = [int(v) for v in arm["audio_positions"]]
                cap = AttentionInputCapture(model, layers=clamp_layers, positions=w_positions)
                with torch.inference_mode(), cap:
                    model(**arm["inputs"], use_cache=True, return_dict=True)
                take = len(audio_positions)
                # Exp3's recorded `span_len` is the HOST span length, not the
                # partner's own. Partners shorter than the host are tiled to
                # length; see design.md §4 (2026-08-01).
                reps = max(1, -(-take // len(w_positions)))
                wrong_cache[key] = {
                    layer: (state.repeat(reps, 1) if reps > 1 else state)[:take].clone()
                    for layer, state in cap.states().items()
                }
                wrong_native_len[key] = len(w_positions)
            wrong_states = wrong_cache[key]
        # Fail closed: without this condition the frozen specificity contrast in
        # design.md §0 cannot be computed, and a silently missing arm would look
        # like a completed run.
        if wrong_states is None:
            raise RuntimeError(
                f"Exp6 could not build a length-matched wrong-item edge for {pid} "
                f"(wrong_item_id={wrong_id})"
            )

        for direction in cfg.exp4.directions:
            if all((pid, direction, c) in state for c in CONDITIONS):
                continue
            if direction == "source_to_target":
                host, host_b, donor_b = source, bundles["source"], bundles["target"]
                host_arm, donor_arm = str(pair["source_arm"]), str(pair["target_arm"])
            else:
                host, host_b, donor_b = target, bundles["target"], bundles["source"]
                host_arm, donor_arm = str(pair["target_arm"]), str(pair["source_arm"])
            prompt_ids = host["inputs"]["input_ids"]

            def injected_ctx(host=host, donor_b=donor_b, host_b=host_b, relay=relay_positions):
                ctx = [
                    SpanStateIntervention(
                        model,
                        layer_idx=cfg.exp4.inject_layer,
                        positions=host["audio_positions"],
                        replacement=donor_b.audio_state,
                    )
                ]
                ctx.extend(
                    SpanStateIntervention(
                        model,
                        layer_idx=layer,
                        positions=relay,
                        replacement=host_b.relay_states[layer],
                    )
                    for layer in clamp_layers
                )
                return ctx

            def run(extra=(), capture=None, host=host, ctx_factory=None):
                with torch.inference_mode(), contextlib.ExitStack() as stack:
                    for c in (ctx_factory or injected_ctx)():
                        stack.enter_context(c)
                    for c in extra:
                        stack.enter_context(c)
                    if capture is not None:
                        stack.enter_context(capture)
                    return model(**host["inputs"], use_cache=True, return_dict=True)

            cap = AttentionInputCapture(model, layers=clamp_layers, positions=audio_positions)
            base = run(capture=cap)
            inj_states = cap.states()
            base_logits = base.logits[0, -1].detach().clone()
            base_y1 = select_next_token(processors, prompt_ids, base_logits)
            host_y1 = select_next_token(processors, prompt_ids, host_b.last_logits)

            host_cap = AttentionInputCapture(model, layers=clamp_layers, positions=audio_positions)
            with torch.inference_mode(), host_cap:
                model(**host["inputs"], use_cache=True, return_dict=True)
            host_states = host_cap.states()

            ref = exp4_y1.get((pid, direction))
            if ref is not None and (base_y1, host_y1) != ref:
                raise RuntimeError(
                    f"Exp6 y1 does not reproduce Exp4 for {pid} {direction}: "
                    f"{(base_y1, host_y1)} vs {ref}"
                )

            for condition in CONDITIONS:
                if (pid, direction, condition) in state:
                    continue
                if condition == "no_patch":
                    y1, exact, counts = base_y1, None, []
                else:
                    states = {
                        "identity": inj_states,
                        "host_edge": host_states,
                        "wrong_edge": wrong_states,
                    }[condition]
                    if states is None:
                        continue
                    ctx = [
                        AudioEdgeIntervention(
                            model,
                            layer_idx=layer,
                            source_positions=audio_positions,
                            target_position=t_ab,
                            replacement=states[layer],
                        )
                        for layer in clamp_layers
                    ]
                    out = run(extra=ctx)
                    counts = [c.applied_count for c in ctx]
                    logits = out.logits[0, -1].detach()
                    y1 = select_next_token(processors, prompt_ids, logits)
                    if condition == "identity":
                        exact = float((logits - base_logits).abs().max()) == 0.0
                        if not exact:
                            raise RuntimeError(f"Exp6 identity gate failed for {pid} {direction}")
                    else:
                        exact = None
                    if any(c != 1 for c in counts):
                        raise RuntimeError(f"Exp6 edge hooks fired {counts} times")

                state[(pid, direction, condition)] = {
                    "schema_version": "exp6.v1",
                    "pair_id": pid,
                    "item_id": pair["item_id"],
                    "role": pair["role"],
                    "selection_role": roles[pid],
                    "direction": direction,
                    "condition": condition,
                    "host_arm": host_arm,
                    "donor_arm": donor_arm,
                    "wrong_item_id": wrong_id,
                    "wrong_item_native_span": wrong_native_len.get(
                        f"{wrong_id}:{len(audio_positions)}"
                    ),
                    "n_audio_positions": len(audio_positions),
                    "t_ab": t_ab,
                    "y1_id": y1,
                    "injected_y1_id": base_y1,
                    "host_y1_id": host_y1,
                    "injected_y1_changed_from_host": base_y1 != host_y1,
                    "reverted_to_host_y1": y1 == host_y1,
                    "identity_exact": exact,
                    "hook_counts_ok": all(c == 1 for c in counts) if counts else None,
                    "host_explicit_refusal": bool(endpoints[(pid, host_arm)]["explicit_refusal"]),
                    "donor_explicit_refusal": bool(endpoints[(pid, donor_arm)]["explicit_refusal"]),
                }
                done += 1
        if done:
            atomic_save_jsonl([state[k] for k in sorted(state)], records_path)
        print(
            f"[exp6] shard {args.shard_index}/{args.shard_count} "
            f"pair {index + 1}/{len(cohort)} rows={done}",
            flush=True,
        )

    atomic_save_jsonl([state[k] for k in sorted(state)], records_path)
    save_json(
        {"conditions": list(CONDITIONS), "n_rows": len(state)}, run_dir / "provenance_exp6.json"
    )
    print(f"[exp6] done rows={len(state)} -> {records_path}", flush=True)


if __name__ == "__main__":
    main()
