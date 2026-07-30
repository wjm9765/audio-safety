#!/usr/bin/env -S uv run python
"""Does the L10 audio effect travel through the 15 relay tokens, or around them?

Post-hoc routing test on an existing Exp3 run. Does not modify the pre-registered
pipeline.

Prompt geometry (measured):

    [0 .. a0-1]    system + "Audio 1:"   -- causally CANNOT attend to the audio
    [a0 .. a1]     A = audio tokens      -- the L10 injection site
    [a1+1 .. t-1]  R = relay set         -- 15 tokens: the instruction text and
                                            the chat control tokens
    [t]            t_AB                  -- the readout position

So the "audio -> text -> answer" route bottlenecks through just 15 positions.
Clamping every R position back to its HOST state after every block from L11 to
the last layer removes exactly the injection-induced change in R.

Why a state clamp is equivalent to closing A->R edges here: in a causal
transformer, contaminated R output at layer L cannot affect other positions
within that same layer -- they read R's layer-INPUT state. Clamping R after every
block therefore destroys the contamination before it can become R's K/V at the
next block, and by induction nothing leaks through R. R has no other contaminated
predecessor in this geometry. Unlike clamping t_AB, R is not the readout, so no
outcome can be forced algebraically through the unembedding.

Arms:
    identity        inject the HOST span at L10 (self-patch)   -- exact no-op
    open            inject the DONOR span at L10               -- the effect
    relay_closed    open + clamp R to host at L11..last
    sham_preaudio   open + clamp the pre-audio positions to host at L11..last

The sham is free and strong: pre-audio positions cannot attend to A, so in the
injected run they already equal host. Clamping them is replacing a state with
itself, and must reproduce `open` exactly -- if it does not, the clamping
machinery itself is perturbing the run.

Reading (frozen before running):
    retention r = donorward(relay_closed) / donorward(open), pair-clustered
    r CI entirely <= 0.20  -> relay transmission is NECESSARY
    r CI entirely >= 0.80  -> R-avoiding routes are SUFFICIENT
    otherwise              -> mixed/ambiguous; report the estimate, no binary claim

"R-avoiding sufficient" does NOT mean "direct A->t_AB": generated positions can
attend to A themselves and relay through earlier generated positions.

Usage:
    ./scripts/exp3/relay_closure.py --run-name exp3_YYYYMMDD_HHMM_tag
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from audio_safety.config import load_exp3_config
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _artifact,
    _behavior_index,
    _capture_audio_spans,
    _checkpoint_mapping,
    _contrast,
    _generate_from_inputs,
    _mechanism_cohort,
    _oriented,
    _prepare_aligned_pair,
    _validated_physical_baselines,
    readout_token_ids,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

ARTIFACT = Path("relay_closure/records.jsonl")
INJECT_LAYER = 10
CONDITIONS = ("identity", "open", "relay_closed", "sham_preaudio")
NON_LATIN = re.compile(r"[一-鿿぀-ヿ가-힯]")


def degenerate(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(NON_LATIN.findall(stripped)) > 10:
        return "non_latin"
    tokens = stripped.split()
    for size in (8, 12):
        if len(tokens) >= 2 * size:
            for start in range(len(tokens) - 2 * size + 1):
                if tokens[start : start + size] == tokens[start + size : start + 2 * size]:
                    return "verbatim_repeat"
    return None


def capture_positions(model, prepared, layers, positions):
    """Capture full residual states at arbitrary positions (not just the audio span)."""
    import torch

    from audio_safety.models.hooks import AudioSpanCapture

    capture = AudioSpanCapture(model, layers=layers, positions=positions)
    with torch.inference_mode(), capture:
        model(**prepared["inputs"], use_cache=False, return_dict=True)
    return {
        layer: state.numpy().astype("float32", copy=False)
        for layer, state in capture.states().items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml"),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_exp3_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths)
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    if not (run_dir / cfg.exp3.artifacts.span_patch_file).is_file():
        raise FileNotFoundError("run has no span-patch records; run the pilot first")

    gate = cfg.exp3.span_patch
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    frozen_behavior = _behavior_index(cfg, run_dir)

    path = _artifact(run_dir, ARTIFACT)
    existing = load_jsonl(path) if path.is_file() else []
    state = {(str(r["pair_id"]), str(r["direction"]), str(r["condition"])): r for r in existing}

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("relay closure requires CUDA")
    from audio_safety.models.hooks import SpanStateIntervention, get_decoder_layers
    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    n_layers = len(get_decoder_layers(model))
    clamp_layers = list(range(INJECT_LAYER + 1, n_layers))
    print(f"[exp3-relay] clamping layers {clamp_layers[0]}..{clamp_layers[-1]}", flush=True)

    completed = 0
    for index, pair in enumerate(cohort):
        pending = [
            (d, c)
            for d in gate.directions
            for c in CONDITIONS
            if (str(pair["pair_id"]), d, c) not in state
        ]
        if not pending:
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        src_inj = _capture_audio_spans(model, source, [INJECT_LAYER])
        tgt_inj = _capture_audio_spans(model, target, [INJECT_LAYER])

        # Relay / pre-audio position groups, derived per pair from the geometry.
        audio_pos = source["audio_positions"]
        t_ab = int(source["t_ab"])
        relay_pos = list(range(max(audio_pos) + 1, t_ab))
        pre_pos = list(range(0, min(audio_pos)))
        if not relay_pos or not pre_pos:
            raise RuntimeError(f"degenerate geometry for {pair['pair_id']}")

        src_relay = capture_positions(model, source, clamp_layers, relay_pos)
        tgt_relay = capture_positions(model, target, clamp_layers, relay_pos)
        src_pre = capture_positions(model, source, clamp_layers, pre_pos)
        tgt_pre = capture_positions(model, target, clamp_layers, pre_pos)

        source_baseline, target_baseline = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            frozen_behavior=frozen_behavior,
        )

        for direction, condition in pending:
            host, _donor, host_inj, donor_inj = _oriented(
                direction, source, target, src_inj, tgt_inj
            )
            _a, _b, host_relay, _dr = _oriented(direction, source, target, src_relay, tgt_relay)
            _c, _d, host_pre, _dp = _oriented(direction, source, target, src_pre, tgt_pre)
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )

            span = donor_inj if condition != "identity" else host_inj
            contexts: list[object] = [
                SpanStateIntervention(
                    model,
                    layer_idx=INJECT_LAYER,
                    positions=host["audio_positions"],
                    replacement=span[INJECT_LAYER],
                )
            ]
            if condition in {"relay_closed", "sham_preaudio"}:
                states = host_relay if condition == "relay_closed" else host_pre
                group = relay_pos if condition == "relay_closed" else pre_pos
                for layer in clamp_layers:
                    contexts.append(
                        SpanStateIntervention(
                            model,
                            layer_idx=layer,
                            positions=group,
                            replacement=states[layer],
                        )
                    )

            generated = _generate_from_inputs(
                cfg,
                model,
                processor,
                host["inputs"],
                refusal_ids,
                nonrefusal_ids,
                contexts=contexts,
            )
            state[(str(pair["pair_id"]), direction, condition)] = {
                "schema_version": cfg.exp3.schema_version,
                "stage": "relay_closure",
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "transition": pair.get("transition"),
                "selection_role": pair.get("selection_role"),
                "direction": direction,
                "condition": condition,
                "n_audio_positions": len(audio_pos),
                "n_relay_positions": len(relay_pos),
                "n_pre_positions": len(pre_pos),
                "clamp_layers": [clamp_layers[0], clamp_layers[-1]],
                "baseline_response": baseline["response"],
                "baseline_explicit_refusal": baseline["explicit_refusal"],
                "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                "response": generated["response"],
                "explicit_refusal": generated["explicit_refusal"],
                "r_tab_margin": generated["r_tab_margin"],
                "n_hooks": len(contexts),
                "applied_counts_ok": all(c == 1 for c in generated["applied_counts"]),
                "quality_flag": degenerate(generated["response"]),
                "identity_exact": (
                    generated["response"] == baseline["response"]
                    if condition == "identity"
                    else None
                ),
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=20)
        print(f"[exp3-relay] pair {index + 1}/{len(cohort)} cells={completed}", flush=True)

    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    bad = [r for r in state.values() if not r["applied_counts_ok"]]
    if bad:
        raise RuntimeError(f"{len(bad)} cells did not apply every hook exactly once")
    ident_bad = [r for r in state.values() if r["identity_exact"] is False]
    if ident_bad:
        raise RuntimeError(f"{len(ident_bad)} identity cells did not reproduce the baseline")
    # The pre-audio sham must be an exact no-op: those positions cannot attend to
    # the audio, so clamping them to host replaces each state with itself.
    sham_mismatch = 0
    for key, row in state.items():
        if row["condition"] != "sham_preaudio":
            continue
        open_row = state.get((key[0], key[1], "open"))
        if open_row and open_row["response"] != row["response"]:
            sham_mismatch += 1
    flagged = sum(1 for r in state.values() if r["quality_flag"])
    print(
        f"[exp3-relay] done cells={len(state)} sham_mismatch={sham_mismatch} "
        f"quality_flagged={flagged} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
