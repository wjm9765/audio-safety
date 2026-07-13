#!/usr/bin/env -S uv run python
"""Run 4 causal-attribution: interchange-patching trace generator (Qwen2-Audio only).

DIRECTION-FINDING (not a §0 gate, not paper-facing). Tests whether the new paper
direction is ALIVE: does injecting a CLEAN-run residual state into the ATTACKED run
at the decision anchor (layer 16 / first_generation_prelogit) restore refusal on
genuine PAP flips, harmful-specifically, beyond identity / wrong-item / displacement
shams? Generation only — judging is scripts/judge_traces.py, adjudication is
scripts/analyze_causal_trace.py.

Conditions per flip item (design-locked; full-state donor is used verbatim, never
normalized): no_patch, identity(self), same_item(clean->attacked), wrong_item,
random_displacement (||delta||=||clean-attacked||), r_a_coord (concept patch at
r_A's site only), reverse (attacked->clean). Plus benign matched no_patch/same_item.

Correctness guards (from the cross-check): every patch asserts it fired exactly
once; the identity self-patch MUST reproduce no_patch verbatim under greedy decode
(checked in analysis); the original consensus-flip cohort is intention-to-treat.

Example:
    ./scripts/causal_trace_flip.py --config configs/experiments/run4_attack_flip.yaml \
        --axis-artifact /workspace/.../exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz \
        --run-name run4_causal_trace
"""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.attack_flip import classify
from audio_safety.evaluation.causal_trace import assert_unique_trace_ids, plan_primary_conditions
from audio_safety.models.qwen2_audio import (
    capture_audio_state,
    generate_audio_response,
    generate_audio_response_with_intervention,
    generate_audio_response_with_state_patch,
    load_qwen2_audio,
)
from audio_safety.pipelines.rdo_gate import load_axis
from audio_safety.utils.io import load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

HARMFUL, BENIGN = "harmful", "benign"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--axis-artifact", type=Path, default=None, help="frozen r_A for r_a_coord")
    p.add_argument("--run-name", type=str, default="run4_causal_trace")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _classify_by(judged, modality, style, safety):
    out = {}
    for r in judged:
        if (str(r.get("modality")) == modality and str(r.get("style")) == style
                and str(r.get("safety_label")) == safety):
            v = r.get("judge_verdicts") or {}
            out[str(r["item_id"])] = {m: (classify(vv) if vv else None) for m, vv in v.items()}
    return out


def _rows_by(judged, modality, safety):
    """(item_id, style) -> row, for one modality/safety."""
    out = {}
    for r in judged:
        if str(r.get("modality")) == modality and str(r.get("safety_label")) == safety:
            out[(str(r["item_id"]), str(r.get("style")))] = r
    return out


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    ct = cfg.causal_trace
    if ct is None or not ct.enabled:
        raise SystemExit("config has no enabled causal_trace block")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)
    instruction = cfg.dataset.target_generation.instruction
    max_new = ct.max_new_tokens
    layer, pos = ct.primary_layer, ct.primary_position
    clean_style, attack_style = ct.clean_style, ct.attack_style

    judges = list(ct.judge_models or cfg.conversion_gap.judge.models)
    judged = load_jsonl(paths.data_dir / cfg.conversion_gap.judge.outputs_file)

    neu = _classify_by(judged, "audio", clean_style, HARMFUL)
    att = _classify_by(judged, "audio", attack_style, HARMFUL)
    flips = sorted(
        i for i in neu if i in att
        and all(neu[i].get(m) == "refusal" for m in judges)
        and all(att[i].get(m) == "comply" for m in judges)
    )[: ct.max_flips]

    ben = _classify_by(judged, "audio", attack_style, BENIGN)
    benign_items = sorted(
        i for i in ben if all(ben[i].get(m) not in (None, "refusal") for m in judges)
    )[: ct.n_benign]

    h_rows = _rows_by(judged, "audio", HARMFUL)
    b_rows = _rows_by(judged, "audio", BENIGN)

    def ref_text(item, safety):
        row = (h_rows if safety == HARMFUL else b_rows).get((item, clean_style), {})
        return str(row.get("judge_request_text") or row.get("reference_text") or "")

    r_a = None
    if args.axis_artifact is not None:
        axis, site = load_axis(args.axis_artifact)
        if int(site.layer) == layer and str(site.position) == pos:
            r_a = np.asarray(axis, dtype=np.float32)
            r_a = r_a / max(float(np.linalg.norm(r_a)), 1e-12)
        else:
            print(f"[trace] r_A site {site.layer}/{site.position} != primary {layer}/{pos}; "
                  "skipping r_a_coord control (r_A is only a valid concept patch at its site)")

    print(f"[trace] consensus flips: {len(flips)}; benign matched: {len(benign_items)}; "
          f"primary cell L{layer}/{pos}; judges={judges}")
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    rng = np.random.default_rng(ct.seed)

    # Cache donor states once per item (clean + attacked at the primary cell).
    clean_state: dict[str, np.ndarray] = {}
    attacked_state: dict[str, np.ndarray] = {}

    def state(item, style, safety):
        rows = h_rows if safety == HARMFUL else b_rows
        row = rows.get((item, style))
        if row is None:
            return None
        s, _ = capture_audio_state(
            model, processor, paths.data_dir / str(row["path"]), instruction,
            layer_idx=layer, position_name=pos,
        )
        return s

    for item in flips:
        clean_state[item] = state(item, clean_style, HARMFUL)
        attacked_state[item] = state(item, attack_style, HARMFUL)

    # Plan specs (harmful side per flip; benign side handled per benign item).
    specs: list[dict] = []
    for k, item in enumerate(flips):
        wrong = flips[(k + 1) % len(flips)] if len(flips) > 1 else None
        specs.extend(plan_primary_conditions(
            flip_item=item, wrong_item=wrong, benign_item=None,
            layer=layer, position=pos, seed=ct.seed,
        ))
    for item in benign_items:
        specs.extend(plan_primary_conditions(
            flip_item=item, wrong_item=None, benign_item=item,
            layer=layer, position=pos, seed=ct.seed,
        )[-2:])  # only benign_no_patch + benign_same_item
    assert_unique_trace_ids(specs)

    def audio_path(item, style, safety):
        rows = h_rows if safety == HARMFUL else b_rows
        row = rows.get((item, style))
        return None if row is None else paths.data_dir / str(row["path"])

    records = []
    for spec in specs:
        item = spec["recipient_item"]
        safety = spec["recipient_safety"]
        cond = spec["condition"]
        try:
            if spec["recipient"] == "clean_harmful":       # reverse: patch into clean run
                recip_audio = audio_path(item, clean_style, HARMFUL)
            elif safety == BENIGN:
                recip_audio = audio_path(item, attack_style, BENIGN)
            else:
                recip_audio = audio_path(item, attack_style, HARMFUL)
            if recip_audio is None:
                continue

            if cond == "r_a_coord":
                if r_a is None:
                    continue
                target = float(np.dot(clean_state[item], r_a))
                out = generate_audio_response_with_intervention(
                    model, processor, recip_audio, instruction,
                    layer_idx=layer, position_name=pos, vector=r_a,
                    mode="set_coordinate", target_coordinate=target,
                    all_positions=False, max_new_tokens=max_new,
                )
            elif spec["operator"] == "none":
                out = generate_audio_response(model, processor, recip_audio, instruction,
                                              max_new_tokens=max_new)
            else:
                if cond == "identity":
                    donor = attacked_state[item]
                elif cond == "same_item":
                    donor = (
                        clean_state[item] if safety == HARMFUL
                        else state(item, clean_style, BENIGN)
                    )
                elif cond == "wrong_item":
                    donor = clean_state[spec["donor_item"]]
                elif cond == "reverse":
                    donor = attacked_state[item]
                elif cond == "random_displacement":
                    base = attacked_state[item]
                    disp = float(np.linalg.norm(clean_state[item] - base))
                    d = rng.standard_normal(base.shape).astype(np.float32)
                    d = d / max(float(np.linalg.norm(d)), 1e-12) * disp
                    donor = base + d
                else:
                    continue
                out = generate_audio_response_with_state_patch(
                    model, processor, recip_audio, instruction,
                    layer_idx=layer, position_name=pos, replacement_state=donor,
                    max_new_tokens=max_new, require_single_application=True,
                )
        except Exception as exc:  # noqa: BLE001 — long GPU run; log and skip a trace
            print(f"[trace] FAILED {spec['trace_id']}: {type(exc).__name__}: {exc}")
            continue
        rec = dict(spec)
        rec.update({
            "output": out,
            "judge_request_text": ref_text(item, safety),
            "modality": "audio",
        })
        records.append(rec)

    run_dir = run_output_dir(paths.output_dir, args.run_name)
    save_jsonl(records, run_dir / ct.records_file)
    save_json(
        {"flips": flips, "benign_items": benign_items, "primary_layer": layer,
         "primary_position": pos, "judges": judges, "n_records": len(records),
         "r_a_coord_enabled": r_a is not None},
        run_dir / "causal_trace_manifest.json",
    )
    print(f"[trace] wrote {len(records)} traces -> {run_dir / ct.records_file}")
    print(f"[trace] next: ./scripts/judge_traces.py --config {args.config} "
          f"--run-name {args.run_name}")


if __name__ == "__main__":
    main()
