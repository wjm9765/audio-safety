#!/usr/bin/env -S uv run python
"""Run 10 STEP 3 (GPU): channel-axis L18 confirmatory, bidirectional projected transport.

For each recognized-in-both clean/attack pair, fit a channel subspace U (mean-anchored SVD
of paired L18 clean-attack differences, per support) on train/dev ONLY, then on the TEST
pairs apply pair-specific projected transport at block-18 post-residual:

  Arm A (propagation): patch the whole audio-token span.
  Arm B (expression):  patch only the readout token t_AB = first_generation_prelogit.
  Restoration: forward on ATTACK audio, donor = CLEAN L18 states.
  Corruption:  forward on CLEAN audio, donor = ATTACK L18 states.
  Dose lambda scales the projected update. Endpoint = first-token refusal margin M at t_AB.

Controls: sham/orthogonal ensemble (cov-matched, perp to the refusal-DiM axis) + a
`refusal-DiM global` positive control (refused-complied DiM RECOMPUTED on this cohort,
added at all positions). See docs/.../run10_channel_invariance_audit_direction_20260719.md.

Reuses: models.hooks.ProjectedTransportIntervention / ResidualStreamIntervention,
pipelines.channel_axis (U estimator), pipelines.channel_patching (alignment guards, margin).
GPU-only. Writes <run>/channel_patch/l18_patch.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_pairs(pairs_path: Path, clean_style: str) -> list[dict]:
    """Group manifest rows into clean/attack pairs by (item_id, sign, attack style).

    Each manifest row needs: item_id, style, path, reference_text; optional sign,
    behavior_label ('refused'|'complied'), recognized_both (bool, from Step 2).
    """
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    clean_by_item: dict[str, dict] = {}
    attacks: list[dict] = []
    for row in rows:
        item = str(row["item_id"])
        if str(row["style"]) == clean_style:
            clean_by_item.setdefault(item, row)  # clean/neutral is sign-agnostic: one per item
        else:
            attacks.append(row)
    pairs = []
    dropped = {"no_clean": 0, "not_recognized": 0}
    for row in attacks:
        item = str(row["item_id"])
        # Step-2 gate: fail closed — a missing recognized_both flag is treated as NOT recognized.
        if not bool(row.get("recognized_both", False)):
            dropped["not_recognized"] += 1
            continue
        clean = clean_by_item.get(item)  # join clean by item_id (sign is an attack-side attribute)
        if clean is None:
            dropped["no_clean"] += 1
            continue
        pairs.append({
            "item_id": item, "sign": round(float(row.get("sign", 0.0)), 6),
            "style": str(row["style"]), "clean": clean, "attack": row,
        })
    if dropped["no_clean"] or dropped["not_recognized"]:
        print(f"dropped pairs: {dropped}")
    return sorted(pairs, key=lambda p: (p["item_id"], p["sign"], p["style"]))


def _split_items(items: list[str], seed: int) -> dict[str, str]:
    """Deterministic item-level train/dev/test assignment (60/20/20), frozen by seed."""
    rng = np.random.RandomState(seed)
    order = sorted(set(items))
    rng.shuffle(order)
    n = len(order)
    n_train, n_dev = int(0.6 * n), int(0.2 * n)
    split = {}
    for i, item in enumerate(order):
        split[item] = "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="clean/attack manifest JSONL")
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    ap.add_argument("--clean-style", default="neutral")
    ap.add_argument("--family", default=None,
                    help="restrict to one attack style so the channel axis U is single-family "
                         "(recommended for the confirmatory; the GO gate is evaluated per family)")
    ap.add_argument("--split-file", type=Path, default=None,
                    help="frozen {item_id: train|dev|test} JSON shared with the Step-2 tau "
                         "freeze (recognition_gate --dev-items). REQUIRED for a real gate so "
                         "the pre-registered split is honored; without it the split would be "
                         "re-randomized over the (outcome-dependent) recognized subset.")
    ap.add_argument("--allow-unfrozen-split", action="store_true",
                    help="permit deriving a fresh 60/20/20 split from --seed over the recognized "
                         "subset when --split-file is absent (SMOKE ONLY: leaks pre-registered "
                         "test items into the U fit; fails closed otherwise).")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--dose", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--candidate-ranks", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--min-reconstruction", type=float, default=0.6)
    ap.add_argument("--max-angle-rad", type=float, default=0.5)
    ap.add_argument("--k-orth", type=int, default=30,
                    help="legacy global-additive orthogonal-null ensemble size (rig sanity only)")
    ap.add_argument("--k-sham", type=int, default=20,
                    help="matched projected-transport sham ensemble size: random rank-matched "
                         "subspaces perp to U, patched with the SAME operator/support/dose as the "
                         "channel axis (the valid specificity null; Codex 2026-07-19)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="l18_patch.json")
    args = ap.parse_args()

    import torch

    from audio_safety.config import load_experiment_config
    from audio_safety.models.hooks import (
        ProjectedTransportIntervention,
        ResidualStreamIntervention,
        get_decoder_layers,
    )
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        load_qwen2_audio,
        model_input_device,
        prepare_qwen2_audio_inputs,
        resolve_audio_position_indices,
    )
    from audio_safety.pipelines.channel_axis import (
        largest_principal_angle,
        mean_anchored_basis,
        reconstruction_ratio,
        refusal_dim_direction,
        select_rank,
    )
    from audio_safety.pipelines.channel_patching import assert_pair_alignment, refusal_margin
    from audio_safety.pipelines.pitch_representation import _first_token_ids
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths)
    gate = cfg.pitch_representation
    instr = cfg.dataset.target_generation.instruction

    pairs = _load_pairs(args.pairs, args.clean_style)
    if args.family:
        pairs = [p for p in pairs if p["style"] == args.family]
    if not pairs:
        raise SystemExit("no recognized clean/attack pairs found in manifest")
    styles = sorted({p["style"] for p in pairs})
    if len(styles) > 1:
        print(f"WARNING: fitting ONE pooled channel axis U across families {styles}; "
              "pass --family <style> to fit a single-family U for the confirmatory.")
    if args.split_file is not None:
        # Honor the frozen split shared with Step 2's tau freeze so clean-dev (tau),
        # the U fit (train+dev), and the held-out test set are consistent across stages.
        split = json.loads(args.split_file.read_text())
        missing = sorted({p["item_id"] for p in pairs if p["item_id"] not in split})
        if missing:
            raise SystemExit(
                f"--split-file lacks {len(missing)} manifest item(s), e.g. {missing[:3]}"
            )
    elif args.allow_unfrozen_split:
        # SMOKE ONLY: _split_items depends on WHICH items are in the set, so deriving it
        # over the recognized subset re-randomizes fold membership and leaks pre-registered
        # test items into the U fit (research-code-reviewer, 2026-07-19).
        print("WARNING: --allow-unfrozen-split: split re-derived over the recognized subset; "
              "pre-registered test items may leak into the channel-axis U fit. SMOKE ONLY.")
        split = _split_items([p["item_id"] for p in pairs], args.seed)
    else:
        raise SystemExit(
            "refusing to re-randomize the train/dev/test split over the recognized subset "
            "(leaks test items into the U fit). Pass --split-file <run>/inputs/splits.json for "
            "a real gate, or --allow-unfrozen-split for a smoke test."
        )
    for p in pairs:
        p["split"] = split[p["item_id"]]

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    device = model_input_device(model)
    layer_mod = get_decoder_layers(model)[args.layer]
    refusal_ids = np.asarray(_first_token_ids(processor.tokenizer, gate.refusal_prefixes))
    compliance_ids = np.asarray(_first_token_ids(processor.tokenizer, gate.compliance_prefixes))
    rng = np.random.RandomState(args.seed)

    def prepare(wav: str):
        conv = build_audio_analysis_conversation(wav, instr, system_prompt=gate.system_prompt)
        inputs = prepare_qwen2_audio_inputs(processor, conv, device=device)
        t_ab = int(resolve_audio_position_indices(processor, conv)["first_generation_prelogit"])
        ids = inputs["input_ids"][0].detach().cpu().numpy()
        mask = (
            inputs["attention_mask"][0].detach().cpu().numpy()
            if "attention_mask" in inputs
            else np.ones_like(ids)
        )
        return conv, inputs, t_ab, ids, mask

    def capture(inputs, t_ab):
        store: dict[str, np.ndarray] = {}

        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store["h"] = hidden[0].detach().float().cpu().numpy()  # (T, d)

        handle = layer_mod.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        logits = out.logits[0, t_ab, :].detach().float().cpu().numpy()
        return store["h"], logits

    # ---- capture per-pair L18 states, with alignment guards ----
    audio_id = int(model.config.audio_token_id)
    for p in pairs:
        _, ci, ct, cids, cmask = prepare(str(args.run_dir / p["clean"]["path"]))
        _, ai, at, aids, amask = prepare(str(args.run_dir / p["attack"]["path"]))
        positions, t_ab = assert_pair_alignment(
            cids, cmask, aids, amask, audio_token_id=audio_id, clean_t_ab=ct, attack_t_ab=at
        )
        clean_h, clean_logits = capture(ci, t_ab)
        attack_h, attack_logits = capture(ai, t_ab)
        p.update(
            positions=positions, t_ab=t_ab,
            clean_audio=clean_h[positions], attack_audio=attack_h[positions],
            clean_read=clean_h[t_ab], attack_read=attack_h[t_ab],
            clean_inputs=ci, attack_inputs=ai,
            base_margin_attack=refusal_margin(attack_logits, refusal_ids, compliance_ids),
            base_margin_clean=refusal_margin(clean_logits, refusal_ids, compliance_ids),
        )

    # ---- fit channel subspace U per support on train/dev only ----
    def support_diffs(subset: list[dict], support: str) -> np.ndarray:
        if support == "A":
            return np.concatenate([p["clean_audio"] - p["attack_audio"] for p in subset], axis=0)
        return np.stack([p["clean_read"] - p["attack_read"] for p in subset])

    train = [p for p in pairs if p["split"] == "train"]
    dev = [p for p in pairs if p["split"] == "dev"]
    test = [p for p in pairs if p["split"] == "test"]
    if not (train and dev and test):
        raise SystemExit(f"need non-empty train/dev/test (got {len(train)}/{len(dev)}/{len(test)})")

    # select_rank holds dev out to choose the rank (train-fit basis, dev reconstruction +
    # train/dev subspace stability); the FINAL basis is then fit on train+dev, consistent
    # with the spec ("fit U on train+dev") and with the refusal-DiM / null pool below
    # (research-code-reviewer, 2026-07-19).
    bases: dict[str, np.ndarray] = {}
    rank_info: dict[str, int] = {}
    rank_ambiguous: dict[str, bool] = {}
    rank_diag: dict[str, list] = {}
    diag_rng = np.random.RandomState(args.seed)

    def _subsample(m: np.ndarray, cap: int = 2000) -> np.ndarray:
        # Arm A concatenates every audio token across pairs (tens of thousands of rows); a
        # full SVD per candidate rank stalls when two processes run in parallel (BLAS thread
        # thrash). The diagnostic only needs an APPROXIMATE spectrum, so subsample the rows.
        if m.shape[0] <= cap:
            return m
        return m[diag_rng.choice(m.shape[0], cap, replace=False)]

    for arm in args.arms:
        td, dd = support_diffs(train, arm), support_diffs(dev, arm)
        # Diagnostic (subsampled, approximate): dev reconstruction + train/dev subspace angle
        # per candidate rank, so a select_rank failure ("channel not stable") is legible.
        td_s, dd_s = _subsample(td), _subsample(dd)
        diag = []
        for rk in args.candidate_ranks:
            try:
                bt, bd = mean_anchored_basis(td_s, rk), mean_anchored_basis(dd_s, rk)
                rec, ang = reconstruction_ratio(dd_s, bt), largest_principal_angle(bt, bd)
                diag.append({"rank": rk, "dev_recon": round(rec, 4), "angle_rad": round(ang, 4)})
                print(f"  [rank-diag {arm} r={rk}] dev_recon={rec:.3f} angle={ang:.3f}rad "
                      f"(subsampled; need recon>={args.min_reconstruction} angle<={args.max_angle_rad})",
                      flush=True)
            except ValueError as e:
                diag.append({"rank": rk, "error": str(e)})
                print(f"  [rank-diag {arm} r={rk}] {e}", flush=True)
        rank_diag[arm] = diag
        # Real rank selection + basis fit on FULL data. If NO candidate rank is stable, this
        # support has no reliable low-dim channel subspace -> fall back to the registered
        # rank-1 difference-in-means (the nested baseline in the direction doc), FLAGGED as
        # ambiguous, so this arm's causal test still runs but is not read as a confirmatory
        # multi-rank channel. A stable arm proceeds normally; only ALL-arm failure is fatal.
        try:
            rank = int(select_rank(
                td, dd, args.candidate_ranks,
                min_reconstruction=args.min_reconstruction, max_angle_rad=args.max_angle_rad,
            ))
            ambiguous = False
        except ValueError as e:
            rank, ambiguous = 1, True
            print(f"  [rank {arm}] AMBIGUOUS: {e} -> rank-1 DiM baseline (FLAGGED)", flush=True)
        bases[arm] = mean_anchored_basis(support_diffs(train + dev, arm), rank)
        rank_info[arm] = rank
        rank_ambiguous[arm] = ambiguous
        print(f"  [rank {arm}] selected={rank} ambiguous={ambiguous}", flush=True)

    # ---- matched sham subspaces: random rank-matched bases perp to U, for a valid
    # specificity null. The sham is patched with the SAME projected-transport operator,
    # support, and dose as the channel axis U (Codex 2026-07-19: a global-additive
    # orthogonal-null vector is NOT a matched null for a pair-specific projected transport).
    # Each basis is a covariance-matched draw from the arm's displacement span, then
    # orthogonalized against U's rowspace so the sham is a genuinely different direction.
    def _sham_bases(arm: str) -> list[np.ndarray]:
        diffs = support_diffs(train + dev, arm)  # (n, d)
        u = bases[arm]
        r = rank_info[arm]
        n = diffs.shape[0]
        out: list[np.ndarray] = []
        for _ in range(args.k_sham):
            rows: list[np.ndarray] = []
            guard = 0
            while len(rows) < r and guard < 200 * r:
                guard += 1
                v = (rng.standard_normal(n) @ diffs).astype(np.float64)  # covariance-matched
                for b in list(u) + rows:
                    v = v - (v @ b) * b
                nv = float(np.linalg.norm(v))
                if nv > 1e-8:
                    rows.append(v / nv)
            if len(rows) == r:
                out.append(np.stack(rows))
        return out

    sham_bases = {arm: _sham_bases(arm) for arm in args.arms}
    max_dose = max(float(d) for d in args.dose)

    # ---- refusal-DiM global control (RECOMPUTED on this cohort) + orthogonal null pool ----
    def _behavior_label(value: str | None) -> int:
        # repo-canonical vocabulary is policy_refusal / harmful_compliance
        if value in ("policy_refusal", "refused"):
            return 1
        if value in ("harmful_compliance", "complied"):
            return 0
        return -1

    labels, reads = [], []
    for p in train + dev:
        labels.append(_behavior_label(p["attack"].get("behavior_label")))
        reads.append(p["attack_read"])
        labels.append(_behavior_label(p["clean"].get("behavior_label")))
        reads.append(p["clean_read"])
    has_both = any(v == 0 for v in labels) and any(v == 1 for v in labels)
    r_dim = refusal_dim_direction(np.stack(reads), np.asarray(labels)) if has_both else None
    null_pool = np.stack([p["clean_read"] - p["attack_read"] for p in train + dev])

    def margin_with_ctx(inputs, t_ab, ctx):
        # No applied_count assertion here: only ProjectedTransportIntervention increments it
        # (patch/transport). The additive ResidualStreamIntervention controls (add/all_positions)
        # legitimately leave it at 0, so the transport-only invariant is checked in transport_margin.
        with torch.inference_mode():
            if ctx is None:
                logits = model(**inputs, use_cache=False, return_dict=True).logits[0, t_ab, :]
            else:
                with ctx:
                    logits = model(**inputs, use_cache=False, return_dict=True).logits[0, t_ab, :]
        return refusal_margin(logits.detach().float().cpu().numpy(), refusal_ids, compliance_ids)

    def transport_margin(receiver_inputs, positions, donor, basis, scale, t_ab):
        ctx = ProjectedTransportIntervention(
            model, layer_idx=args.layer, positions=positions, donor=donor, basis=basis, scale=scale
        )
        margin = margin_with_ctx(receiver_inputs, t_ab, ctx)
        if ctx.applied_count != 1:  # prefill-only, one-shot invariant
            raise RuntimeError(f"transport applied {ctx.applied_count} times, expected 1")
        return margin

    # ---- test-pair patches: Arm x direction x dose, plus controls ----
    results = []
    for p in test:
        row = {"item_id": p["item_id"], "sign": p["sign"], "style": p["style"],
               "t_ab": p["t_ab"], "n_audio": len(p["positions"]),
               "base_margin_attack": p["base_margin_attack"], "base_margin_clean": p["base_margin_clean"],
               "arms": {}}
        for arm in args.arms:
            basis = bases[arm]
            positions = p["positions"] if arm == "A" else [p["t_ab"]]
            clean_donor = p["clean_audio"] if arm == "A" else p["clean_read"][None, :]
            attack_donor = p["attack_audio"] if arm == "A" else p["attack_read"][None, :]
            arm_out = {"restore": {}, "corrupt": {}}
            for scale in args.dose:
                # Restoration: forward on ATTACK, donate CLEAN subspace coordinate.
                arm_out["restore"][f"{scale}"] = transport_margin(
                    p["attack_inputs"], positions, clean_donor, basis, scale, p["t_ab"]
                )
                # Corruption: forward on CLEAN, donate ATTACK subspace coordinate.
                arm_out["corrupt"][f"{scale}"] = transport_margin(
                    p["clean_inputs"], positions, attack_donor, basis, scale, p["t_ab"]
                )
            # MAGNITUDE-matched sham null: same restore operator/support, but each sham dose
            # is rescaled so its L18 edit norm equals U's max-dose edit norm per pair, so
            # "restore beats sham" reflects the DIRECTION U, not a larger edit (research-code-
            # reviewer 2026-07-19: U captures most displacement energy, so ‖P_U δ‖ > ‖P_sham δ‖).
            # delta = donor - receiver-state at the patched positions (= what the transport scales).
            attack_states = p["attack_audio"] if arm == "A" else p["attack_read"][None, :]
            delta = clean_donor - attack_states
            u_proj = float(np.linalg.norm(delta @ basis.T))          # ‖P_U δ‖
            sham_margins, sham_norms = [], []
            for sb in sham_bases[arm]:
                s_proj = float(np.linalg.norm(delta @ sb.T))         # ‖P_sham δ‖
                sham_norms.append(s_proj)
                mm_scale = max_dose * (u_proj / s_proj) if s_proj > 1e-9 else 0.0
                sham_margins.append(
                    transport_margin(p["attack_inputs"], positions, clean_donor, sb, mm_scale, p["t_ab"])
                )
            arm_out["sham_restore_max_dose"] = sham_margins
            arm_out["u_proj_norm"] = u_proj
            arm_out["sham_proj_norms"] = sham_norms
            row["arms"][arm] = arm_out

        # refusal-DiM global positive control (added at all positions on the attack pass)
        if r_dim is not None:
            a = float(r_dim @ (p["clean_read"] - p["attack_read"]))
            ctx = ResidualStreamIntervention(
                model, layer_idx=args.layer, vector=torch.tensor(r_dim, dtype=torch.float32),
                mode="add", scale=a, all_positions=True,
            )
            row["rdim_global_margin"] = margin_with_ctx(p["attack_inputs"], p["t_ab"], ctx)
            # orthogonal-null ensemble (matched |a|, perp to r_dim)
            orth = []
            for pi in rng.choice(len(null_pool), size=min(args.k_orth, len(null_pool)), replace=False):
                v = null_pool[pi].astype(np.float64)
                v = v - (r_dim @ v) * r_dim
                nv = np.linalg.norm(v)
                if nv > 1e-6:
                    ctx = ResidualStreamIntervention(
                        model, layer_idx=args.layer, vector=torch.tensor(v / nv, dtype=torch.float32),
                        mode="add", scale=abs(a), all_positions=True,
                    )
                    orth.append(margin_with_ctx(p["attack_inputs"], p["t_ab"], ctx))
            row["orth_null_margins"] = orth
        results.append(row)
        print(f"  {p['item_id'][-6:]} {p['style']} arms={list(row['arms'])} base_att={p['base_margin_attack']:+.2f}")

    out = {
        "layer": args.layer,
        "seed": args.seed,  # the sham ensemble (the GO-gating null) is drawn from this seed
        "pairs_file": str(args.pairs),
        "split_file": str(args.split_file) if args.split_file else None,
        "clean_style": args.clean_style,
        "n_pairs": len(pairs),
        "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        "ranks": rank_info,
        "rank_ambiguous": rank_ambiguous,
        "rank_diagnostics": rank_diag,
        "dose": args.dose,
        "k_sham": args.k_sham,
        "n_sham_per_arm": {arm: len(sham_bases[arm]) for arm in args.arms},
        "refusal_dim_available": r_dim is not None,
        "results": results,
    }
    out_dir = args.run_dir / "channel_patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out_name).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_dir / args.out_name} (ranks={rank_info}, test={len(test)})")


if __name__ == "__main__":
    main()
