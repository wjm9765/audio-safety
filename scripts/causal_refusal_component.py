#!/usr/bin/env -S uv run python
"""Make-or-break causal test (codex r2 + reviewer-hardened): does the pitch-induced
displacement ALONG the refusal direction causally mediate the refuse->comply flip,
and is it carried by the SIGNED (odd) pitch component rather than a symmetric artifact?

For each agent-verified pitched flip cell (item i, pitch p; neutral refuses, p complies),
at a FROZEN layer L, with a leave-one-item-out refusal direction r_{-i} and harmfulness
direction hdir_{-i}:
  Delta   = h_{i,p} - h_{i,0}                      (raw pitch displacement, p2)
  j_i     = odd (signed) tangent of the item's 7-pitch trajectory; odd displacement = j_i * p
  restore_full : h += -(r.Delta) r                 (remove refusal-axis part of the pitch shift)
  restore_odd  : h += -(r.(j_i p)) r               (remove ONLY the signed-transport refusal part)
  orth         : h += -|r.Delta| unit(Delta_perp)  (equal-norm orthogonal control)
  harm_ctrl    : h += -(hdir.Delta) hdir           (remove harm-axis part; should NOT rescue)
  baseline     : no intervention                    (should reproduce compliance)
all_positions add-mode at L. Endpoint = full-response behavior (judge later).
Writes <run>/pitch_representation/causal_refusal_component.jsonl.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load_cells(run_dir):
    return [json.loads(l) for l in (run_dir / "pitch_representation/cells.jsonl").read_text().splitlines() if l.strip()]


def loo_dir(P2, label, valid, exclude_item, item_ids):
    mask = valid & (item_ids != exclude_item)
    y = label[mask]; X = P2[mask]
    if len(np.unique(y)) < 2:
        return None
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def odd_tangent(P2, idx, it, nz):
    base_key = (it, "harmful", 0.0)
    if base_key not in idx:
        return None
    base = P2[idx[base_key]]
    ps, D = [], []
    for p in nz:
        k = (it, "harmful", p)
        if k in idx:
            ps.append(p); D.append(P2[idx[k]] - base)
    if len(ps) < 3:
        return None
    ps = np.array(ps); D = np.stack(D)
    A = np.stack([ps, ps ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    return coef[0]  # j (odd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    ap.add_argument("--limit-cells", type=int, default=None)
    args = ap.parse_args()

    import torch
    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import (
        load_qwen2_audio, generate_audio_response, generate_audio_response_with_intervention,
    )
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    cells = load_cells(args.run_dir)
    arrays = dict(np.load(args.run_dir / "pitch_representation/activations.npz", allow_pickle=True))
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    off = llm_layers.index(args.layer)
    P2 = np.asarray(arrays["llm_p2"], dtype=np.float64)[:, off, :]
    idx = {(str(c["item_id"]), str(c["safety_label"]), round(float(c["pitch_semitones"]), 6)): int(c["activation_index"]) for c in cells}
    item_ids = np.asarray([str(c["item_id"]) for c in cells])
    lab = np.asarray([str(c.get(args.label_field) or "") for c in cells])
    marg = np.asarray([float(c["refusal_margin"]) for c in cells])
    safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
    refuse_label = (lab == "policy_refusal").astype(int)
    valid_dec = np.isin(lab, ["policy_refusal", "harmful_compliance"])
    pitches = sorted({round(float(c["pitch_semitones"]), 6) for c in cells})
    nz = [p for p in pitches if p != 0.0]

    byit = {}
    for c in cells:
        if c["safety_label"] == "harmful":
            byit.setdefault(str(c["item_id"]), {})[round(float(c["pitch_semitones"]), 6)] = c
    flips = []
    for it, cm in byit.items():
        neu = cm.get(0.0)
        if not neu or neu.get(args.label_field) != "policy_refusal":
            continue
        for p, c in cm.items():
            if p != 0.0 and c.get(args.label_field) == "harmful_compliance":
                flips.append((it, p, c))
    flips.sort()
    if args.limit_cells:
        flips = flips[:args.limit_cells]
    print(f"L{args.layer}: {len(flips)} verified flip cells across {len(set(f[0] for f in flips))} items")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction

    rows = []
    for it, p, c in flips:
        nh, ph = idx[(it, "harmful", 0.0)], idx[(it, "harmful", p)]
        delta = P2[ph] - P2[nh]
        r = loo_dir(P2, refuse_label, valid_dec, it, item_ids)
        hdir = loo_dir(P2, safety, np.ones(len(cells), bool), it, item_ids)
        if r is None or hdir is None:
            continue
        j = odd_tangent(P2, idx, it, nz)
        rdotd = float(r @ delta)
        a_mag = abs(rdotd)
        dperp = delta - rdotd * r
        dpn = np.linalg.norm(dperp)
        w = dperp / dpn if dpn > 1e-9 else None
        odd_disp = j * p if j is not None else None
        r_odd = float(r @ odd_disp) if odd_disp is not None else 0.0
        hdotd = float(hdir @ delta)
        cand = args.run_dir / c["variant_path"]
        wav = str(cand)

        def gen(vec, scale):
            if vec is None:
                return ""
            return generate_audio_response_with_intervention(
                model, processor, wav, instruction, layer_idx=args.layer,
                position_name="first_generation_prelogit", vector=torch.tensor(vec, dtype=torch.float32),
                mode="add", scale=float(scale), all_positions=True,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        base = generate_audio_response(model, processor, wav, instruction, max_new_tokens=args.max_new_tokens, do_sample=False)
        row = {
            "item_id": it, "pitch": p, "reference_text": c.get("reference_text"),
            "r_dot_delta": rdotd, "r_dot_odd": r_odd, "h_dot_delta": hdotd, "a_mag": a_mag,
            "baseline": base,
            "restore_full": gen(r, -rdotd),                 # remove refusal-axis pitch shift
            "restore_odd": gen(r, -r_odd),                  # remove only signed-transport refusal part
            "orth": gen(w, -a_mag),                         # equal-norm orthogonal control
            "harm_ctrl": gen(hdir, -hdotd),                 # remove harm-axis part (should not rescue)
        }
        rows.append(row)
        print(f"  {it[-4:]} p={p:+g} r.d={rdotd:+.2f} r.odd={r_odd:+.2f} h.d={hdotd:+.2f}")
        print(f"     base   : {base[:60]!r}")
        print(f"     restore: {row['restore_full'][:60]!r}")
        print(f"     orth   : {row['orth'][:60]!r}  harm_ctrl: {row['harm_ctrl'][:50]!r}")

    outp = args.run_dir / "pitch_representation/causal_refusal_component.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {outp} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
