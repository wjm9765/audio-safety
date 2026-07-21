#!/usr/bin/env -S uv run python
"""Run 13 analysis (CPU): label the sweep generations, build the refusal-rate / margin / transition
tables per rank, and write the report. NO verdict / decision.json — the human reads the tables.

Primary = full-generation policy-refusal RATE. Margin is secondary. 4-way auto-labels are a cheap
decoding-confound side-channel and are NON-AUTHORITATIVE for harmful_compliance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_safety.evaluation.labeling import label_output

LABELS = ["policy_refusal", "harmful_compliance", "benign_answer", "decoding_failure"]


def _label(text: str, safety: str) -> str:
    return label_output(str(text or ""), safety_label=safety)[0]


def _boot_ci(per_item: list[float], n_boot: int, rng: np.random.RandomState):
    """Item-clustered bootstrap 95% CI of the mean of per-item values."""
    a = np.asarray([x for x in per_item if x is not None], dtype=float)
    if a.size == 0:
        return None, None, None
    idx = rng.randint(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return round(float(a.mean()), 4), round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def _per_item(rows_by_item: dict, fn) -> list[float]:
    """Average fn over the sign rows of each item -> one value per item."""
    out = []
    for _it, rows in rows_by_item.items():
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pb = args.run_dir / "phaseB"
    gens = []
    for gf in sorted(pb.glob("gens_*.jsonl")):
        gens += [json.loads(x) for x in gf.read_text().splitlines() if x.strip()]
    margins = []
    for mf in sorted(pb.glob("margins_*.jsonl")):
        margins += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    ranks = sorted({int(k) for g in gens for k in g.get("ranks", {})})
    rng = np.random.RandomState(args.seed)

    gbi: dict[str, list] = {}
    for g in gens:
        gbi.setdefault(g["item_id"], []).append(g)
    mbi: dict[str, list] = {}
    for m in margins:
        mbi.setdefault(m["item_id"], []).append(m)

    def refusal(text):
        return 1.0 if _label(text, "harmful") == "policy_refusal" else 0.0

    def brefusal(text):
        return 1.0 if _label(text, "benign") == "policy_refusal" else 0.0

    report = {"ranks": ranks, "n_items": len(gbi), "identity": {}, "per_rank": {}}
    # rank-independent bases
    base = _per_item(gbi, lambda g: refusal(g["aH_identity"]))
    full = _per_item(gbi, lambda g: refusal(g["aH_fullstate"]))
    cbase = _per_item(gbi, lambda g: refusal(g["cH_identity"]))
    bbase = _per_item(gbi, lambda g: brefusal(g["aB_identity"]))
    report["identity"] = {
        "attack_H_refusal": _boot_ci(base, args.n_bootstrap, rng),
        "fullstate_refusal": _boot_ci(full, args.n_bootstrap, rng),
        "clean_H_refusal": _boot_ci(cbase, args.n_bootstrap, rng),
        "benign_attack_refusal": _boot_ci(bbase, args.n_bootstrap, rng),
    }

    transitions = {}
    for k in ranks:
        sk = str(k)

        def gr(g, key):
            return g["ranks"][sk][key] if sk in g.get("ranks", {}) else None

        # refusal rates (per-item averaged over signs), and paired diffs vs their identity baseline
        d_restore = _per_item(gbi, lambda g: refusal(gr(g, "aH_restore")) - refusal(g["aH_identity"]))
        d_sham = _per_item(gbi, lambda g: refusal(gr(g, "aH_sham0")) - refusal(g["aH_identity"]))
        d_generic = _per_item(gbi, lambda g: refusal(gr(g, "aH_generic")) - refusal(g["aH_identity"]))
        d_corrupt = _per_item(gbi, lambda g: refusal(gr(g, "cH_corrupt")) - refusal(g["cH_identity"]))
        d_borr = _per_item(gbi, lambda g: brefusal(gr(g, "aB_brestore")) - brefusal(g["aB_identity"]))
        rs = _per_item(gbi, lambda g: refusal(gr(g, "aH_restore")) - refusal(gr(g, "aH_sham0")))
        rg = _per_item(gbi, lambda g: refusal(gr(g, "aH_restore")) - refusal(gr(g, "aH_generic")))

        # margin: restore@1 - identity margin; corrupt@1 sign; restore@1 - mean(sham)
        def m_restore(m):
            r = m["ranks"].get(sk)
            return (r["restore"]["1.0"] - m["M_attack_H"]) if r else None

        def m_corrupt(m):
            r = m["ranks"].get(sk)
            return (r["corrupt"]["1.0"] - m["M_clean_H"]) if r else None

        def m_rs(m):
            r = m["ranks"].get(sk)
            return (r["restore"]["1.0"] - float(np.mean(r["sham"]))) if r else None

        report["per_rank"][sk] = {
            "d_refusal_restore": _boot_ci(d_restore, args.n_bootstrap, rng),
            "d_refusal_sham": _boot_ci(d_sham, args.n_bootstrap, rng),
            "d_refusal_generic": _boot_ci(d_generic, args.n_bootstrap, rng),
            "restore_minus_sham": _boot_ci(rs, args.n_bootstrap, rng),
            "restore_minus_generic": _boot_ci(rg, args.n_bootstrap, rng),
            "d_refusal_corrupt_on_clean": _boot_ci(d_corrupt, args.n_bootstrap, rng),
            "benign_over_refusal_delta": _boot_ci(d_borr, args.n_bootstrap, rng),
            "dM_restore": _boot_ci(_per_item(mbi, m_restore), args.n_bootstrap, rng),
            "dM_corrupt": _boot_ci(_per_item(mbi, m_corrupt), args.n_bootstrap, rng),
            "dM_restore_minus_sham": _boot_ci(_per_item(mbi, m_rs), args.n_bootstrap, rng),
        }

        # transition matrix identity -> restore (4x4), pooled over rows
        T = {a: {b: 0 for b in LABELS} for a in LABELS}
        for g in gens:
            if sk not in g.get("ranks", {}):
                continue
            a = _label(g["aH_identity"], "harmful")
            b = _label(g["ranks"][sk]["aH_restore"], "harmful")
            T[a][b] += 1
        transitions[sk] = T

    # geometry passthrough
    geom_path = args.run_dir / "subspaces" / "geometry.json"
    geometry = json.loads(geom_path.read_text()) if geom_path.exists() else {}

    an = args.run_dir / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    (an / "screen_report.json").write_text(json.dumps(
        {"report": report, "geometry": geometry}, indent=2) + "\n")
    (an / "transition_tables.json").write_text(json.dumps(transitions, indent=2) + "\n")

    # markdown
    lines = ["# Run 13 readout rank-sweep — screen report (no verdict)\n",
             f"Items: {report['n_items']}; ranks: {ranks}\n",
             "## Identity baselines (refusal rate [95% CI])",
             f"- attack_H: {report['identity']['attack_H_refusal']}",
             f"- fullstate ceiling: {report['identity']['fullstate_refusal']}",
             f"- clean_H: {report['identity']['clean_H_refusal']}",
             f"- benign(attack): {report['identity']['benign_attack_refusal']}\n",
             "## Per-rank refusal-rate deltas (mean [95% CI]); primary = restore vs sham/generic",
             "| k | Δrefusal restore | Δrefusal sham | restore−sham | restore−generic | corrupt(clean) | benign ORR Δ | ΔM restore | ΔM restore−sham |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in ranks:
        r = report["per_rank"][str(k)]
        def c(x):
            return f"{x[0]} [{x[1]},{x[2]}]" if x and x[0] is not None else "—"
        lines.append(f"| {k} | {c(r['d_refusal_restore'])} | {c(r['d_refusal_sham'])} | "
                     f"{c(r['restore_minus_sham'])} | {c(r['restore_minus_generic'])} | "
                     f"{c(r['d_refusal_corrupt_on_clean'])} | {c(r['benign_over_refusal_delta'])} | "
                     f"{c(r['dM_restore'])} | {c(r['dM_restore_minus_sham'])} |")
    lines += ["\n## Transition (identity→restore) — see transition_tables.json",
              "## Geometry (held-out recon / cross-fold angle / perm-p) — see subspaces/geometry.json\n",
              "_Note: 4-way labels are heuristic and non-authoritative for harmful_compliance; "
              "primary endpoint is policy-refusal rate. No GO/NO-GO applied._"]
    (an / "screen_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {an}/screen_report.md (+ .json, transition_tables.json)")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
