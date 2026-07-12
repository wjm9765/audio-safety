#!/usr/bin/env -S uv run python
"""Compute the Run 4 Stage A / T0 behavioral-gap decision and report.

Reads the judged manifest and emits the PROCEED / STOP / AMBIGUOUS decision for
the neutral+harmful matched audio-vs-text attack gap (design §7.1), plus the
own-transcript reframe signal, benign over-refusal baseline, and judge agreement.
The gap is reported on all items AND on the untouched heldout split so the
heldout number can anchor a later confirmatory claim.

Example:
    ./scripts/analyze_t0.py \
        --config configs/experiments/run4_conversion_gap.yaml \
        --run-name run4_$(date +%Y%m%d_%H%M)_t0
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.evaluation.conversion_gap import compute_t0
from audio_safety.pipelines.rdo_gate import split_ids
from audio_safety.utils.io import load_jsonl, save_json, snapshot_config
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args()


def _t0_from(rows: list[dict], cfg) -> dict:
    cg = cfg.conversion_gap
    return compute_t0(
        rows,
        judge_models=cg.judge.models,
        primary_style=cg.t0.primary_style,
        min_audio_minus_text_attack_pp=cg.t0.min_audio_minus_text_attack_pp,
        p_threshold=cg.t0.p_threshold,
        require_ci_lower_above_zero=cg.t0.require_ci_lower_above_zero,
        require_both_judges=cg.t0.require_both_judges,
        n_bootstrap=cg.t0.n_bootstrap,
        ci_alpha=cg.t0.ci_alpha,
        faithful_wer_max=cg.text_arm.faithful_wer_max,
        seed=cfg.seed,
    )


def _render_markdown(report: dict, run_name: str) -> str:
    lines = [f"# T0 report — {run_name}", ""]
    for scope in ("all_items", "heldout_only"):
        block = report.get(scope)
        if not block:
            continue
        dec = block["decision"]
        lines += [f"## {scope}", "", f"**Decision: {dec['status']}** — {dec['note']}", ""]
        for reason in dec["reasons"]:
            lines.append(f"- {reason}")
        for model, did in block.get("specificity_did", {}).items():
            if not did.get("insufficient"):
                lines.append(
                    f"- specificity DiD [{model}] (harmful−benign, n={did['n']}): "
                    f"{did['did_pp']:.1f}pp (CI {did['ci_low_pp']:.1f}..{did['ci_high_pp']:.1f})"
                )
        agr = block.get("judge_agreement", {})
        if agr.get("kappa") is not None:
            lines.append(f"- judge agreement (kappa, n={agr['n']}): {agr['kappa']:.3f}")
        for model, summary in block.get("own_transcript", {}).items():
            if not summary.get("insufficient"):
                lines.append(
                    f"- own-transcript [{model}]: attack_rate={summary['attack_rate']:.3f}, "
                    f"faithful_frac={summary.get('faithful_fraction')}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.conversion_gap is None:
        raise SystemExit("this config has no `conversion_gap` block (Run 4 Stage A)")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)

    run_name = args.run_name or f"{cfg.name}_t0_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)

    judged = load_jsonl(paths.data_dir / cfg.conversion_gap.judge.outputs_file)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    heldout_ids = set(split_map["heldout"])
    heldout_rows = [r for r in judged if str(r.get("item_id")) in heldout_ids]

    report = {
        "run_name": run_name,
        "n_judged_rows": len(judged),
        "scope_note": (
            "all_items = every transcript-control-passed item (<=150), NOT a literal "
            "150. Reporting heldout_only consumes the heldout 40% for the direction "
            "call, so a Stage B/C confirmatory claim then requires a FRESH cohort "
            "(design §7.4)."
        ),
        "all_items": _t0_from(judged, cfg),
        "heldout_only": _t0_from(heldout_rows, cfg) if heldout_rows else None,
    }
    report_path = run_dir / cfg.conversion_gap.report_file
    save_json(report, report_path)
    md_path = run_dir / cfg.conversion_gap.report_markdown_file
    md_path.write_text(_render_markdown(report, run_name))

    status = report["all_items"]["decision"]["status"]
    print(f"[t0] all-items decision: {status}")
    print(f"[t0] report -> {report_path}")


if __name__ == "__main__":
    main()
