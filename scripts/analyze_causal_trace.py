#!/usr/bin/env -S uv run python
"""Run 4 causal-attribution: adjudicate the interchange-patching trace.

Reads the judged trace records + the generation manifest and emits the per-judge
causal summary: identity invariance, the benign-adjusted rescue contrast C, the
control ladder (same_item vs displacement/wrong_item), and intention-to-treat
reproduced-flip accounting. No forced GO/KILL threshold — the reader decides.

Example:
    ./scripts/analyze_causal_trace.py --config configs/experiments/run4_attack_flip.yaml \
        --run-name run4_causal_trace
"""

import argparse
import json
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.causal_trace import summarize
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-name", type=str, default="run4_causal_trace")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _md(summary: dict) -> str:
    lines = ["# Run 4 causal-attribution trace — adjudication", ""]
    inv = summary["identity_invariance"]
    lines.append(f"- **Identity invariance:** {inv['invariant']} "
                 f"({inv['n_mismatch']}/{inv['n_checked']} mismatches). "
                 "MUST be true; else the patch operator is wrong and the run is void.")
    lines.append(f"- **Primary cell:** layer {summary['primary_layer']} / "
                 f"{summary['primary_position']}; {summary['n_records']} traces.")
    lines.append("")
    for model, blk in summary["per_judge"].items():
        pc = blk["primary_contrast"]
        rep = blk["reproduced_flips"]
        lines += [
            f"## Judge: {model}",
            f"- reproduced flips (ITT): {rep['n_reproduced_flips']}/{rep['n_original_flips']} "
            f"(no_patch scored {rep['n_no_patch_scored']})",
            f"- **contrast C (harmful rescue − benign over-refusal):** {pc['contrast_C']}",
            f"  - harmful rescue = {pc['harmful_rescue']} (same_item {pc['same_item_refusal']} "
            f"vs no_patch {pc['no_patch_refusal']})",
            f"  - benign over-refusal = {pc['benign_overrefusal']}",
            f"- control ladder: wrong_item {pc['wrong_item_refusal']}, "
            f"random_displacement {pc['random_displacement_refusal']}, "
            f"identity {pc['identity_refusal']}, r_a_coord {pc['r_a_coord_refusal']}, "
            f"reverse {pc['reverse_refusal']}",
            "",
        ]
    lines.append("**Alive** iff identity invariant, C>0 (CI), same_item beats "
                 "displacement/wrong_item, reverse tends to compliance. Layer sweep exploratory.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    ct = cfg.causal_trace
    if ct is None or not ct.enabled:
        raise SystemExit("config has no enabled causal_trace block")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir)
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    judged = load_jsonl(run_dir / ct.judged_file)
    manifest = json.loads((run_dir / "causal_trace_manifest.json").read_text())
    judges = list(ct.judge_models or manifest.get("judges") or cfg.conversion_gap.judge.models)

    summary = summarize(
        judged,
        judge_models=judges,
        primary_layer=ct.primary_layer,
        primary_position=ct.primary_position,
        original_flip_items=[str(i) for i in manifest.get("flips", [])],
    )
    save_json(summary, run_dir / ct.report_file)
    (run_dir / ct.report_markdown_file).write_text(_md(summary))
    print(json.dumps({m: b["primary_contrast"] for m, b in summary["per_judge"].items()},
                     indent=2, default=str))
    print(f"[analyze-trace] report -> {run_dir / ct.report_file}")


if __name__ == "__main__":
    main()
