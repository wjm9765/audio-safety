#!/usr/bin/env -S uv run python
"""Compute the Run 4 §8 attack-induced-flip report from the judged manifest.

Reads the judged behavior manifest (the same one Stage A judges write) and, per
attack family and judge, reports the within-modality flip (genuine refusal ->
comply), the benign difference-in-differences specificity, and the audio-vs-text
interaction (design §8.3-8.4). This is descriptive direction-finding, not a §0
gate, so it prints numbers and writes a report rather than a PROCEED/STOP verdict.

Example:
    ./scripts/analyze_flip.py \
        --config configs/experiments/run4_attack_flip.yaml \
        --run-name run4_$(date +%Y%m%d_%H%M)_flip
"""

import argparse
from datetime import datetime
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.attack_flip import compute_attack_flip
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


def _judge_models(cfg) -> list[str]:
    af = cfg.attack_flip
    if af.judge_models:
        return list(af.judge_models)
    if cfg.conversion_gap is not None:
        return list(cfg.conversion_gap.judge.models)
    raise SystemExit(
        "no judge models: set attack_flip.judge_models or a conversion_gap.judge block"
    )


def _judged_path(cfg, paths) -> Path:
    if cfg.conversion_gap is not None:
        return paths.data_dir / cfg.conversion_gap.judge.outputs_file
    raise SystemExit("no judged manifest: a conversion_gap.judge block is required")


def _render_markdown(report: dict, run_name: str) -> str:
    lines = [f"# Attack-flip report — {run_name}", "", report.get("note", ""), ""]
    for family in report.get("families", []):
        lines += [f"## family: {family['name']}", ""]
        for style_block in family.get("styles", []):
            style = style_block["attack_style"]
            lines += [f"### attack_style: {style}", ""]
            for model, blocks in style_block.get("per_judge", {}).items():
                flip = blocks["flip"]
                if flip.get("insufficient"):
                    lines.append(f"- [{model}] flip: insufficient pairs")
                    continue
                p1 = flip["mcnemar_attacked_gt_clean"]["p_one_sided_audio_gt_text"]
                flip_ci = f"{flip['ci_low_pp']:.1f}..{flip['ci_high_pp']:.1f}"
                rate = flip["flip_rate_given_clean_refuse"]
                rate_str = f"{rate:.3f}" if rate is not None else "n/a"
                flips = f"{flip['genuine_flips']}/{flip['genuine_refuse_clean']}"
                lines.append(
                    f"- [{model}] flip: attacked={flip['attacked_attack_rate']:.3f} "
                    f"clean={flip['clean_attack_rate']:.3f} "
                    f"RD={flip['rd_pp']:.1f}pp (CI {flip_ci}), p1={p1:.4f}; "
                    f"genuine_flips={flips} (rate={rate_str}), "
                    f"reverse={flip['reverse_unflips']}"
                )
                lines.append(f"    - attacked taxonomy: {flip['attacked_taxonomy']}")
                did = blocks["benign_did"]
                if not did.get("insufficient"):
                    did_ci = f"{did['ci_low_pp']:.1f}..{did['ci_high_pp']:.1f}"
                    lines.append(
                        f"    - benign DiD (harmful-specificity, n={did['n']}): "
                        f"{did['did_pp']:.1f}pp (CI {did_ci})"
                    )
                spec = blocks["audio_specificity"]
                if not spec.get("insufficient"):
                    spec_ci = f"{spec['ci_low_pp']:.1f}..{spec['ci_high_pp']:.1f}"
                    am, tm = spec["audio_modality"], spec["text_modality"]
                    legs = (
                        f"{am} {spec['audio_attack_effect_pp']:.1f} "
                        f"vs {tm} {spec['text_attack_effect_pp']:.1f}"
                    )
                    lines.append(
                        f"    - {am}-vs-{tm} interaction (n={spec['n']}): "
                        f"{spec['interaction_pp']:.1f}pp (CI {spec_ci}) [{legs}]"
                    )
            agr = style_block.get("judge_agreement", {})
            if agr.get("kappa") is not None:
                lines.append(f"- judge agreement (kappa, n={agr['n']}): {agr['kappa']:.3f}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.attack_flip is None or not cfg.attack_flip.enabled:
        raise SystemExit("this config has no enabled `attack_flip` block (Run 4 §8)")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)

    run_name = args.run_name or f"{cfg.name}_flip_{datetime.now():%Y%m%d_%H%M}"
    run_dir = run_output_dir(paths.output_dir, run_name)
    snapshot_config(cfg, run_dir)

    judged = load_jsonl(_judged_path(cfg, paths))
    af = cfg.attack_flip
    report = {
        "run_name": run_name,
        "n_judged_rows": len(judged),
        **compute_attack_flip(
            judged,
            judge_models=_judge_models(cfg),
            families=[
                {"name": fam.name, "attack_styles": fam.attack_styles} for fam in af.families
            ],
            clean_style=af.clean_style,
            primary_modality=af.primary_modality,
            text_modality=af.text_modality,
            n_bootstrap=af.n_bootstrap,
            ci_alpha=af.ci_alpha,
            seed=cfg.seed,
        ),
    }
    report_path = run_dir / af.report_file
    save_json(report, report_path)
    (run_dir / af.report_markdown_file).write_text(_render_markdown(report, run_name))

    print(f"[flip] families: {[f['name'] for f in report['families']]}")
    print(f"[flip] report -> {report_path}")


if __name__ == "__main__":
    main()
