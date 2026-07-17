#!/usr/bin/env -S uv run python
"""Sweep SARSteer ``alpha`` on the DEVELOPMENT controls only.

Loads the model once and generates one defended response per (alpha, row), reusing
a precomputed undefended baseline (the undefended arm does not depend on alpha).
Resumes by the stable ``(alpha, item_id, safety_label, condition/style, sign)`` key.

Selection discipline (pre-registered for this gate): alpha is chosen ONLY on
development controls — the non-target positive control plus the benign
soft-overrefusal/utility rows — and never on held-out channel-attack outcomes.
``--gate-roles`` enforces that at the input boundary rather than by convention, so
a channel row cannot silently enter the selection set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.sarsteer import (
    generate_audio_response_with_sarsteer,
    load_sarsteer_metadata,
    load_sarsteer_vectors,
    resolve_sarsteer_implementation,
    sarsteer_system_prompt,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_sarsteer_defense import prepare_eval_rows, record_id_for_key, row_key  # noqa: E402

SELECTION_ROLES = ("positive_control_eval", "soft_overrefusal", "utility_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--undefended-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=str,
        required=True,
        help="comma-separated alpha grid, e.g. 0.01,0.05,0.1,0.15,0.2,0.3",
    )
    parser.add_argument(
        "--gate-roles",
        type=str,
        default=",".join(SELECTION_ROLES),
        help="gate_role allowlist for the selection set (default: development controls only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled")
    sar = cfg.sarsteer
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        raise SystemExit("--alphas is empty")
    roles = {r.strip() for r in args.gate_roles.split(",") if r.strip()}

    paths = resolve_paths(
        cfg.paths, data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    manifest = (
        args.manifest if args.manifest.is_absolute() else paths.data_dir / args.manifest
    ).resolve()
    rows = [r for r in load_jsonl(manifest) if r.get("gate_role") in roles]
    if not rows:
        raise SystemExit(f"no manifest row matched --gate-roles {sorted(roles)}")
    prepared = prepare_eval_rows(rows, data_dir=paths.data_dir)

    vectors_path = args.vectors.resolve()
    vectors = load_sarsteer_vectors(vectors_path)
    meta = load_sarsteer_metadata(vectors_path)
    implementation = resolve_sarsteer_implementation(meta)
    if implementation != sar.implementation:
        raise SystemExit(
            f"vector/config implementation mismatch: {implementation} vs {sar.implementation}"
        )
    system_prompt = sarsteer_system_prompt(implementation)

    # Reuse apply's hardened loader: it rejects a cache generated on re-rendered
    # audio or under a different implementation, so a mispaired undefended arm
    # cannot slip in silently.
    from apply_sarsteer_defense import load_undefended_cache

    cache = load_undefended_cache(
        args.undefended_cache.resolve(),
        prepared=prepared,
        implementation=implementation,
        data_dir=paths.data_dir,
    )
    missing = [item.key for item in prepared if item.key not in cache]
    if missing:
        raise SystemExit(
            f"undefended cache is missing {len(missing)} selection rows, e.g. {missing[:3]}; "
            "regenerate the baseline before sweeping so every alpha shares one undefended arm"
        )

    out_path = (args.out if args.out.is_absolute() else run_dir / args.out).resolve()
    done: set[tuple] = set()
    if out_path.exists():
        for index, row in enumerate(load_jsonl(out_path)):
            done.add((float(row["sweep_alpha"]), *row_key(row, role="existing", index=index)))

    jobs = [
        (alpha, item) for alpha in alphas for item in prepared if (alpha, *item.key) not in done
    ]
    print(
        f"[sweep] {len(prepared)} selection rows x {len(alphas)} alphas = "
        f"{len(prepared) * len(alphas)} cells; {len(done)} cached; {len(jobs)} to run\n"
        f"[sweep] roles={sorted(roles)} -> {out_path}",
        flush=True,
    )
    if not jobs:
        return

    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for index, (alpha, item) in enumerate(jobs, start=1):
            defended = generate_audio_response_with_sarsteer(
                model,
                processor,
                item.audio_path,
                instruction,
                vectors=vectors,
                alpha=alpha,
                max_new_tokens=sar.max_new_tokens,
                do_sample=False,
                system_prompt=system_prompt,
                implementation=implementation,
            )
            record = dict(item.row)
            record.update(
                {
                    "record_id": item.row.get("record_id") or record_id_for_key(item.key),
                    "sweep_alpha": alpha,
                    "undefended_output": cache[item.key],
                    "defended_output": defended,
                    "sarsteer_implementation": implementation,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            print(f"[sweep] {index}/{len(jobs)} alpha={alpha} key={item.key}", flush=True)


if __name__ == "__main__":
    main()
