#!/usr/bin/env -S uv run python
"""Generate the undefended (no-steering) arm for SARSteer eval rows.

The undefended arm depends only on the frozen model and the eval audio — never on
the steering vector or ``alpha``. Splitting it out lets it run while the vector is
still being built/tuned, and lets an alpha re-sweep reuse it instead of paying for
the baseline again. ``apply_sarsteer_defense.py --undefended-cache`` consumes this
file; row keys are imported from that script so the two agree by construction.

Rows are flushed immediately and resume by the same stable
``(item_id, safety_label, condition/style, sign)`` key, so interrupting this run
(e.g. to hand the GPU to the defended pass) loses at most the in-flight row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.sarsteer import sarsteer_system_prompt
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_sarsteer_defense import (  # noqa: E402
    RowKey,
    prepare_eval_rows,
    record_id_for_key,
    row_key,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="output JSONL (default in run dir)")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="comma-separated condition allowlist (default: keep every manifest row)",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def load_completed_undefended(path: Path, *, valid_keys: set[RowKey]) -> set[RowKey]:
    """Keys already carrying a usable undefended generation."""

    if not path.exists():
        return set()
    done: set[RowKey] = set()
    for index, row in enumerate(load_jsonl(path)):
        if not isinstance(row, dict):
            raise SystemExit(f"existing baseline row {index} is not a JSON object")
        key = row_key(row, role="existing baseline", index=index)
        if key not in valid_keys:
            raise SystemExit(f"existing baseline row key absent from manifest: {key}")
        if key in done:
            raise SystemExit(f"duplicate existing baseline row key: {key}")
        if not isinstance(row.get("undefended_output"), str):
            raise SystemExit(f"existing baseline row {index} lacks string 'undefended_output'")
        done.add(key)
    return done


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled; enable it in the run9 config")
    sar = cfg.sarsteer
    paths = resolve_paths(
        cfg.paths, data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    manifest = (
        args.manifest if args.manifest.is_absolute() else paths.data_dir / args.manifest
    ).resolve()
    if not manifest.is_file():
        raise SystemExit(f"eval manifest not found: {manifest}")
    rows = load_jsonl(manifest)
    if args.conditions:
        keep = {c.strip() for c in args.conditions.split(",") if c.strip()}
        rows = [r for r in rows if r.get("condition") in keep]
        if not rows:
            raise SystemExit(f"no manifest row matched --conditions {sorted(keep)}")

    prepared = prepare_eval_rows(rows, data_dir=paths.data_dir)
    out_path = (
        args.out
        if args.out is not None and args.out.is_absolute()
        else (run_dir / (args.out or Path("sarsteer_undefended_baseline.jsonl")))
    ).resolve()
    valid_keys = {p.key for p in prepared}
    done = load_completed_undefended(out_path, valid_keys=valid_keys)
    pending = [p for p in prepared if p.key not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"[baseline] {len(prepared)} rows, {len(done)} cached, "
        f"{len(pending)} pending -> {out_path}",
        flush=True,
    )
    if not pending:
        return

    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    # Must match apply_sarsteer_defense.py exactly, or the paired contrast is confounded.
    system_prompt = sarsteer_system_prompt(sar.implementation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending, start=1):
            undefended = generate_audio_response(
                model,
                processor,
                item.audio_path,
                instruction,
                max_new_tokens=sar.max_new_tokens,
                do_sample=False,
                system_prompt=system_prompt,
            )
            record = dict(item.row)
            record.update(
                {
                    "record_id": item.row.get("record_id") or record_id_for_key(item.key),
                    "undefended_output": undefended,
                    "sarsteer_implementation": sar.implementation,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            print(f"[baseline] {index}/{len(pending)} key={item.key}", flush=True)


if __name__ == "__main__":
    main()
