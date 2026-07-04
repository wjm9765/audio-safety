#!/usr/bin/env -S uv run python
"""Score transcript-control fields for rendered Audio-RDO audio.

If dataset.asr.mode is "manifest", fill transcript fields in
manifests/audio_rdo_renders.jsonl first. If mode is "command", the configured
ASR command is run and its stdout is used as the transcript.
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import score_transcript_manifest
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)
    scored = score_transcript_manifest(paths.data_dir, cfg.dataset)
    output_path = paths.data_dir / cfg.dataset.asr.scored_manifest_file
    passed = sum(bool(row.get("transcript_control_passed")) for row in scored)
    print(f"[asr] wrote {len(scored)} scored records -> {output_path}")
    print(f"[asr] transcript-control passed: {passed}/{len(scored)}")


if __name__ == "__main__":
    main()
