#!/usr/bin/env -S uv run python
"""Single-file Qwen2-Audio inference smoke script.

This follows the official Qwen2-Audio model-card flow:
AutoProcessor.apply_chat_template -> processor(text, audio=...) ->
Qwen2AudioForConditionalGeneration.generate.

Example:
    ./scripts/qwen2_audio_infer.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --audio /workspace/audio_safety_data/data/audio/demo.wav \
        --instruction "Please answer the question in the audio."
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--audio", type=Path, required=True, help="local audio file")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Please answer the question in the audio.",
        help="text instruction paired with the audio",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=None, help="override model cache root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, cache_dir=args.cache_dir)
    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    response = generate_audio_response(
        model,
        processor,
        args.audio,
        args.instruction,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()
