#!/usr/bin/env -S uv run python
"""Generate the Run 4 Stage A text arm and own-transcript control arm.

For each scored manifest row this produces two matched-content behaviors through
the *same* Qwen2-Audio backbone as the audio arm:

- text arm: send ``reference_text`` (the exact spoken content) as ChatML text,
- own-transcript arm: transcribe the wav with the model itself, then feed that
  transcript back as text; record the transcript and its WER vs ``reference_text``.

Both are resumable by row key and reuse the audio arm's system prompt / decoding
/ token budget so the only deliberate difference is the input modality
(design §7.1).

Example:
    ./scripts/generate_text_behavior.py \
        --config configs/experiments/run4_conversion_gap.yaml
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from audio_safety.config import load_experiment_config
from audio_safety.evaluation import label_output
from audio_safety.models.qwen2_audio import (
    generate_audio_response,
    generate_text_response,
    load_qwen2_audio,
)
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths
from audio_safety.utils.text import token_overlap, word_error_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="max source rows")
    parser.add_argument(
        "--arms",
        default="text,own_transcript",
        help="comma-separated arms to generate (text, own_transcript)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row.get("item_id")), str(row.get("safety_label")), str(row.get("style")))


def _resume(output_path: Path, overwrite: bool) -> tuple[list[dict], set[tuple[str, str, str]]]:
    if output_path.exists() and not overwrite:
        existing = load_jsonl(output_path)
        done = {row_key(r) for r in existing if str(r.get("output") or "").strip()}
        return existing, done
    return [], set()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.conversion_gap is None:
        raise SystemExit("this config has no `conversion_gap` block (Run 4 Stage A)")
    text_cfg = cfg.conversion_gap.text_arm
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir, cache_dir=args.cache_dir)

    source_path = paths.data_dir / cfg.dataset.asr.scored_manifest_file
    rows = [r for r in load_jsonl(source_path) if bool(r.get("transcript_control_passed"))]
    if args.limit is not None:
        rows = rows[: args.limit]

    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    unknown = arms - {"text", "own_transcript"}
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    text_path = paths.data_dir / text_cfg.text_outputs_file
    transcript_path = paths.data_dir / text_cfg.transcript_outputs_file
    text_out, text_done = _resume(text_path, args.overwrite)
    tr_out, tr_done = _resume(transcript_path, args.overwrite)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    for row in tqdm(rows, desc="Stage A text/transcript", unit="row"):
        key = row_key(row)
        reference_text = str(row.get("reference_text") or "")
        safety_label = str(row["safety_label"])

        if "text" in arms and key not in text_done:
            prompt = text_cfg.framing_template.format(reference_text=reference_text)
            response = generate_text_response(
                model, processor, prompt, max_new_tokens=text_cfg.max_new_tokens
            )
            updated = dict(row)
            updated["modality"] = "text"
            updated["judge_request_text"] = reference_text
            updated["output"] = response
            label, mode, needs_review = label_output(response, safety_label=safety_label)
            updated["behavior_label"] = label
            updated["refusal_mode"] = mode
            updated["needs_manual_review"] = needs_review
            text_out.append(updated)
            save_jsonl(text_out, text_path)

        if "own_transcript" in arms and key not in tr_done:
            audio_path = paths.data_dir / str(row["path"])
            transcript = generate_audio_response(
                model,
                processor,
                audio_path,
                text_cfg.transcribe_instruction,
                max_new_tokens=text_cfg.transcribe_max_new_tokens,
            )
            prompt = text_cfg.framing_template.format(reference_text=transcript)
            response = generate_text_response(
                model, processor, prompt, max_new_tokens=text_cfg.max_new_tokens
            )
            updated = dict(row)
            updated["modality"] = "own_transcript"
            # Judge stays blind: it always sees the canonical reference_text, so the
            # only difference across arms is the response, never the request.
            updated["judge_request_text"] = reference_text
            updated["output"] = response
            updated["model_transcript"] = transcript
            updated["transcript_wer"] = word_error_rate(reference_text, transcript)
            updated["transcript_token_overlap"] = token_overlap(reference_text, transcript)
            label, mode, needs_review = label_output(response, safety_label=safety_label)
            updated["behavior_label"] = label
            updated["refusal_mode"] = mode
            updated["needs_manual_review"] = needs_review
            tr_out.append(updated)
            save_jsonl(tr_out, transcript_path)

    if "text" in arms:
        print(f"[text-arm] wrote {len(text_out)} rows -> {text_path}")
    if "own_transcript" in arms:
        print(f"[transcript-arm] wrote {len(tr_out)} rows -> {transcript_path}")


if __name__ == "__main__":
    main()
