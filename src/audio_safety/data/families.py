"""Audio rendering manifests and transcript-control records for Audio-RDO."""

import json
import shlex
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import AsrConfig, AudioRdoDatasetConfig, TtsConfig
from audio_safety.data.datasets import AudioRdoPair
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.text import token_overlap, word_error_rate


@dataclass(frozen=True)
class RenderedAudio:
    path: Path
    content_id: str
    safety_label: str  # harmful | benign
    style: str
    voice_seed: int | None = None
    transcript: str | None = None
    wer: float | None = None
    duration_s: float | None = None
    style_passed: bool | None = None
    core_tokens_preserved: bool | None = None
    reference_text: str | None = None


def expected_audio_path(
    out_dir: Path,
    *,
    content_id: str,
    safety_label: str,
    style: str,
    suffix: str = ".wav",
) -> Path:
    """Deterministic path for one rendered clip."""
    return out_dir / safety_label / style / f"{content_id}{suffix}"


def render_style(
    content_text: str,
    content_id: str,
    safety_label: str,
    style: str,
    cfg: AudioRdoDatasetConfig,
    out_dir: Path,
    seed: int,
) -> list[RenderedAudio]:
    """Render one text with the configured TTS engine.

    The implementation is deliberately not hardcoded here because CosyVoice2 setup
    is deployment-specific. A concrete renderer must write the file at
    ``expected_audio_path`` and return this metadata contract.
    """
    target = expected_audio_path(
        out_dir,
        content_id=content_id,
        safety_label=safety_label,
        style=style,
    )
    raise NotImplementedError(
        f"{cfg.tts_engine} rendering pending; expected output path is {target}"
    )


def _format_command(template: str, values: dict[str, str]) -> list[str]:
    formatted = template.format(**values)
    return shlex.split(formatted)


def render_audio_records(
    pairs: Iterable[AudioRdoPair],
    dataset_cfg: AudioRdoDatasetConfig,
    tts_cfg: TtsConfig,
    data_dir: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Render harmful/benign pairs for every configured style.

    ``dry_run`` writes the same manifest records without invoking the TTS command.
    This is useful for checking paths on a new cloud instance.
    """
    audio_root = data_dir / tts_cfg.audio_subdir
    records: list[dict[str, object]] = []

    if not dry_run and not tts_cfg.command_template:
        raise ValueError("dataset.tts.command_template is required unless dry_run=True")

    for pair in pairs:
        for safety_label, text in (
            ("harmful", pair.harmful_text),
            ("benign", pair.benign_text),
        ):
            for style in dataset_cfg.styles:
                output = expected_audio_path(
                    audio_root,
                    content_id=pair.item_id,
                    safety_label=safety_label,
                    style=style,
                )
                values = {
                    "text": text,
                    "text_json": json.dumps(text, ensure_ascii=False),
                    "style": style,
                    "output": str(output),
                    "output_path": str(output),
                    "item_id": pair.item_id,
                    "query_id": pair.item_id,
                    "safety_label": safety_label,
                    "query_type": safety_label,
                }
                command = (
                    _format_command(tts_cfg.command_template, values)
                    if tts_cfg.command_template
                    else None
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and not tts_cfg.overwrite:
                    status = "exists"
                elif dry_run:
                    status = "planned"
                else:
                    if command is None:
                        raise ValueError("dataset.tts.command_template is required")
                    subprocess.run(command, check=True, timeout=None)
                    status = "rendered"

                records.append(
                    {
                        "item_id": pair.item_id,
                        "category": pair.category,
                        "safety_label": safety_label,
                        "style": style,
                        "path": str(output.relative_to(data_dir)),
                        "reference_text": text,
                        "transcript": None,
                        "wer": None,
                        "duration_s": None,
                        "style_passed": True,
                        "core_tokens_preserved": None,
                        "status": status,
                        "command": command if dry_run else None,
                    }
                )

    save_jsonl(records, data_dir / tts_cfg.manifest_file)
    return records


def score_transcript_records(
    records: Iterable[dict[str, object]],
    dataset_cfg: AudioRdoDatasetConfig,
    asr_cfg: AsrConfig,
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for record in records:
        reference = str(record.get("reference_text") or "")
        transcript = str(record.get("transcript") or "")
        wer = word_error_rate(reference, transcript) if transcript else None
        overlap = token_overlap(reference, transcript) if transcript else 0.0
        core_preserved = overlap >= asr_cfg.min_token_overlap
        style_passed = record.get("style_passed")
        if style_passed is None:
            style_passed = True
        passed = (
            wer is not None
            and wer <= dataset_cfg.transcript_control.wer_max
            and (core_preserved or not dataset_cfg.transcript_control.require_harmful_tokens)
            and (
                bool(style_passed)
                or not dataset_cfg.transcript_control.require_style_classifier_pass
            )
        )
        updated = dict(record)
        updated.update(
            {
                "wer": wer,
                "token_overlap": overlap,
                "core_tokens_preserved": core_preserved,
                "style_passed": bool(style_passed),
                "transcript_control_passed": passed,
            }
        )
        scored.append(updated)
    return scored


def skip_transcript_control_records(
    records: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Pass rendered records through when ASR validation is disabled for a run."""
    skipped: list[dict[str, object]] = []
    for record in records:
        updated = dict(record)
        updated.update(
            {
                "wer": None,
                "token_overlap": None,
                "core_tokens_preserved": None,
                "style_passed": bool(record.get("style_passed", True)),
                "transcript_control_passed": True,
                "transcript_control_skipped": True,
            }
        )
        skipped.append(updated)
    return skipped


def transcribe_records_with_command(
    records: Iterable[dict[str, object]],
    asr_cfg: AsrConfig,
    data_dir: Path,
) -> list[dict[str, object]]:
    """Fill transcript fields by running a simple ASR command per audio file.

    The command should print the transcript to stdout and may reference
    ``{audio}``, ``{path}``, ``{item_id}``, ``{style}``, and ``{safety_label}``.
    """
    if not asr_cfg.command_template:
        raise ValueError("dataset.asr.command_template is required when asr.mode='command'")

    updated_records = []
    for record in records:
        audio_path = data_dir / str(record["path"])
        command = _format_command(
            asr_cfg.command_template,
            {
                "audio": str(audio_path),
                "path": str(audio_path),
                "item_id": str(record["item_id"]),
                "style": str(record["style"]),
                "safety_label": str(record["safety_label"]),
            },
        )
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=None,
        )
        updated = dict(record)
        updated["transcript"] = result.stdout.strip()
        updated_records.append(updated)
    return updated_records


def score_transcript_manifest(
    data_dir: Path,
    dataset_cfg: AudioRdoDatasetConfig,
) -> list[dict[str, object]]:
    records = load_jsonl(data_dir / dataset_cfg.tts.manifest_file)
    if dataset_cfg.asr.mode == "skip":
        scored = skip_transcript_control_records(records)
        save_jsonl(scored, data_dir / dataset_cfg.asr.scored_manifest_file)
        return scored
    if dataset_cfg.asr.mode == "command":
        records = transcribe_records_with_command(records, dataset_cfg.asr, data_dir)
    scored = score_transcript_records(records, dataset_cfg, dataset_cfg.asr)
    save_jsonl(scored, data_dir / dataset_cfg.asr.scored_manifest_file)
    return scored


def transcript_control_passed(audio: RenderedAudio, cfg: AudioRdoDatasetConfig) -> bool:
    """Apply manifest-level transcript/style controls.

    Geometry analysis must only consume samples that pass this predicate. Missing
    WER/style metadata fails closed.
    """
    if cfg.asr.mode == "skip":
        return True
    if audio.wer is None or audio.wer > cfg.transcript_control.wer_max:
        return False
    if cfg.transcript_control.require_harmful_tokens and audio.core_tokens_preserved is False:
        return False
    return not (
        cfg.transcript_control.require_style_classifier_pass and audio.style_passed is not True
    )


def rendered_audio_from_record(record: dict[str, object], data_dir: Path) -> RenderedAudio:
    return RenderedAudio(
        path=data_dir / str(record["path"]),
        content_id=str(record["item_id"]),
        safety_label=str(record["safety_label"]),
        style=str(record["style"]),
        transcript=str(record["transcript"]) if record.get("transcript") else None,
        wer=float(record["wer"]) if record.get("wer") is not None else None,
        duration_s=float(record["duration_s"]) if record.get("duration_s") is not None else None,
        style_passed=bool(record["style_passed"])
        if record.get("style_passed") is not None
        else None,
        core_tokens_preserved=(
            bool(record["core_tokens_preserved"])
            if record.get("core_tokens_preserved") is not None
            else None
        ),
        reference_text=str(record["reference_text"]) if record.get("reference_text") else None,
    )
