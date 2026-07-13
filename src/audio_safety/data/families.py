"""Audio rendering manifests and transcript-control records for Audio-RDO."""

import hashlib
import json
import os
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


def _batch_shard_path(batch_jobs_path: Path, worker_index: int) -> Path:
    suffix = batch_jobs_path.suffix or ".jsonl"
    stem = (
        batch_jobs_path.name[: -len(suffix)]
        if batch_jobs_path.suffix
        else batch_jobs_path.name
    )
    return batch_jobs_path.with_name(f"{stem}.worker{worker_index:02d}{suffix}")


def _split_jobs_round_robin(
    jobs: list[dict[str, object]],
    workers: int,
) -> list[list[dict[str, object]]]:
    shards: list[list[dict[str, object]]] = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        shards[index % workers].append(job)
    return [shard for shard in shards if shard]


def _batch_worker_env(
    tts_cfg: TtsConfig,
    worker_index: int,
    num_workers: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(tts_cfg.batch_worker_env)
    # The isolated TTS wrapper uses these values to draw one shard-level progress
    # bar per long-lived worker. Keep this out of the command template so any
    # compatible batch adapter can ignore the metadata without special CLI args.
    env["AUDIO_SAFETY_TTS_WORKER_INDEX"] = str(worker_index)
    env["AUDIO_SAFETY_TTS_NUM_WORKERS"] = str(num_workers)
    if tts_cfg.batch_worker_cuda_devices:
        device = tts_cfg.batch_worker_cuda_devices[
            worker_index % len(tts_cfg.batch_worker_cuda_devices)
        ]
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def _run_batch_shards(
    tts_cfg: TtsConfig,
    batch_jobs_path: Path,
    pending_jobs: list[dict[str, object]],
) -> None:
    if not tts_cfg.batch_command_template:
        raise ValueError("dataset.tts.batch_command_template is required")

    workers = min(tts_cfg.batch_workers, len(pending_jobs))
    if workers <= 1:
        batch_values = {
            "batch_jsonl": str(batch_jobs_path),
            "batch_jobs_file": str(batch_jobs_path),
            "worker_index": "0",
            "worker_id": "0",
            "num_workers": "1",
            "cuda_device": (
                str(tts_cfg.batch_worker_cuda_devices[0])
                if tts_cfg.batch_worker_cuda_devices
                else ""
            ),
        }
        batch_command = _format_command(tts_cfg.batch_command_template, batch_values)
        subprocess.run(
            batch_command,
            check=True,
            timeout=None,
            env=_batch_worker_env(tts_cfg, 0, 1),
        )
        return

    shards = _split_jobs_round_robin(pending_jobs, workers)
    processes: list[tuple[int, list[str], subprocess.Popen[bytes]]] = []
    for worker_index, shard_jobs in enumerate(shards):
        shard_path = _batch_shard_path(batch_jobs_path, worker_index)
        save_jsonl(shard_jobs, shard_path)
        cuda_device = ""
        if tts_cfg.batch_worker_cuda_devices:
            device_index = worker_index % len(tts_cfg.batch_worker_cuda_devices)
            cuda_device = str(tts_cfg.batch_worker_cuda_devices[device_index])
        batch_values = {
            "batch_jsonl": str(shard_path),
            "batch_jobs_file": str(shard_path),
            "worker_index": str(worker_index),
            "worker_id": str(worker_index),
            "num_workers": str(len(shards)),
            "cuda_device": cuda_device,
        }
        command = _format_command(tts_cfg.batch_command_template, batch_values)
        print(
            f"[render] starting TTS worker {worker_index + 1}/{len(shards)} "
            f"jobs={len(shard_jobs)} cuda={cuda_device or 'inherit'}",
            flush=True,
        )
        processes.append(
            (
                worker_index,
                command,
                subprocess.Popen(
                    command,
                    env=_batch_worker_env(tts_cfg, worker_index, len(shards)),
                ),
            )
        )

    failures: list[tuple[int, list[str], int]] = []
    for worker_index, command, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((worker_index, command, returncode))
    if failures:
        worker_index, command, returncode = failures[0]
        raise subprocess.CalledProcessError(
            returncode,
            command,
            stderr=f"TTS worker {worker_index} failed; {len(failures)} workers failed",
        )


def _record_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


# CosyVoice2 neutral-acoustic instruction (mirrors scripts/cosyvoice2_tts.py). The
# renderer forces this for neutral_acoustic_styles so jailbreak wrappers are spoken
# in a plain voice — the attack is in the wording, not the prosody.
NEUTRAL_STYLE_INSTRUCTION = "Speak in a neutral, clear voice.<|endofprompt|>"


def _parse_variant_override_file(
    path: Path,
    *,
    verify_hash: bool = False,
    strict_unique: bool = False,
) -> dict[tuple[str, str, str], str]:
    """Parse one (item,label,style)->styled_text override file.

    Shared by the style-variant (emotion) and attack-variant (jailbreak) files;
    both use the same record schema. Rows that self-report added operational
    detail, refusal/warning content, or low content preservation are skipped, as
    are empty rewrites.

    For the frozen attack file, ``verify_hash`` enforces that ``styled_text`` still
    hashes to the recorded ``rendered_sha256`` (so an edited wrapper cannot be
    rendered under a stale hash), and ``strict_unique`` rejects duplicate
    (item,label,style) keys (which would otherwise silently keep the last row).
    """
    if not path.exists():
        return {}
    overrides: dict[tuple[str, str, str], str] = {}
    for record in load_jsonl(path):
        if _record_bool(record.get("added_operational_detail")):
            continue
        if _record_bool(record.get("refusal_or_warning")):
            continue
        if str(record.get("content_preservation", "")).strip().lower() == "low":
            continue
        raw_styled = record.get("styled_text")
        styled_text = str(raw_styled or "").strip()
        if not styled_text:
            continue
        key = (
            str(record.get("item_id")),
            str(record.get("safety_label")),
            str(record.get("target_style") or record.get("style")),
        )
        if verify_hash:
            # Every frozen attack row must carry a hash, and it must hash the exact
            # text that is actually used/spoken (the stripped styled_text), so a
            # padded row cannot pass with a hash of its raw (unstripped) form.
            expected = record.get("rendered_sha256")
            if expected is None:
                raise ValueError(
                    f"attack-variant row {key} in {path} has no rendered_sha256. "
                    "Re-run prepare_attack_variants.py to (re)freeze the file."
                )
            actual = hashlib.sha256(styled_text.encode("utf-8")).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"attack-variant hash mismatch for {key} in {path}: "
                    f"spoken text hashes to {actual} but rendered_sha256={expected}. "
                    "Re-run prepare_attack_variants.py instead of editing the file."
                )
        if strict_unique and key in overrides:
            raise ValueError(f"duplicate attack-variant override for {key} in {path}")
        overrides[key] = styled_text
    return overrides


def _load_style_text_overrides(
    data_dir: Path,
    dataset_cfg: AudioRdoDatasetConfig,
) -> tuple[dict[tuple[str, str, str], str], set[tuple[str, str, str]]]:
    """Load reviewed/generated style rewrites for TTS, if present.

    Returns ``(merged_overrides, attack_keys)``. ``merged_overrides`` merges the
    emotion style-variant file with the optional attack-variant file (frozen
    jailbreak wrappers). ``attack_keys`` are exactly the keys sourced from the
    hash-verified, de-duplicated attack file, so the caller can require that a
    text-attack style is satisfied by the ATTACK file specifically (an emotion row
    keyed with an attack style must never masquerade as a verified attack).
    """
    overrides: dict[tuple[str, str, str], str] = {}
    cfg = dataset_cfg.style_variant_generation
    if cfg.enabled:
        overrides.update(_parse_variant_override_file(data_dir / cfg.output_file))
    attack_keys: set[tuple[str, str, str]] = set()
    if dataset_cfg.attack_variant_file is not None:
        attack_overrides = _parse_variant_override_file(
            data_dir / dataset_cfg.attack_variant_file,
            verify_hash=True,
            strict_unique=True,
        )
        attack_keys = set(attack_overrides)
        overrides.update(attack_overrides)
    return overrides, attack_keys


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
    If ``batch_command_template`` is configured, pending TTS jobs are written to a
    JSONL file and rendered with one or more long-lived TTS worker processes.
    """
    audio_root = data_dir / tts_cfg.audio_subdir
    records: list[dict[str, object]] = []
    pending_jobs: list[dict[str, object]] = []
    style_text_overrides, attack_override_keys = _load_style_text_overrides(data_dir, dataset_cfg)
    # Text-only attack styles (jailbreak) carry the attack purely in wording, so a
    # missing wrapper must FAIL, not silently fall back to the clean base text
    # (which would label clean audio as an attacked cell). The override must come
    # from the hash-verified attack file, not an emotion row keyed with this style.
    # Emotion styles may fall back to base text (acoustic-only) and are not here.
    require_override_styles = set(dataset_cfg.neutral_acoustic_styles)

    if not dry_run and not (tts_cfg.batch_command_template or tts_cfg.command_template):
        raise ValueError(
            "dataset.tts.command_template or dataset.tts.batch_command_template "
            "is required unless dry_run=True"
        )

    for pair in pairs:
        for safety_label, text in (
            ("harmful", pair.harmful_text),
            ("benign", pair.benign_text),
        ):
            for style in dataset_cfg.styles:
                style_key = (pair.item_id, safety_label, style)
                if style in require_override_styles and style_key not in attack_override_keys:
                    raise ValueError(
                        f"style {style!r} is a text-attack style (dataset."
                        f"neutral_acoustic_styles) but has no hash-verified wrapped "
                        f"text in the attack file for item {pair.item_id!r} "
                        f"({safety_label}). Run ./scripts/prepare_attack_variants.py "
                        "and set dataset.attack_variant_file before rendering."
                    )
                render_text = style_text_overrides.get(style_key, text)
                style_text_source = (
                    "openrouter_style_variant" if render_text != text else "base_pair_text"
                )
                output = expected_audio_path(
                    audio_root,
                    content_id=pair.item_id,
                    safety_label=safety_label,
                    style=style,
                )
                values = {
                    "text": render_text,
                    "text_json": json.dumps(render_text, ensure_ascii=False),
                    "style": style,
                    "output": str(output),
                    "output_path": str(output),
                    "item_id": pair.item_id,
                    "query_id": pair.item_id,
                    "safety_label": safety_label,
                    "query_type": safety_label,
                    "overwrite": "true" if tts_cfg.overwrite else "false",
                }
                # Force neutral acoustics for text-only attack conditions (e.g.
                # jailbreak wrappers): the batch renderer reads style_instruction.
                if style in dataset_cfg.neutral_acoustic_styles:
                    values["style_instruction"] = NEUTRAL_STYLE_INSTRUCTION
                command = (
                    _format_command(tts_cfg.command_template, values)
                    if tts_cfg.command_template
                    else None
                )
                # Hash the exact spoken text; a sidecar records what each wav was
                # rendered from so a reused file whose text later changed is caught
                # instead of silently mislabeled (freeze integrity).
                reference_sha256 = hashlib.sha256(render_text.encode("utf-8")).hexdigest()
                hash_path = output.with_name(output.name + ".sha256")
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and not tts_cfg.overwrite:
                    prior = hash_path.read_text().strip() if hash_path.exists() else None
                    if prior is None and style in require_override_styles:
                        # Attack audio must be provenance-tracked; an existing wav
                        # with no sidecar could be leftover clean/other-text audio.
                        raise ValueError(
                            f"attack-style render {output} exists without a "
                            f"{hash_path.name} provenance sidecar. Delete it or set "
                            "tts.overwrite=true to re-render."
                        )
                    if prior is not None and prior != reference_sha256:
                        raise ValueError(
                            f"stale render at {output}: existing audio was rendered "
                            f"from text hashing to {prior}, but the current manifest "
                            f"text hashes to {reference_sha256}. Set tts.overwrite=true "
                            f"or delete {output.name} / {hash_path.name}."
                        )
                    status = "exists"
                elif dry_run:
                    status = "planned"
                elif tts_cfg.batch_command_template:
                    pending_jobs.append(values)
                    status = "queued"
                else:
                    if command is None:
                        raise ValueError("dataset.tts.command_template is required")
                    # Delete any stale file first so overwrite is guaranteed even if
                    # the configured command would skip an existing path; only then
                    # is the sidecar an honest record of what was just rendered.
                    if tts_cfg.overwrite and output.exists():
                        output.unlink()
                    subprocess.run(command, check=True, timeout=None)
                    hash_path.write_text(reference_sha256)
                    status = "rendered"

                records.append(
                    {
                        "item_id": pair.item_id,
                        "category": pair.category,
                        "safety_label": safety_label,
                        "style": style,
                        "path": str(output.relative_to(data_dir)),
                        "reference_text": render_text,
                        # Hash of the exact spoken text (auditable against the frozen
                        # attack file; also flags a reused/stale wav whose manifest
                        # text later changed). Persisted next to the wav as a sidecar.
                        "reference_sha256": reference_sha256,
                        "hash_path": str(hash_path.relative_to(data_dir)),
                        "base_reference_text": text if render_text != text else None,
                        "style_text_source": style_text_source,
                        "transcript": None,
                        "wer": None,
                        "duration_s": None,
                        "style_passed": True,
                        "core_tokens_preserved": None,
                        "status": status,
                        "command": command if dry_run else None,
                    }
                )

    if pending_jobs:
        batch_jobs_path = data_dir / tts_cfg.batch_jobs_file
        save_jsonl(pending_jobs, batch_jobs_path)
        _run_batch_shards(tts_cfg, batch_jobs_path, pending_jobs)
        missing = [
            str(job["output_path"])
            for job in pending_jobs
            if not Path(str(job["output_path"])).exists()
        ]
        if missing:
            examples = ", ".join(missing[:3])
            raise RuntimeError(
                "TTS batch command completed but "
                f"{len(missing)} output files are missing: {examples}"
            )
        for record in records:
            if record["status"] == "queued":
                record["status"] = "rendered"
                # Persist the render-text hash sidecar so a later reuse of this wav
                # is checked against the text it was actually rendered from.
                sidecar = data_dir / str(record["hash_path"])
                sidecar.write_text(str(record["reference_sha256"]))

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
