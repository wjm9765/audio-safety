"""Audio rendering manifests and transcript-control records for Audio-RDO.

Actual TTS/ASR/style-classifier execution is intentionally outside this CPU-safe
module. The gate consumes manifests that record which CosyVoice2 render passed
WER, harmful-token, duration, and style checks.
"""

from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import AudioRdoDatasetConfig


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


def transcript_control_passed(audio: RenderedAudio, cfg: AudioRdoDatasetConfig) -> bool:
    """Apply manifest-level transcript/style controls.

    Geometry analysis must only consume samples that pass this predicate. Missing
    WER/style metadata fails closed.
    """
    if audio.wer is None or audio.wer > cfg.transcript_control.wer_max:
        return False
    if cfg.transcript_control.require_style_classifier_pass and audio.style_passed is not True:
        return False
    return True
