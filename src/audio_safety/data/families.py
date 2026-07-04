"""Attack-family audio rendering and comprehension filtering.

Status: interface stubs — rendering is Day 1-2 work (design.md §8).

Families (design.md §2.2):
- plain     : standard TTS of the content (or reuse WeifeiJin/AdvBench-Audio)
- nonspeech : plain speech with environmental audio prepended/appended/mixed
- style     : emotion/accent/age/gender/rate variation, >=3 voice seeds
- perturbed : AJailBench Audio Perturbation Toolkit (time/frequency/amplitude)

Excluded from experiment 1: AdvWave-style gradient optimization (design.md §9).
"""

from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import DriftConfig


@dataclass(frozen=True)
class RenderedAudio:
    path: Path
    content_id: str
    family: str
    voice_seed: int | None = None


def render_family(
    content_text: str,
    content_id: str,
    family: str,
    cfg: DriftConfig,
    out_dir: Path,
    seed: int,
) -> list[RenderedAudio]:
    """Render one content into one family (style family returns one clip per voice
    seed). Must be deterministic given the seed."""
    raise NotImplementedError("Family rendering pending — see design.md §2.2")


def comprehension_check(audio: RenderedAudio, reference_text: str) -> bool:
    """Transcribe (Whisper or the target model) and verify semantic match with the
    reference content. Mandatory filter: without it, refusal is confounded with
    non-comprehension (design.md §3.1) — the most common rejection reason.

    Samples failing this check must be dropped from ALL families of that content
    to preserve pairing (see pipelines.drift.project_drifts).
    """
    raise NotImplementedError("Comprehension filter pending — see design.md §3.1")
