"""Text dataset loading for cone construction and drift-probe contents.

Status: interface stubs — dataset acquisition is Day 1-2 work (design.md §8).
Sources (design.md §2): AdvBench / HarmBench / SORRY-Bench (category-labeled harmful),
XSTest (borderline benign), Alpaca/just-eval (plain benign). See data/README.md.
"""

from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import ConeConfig, DriftConfig


@dataclass(frozen=True)
class TextItem:
    text: str
    category: str  # harm category, or "benign" / "borderline"
    source: str  # originating benchmark
    item_id: str


def load_harmful_texts(cfg: ConeConfig, data_dir: Path) -> dict[str, list[TextItem]]:
    """Category -> harmful texts for cone construction (design.md §2.1)."""
    raise NotImplementedError("Dataset preparation pending — see data/README.md and design.md §2.1")


def load_benign_texts(cfg: ConeConfig, data_dir: Path) -> list[TextItem]:
    """Benign + borderline (XSTest) texts; borderline is required so the axes capture
    refusal rather than harmful-topic detection (design.md §2.1)."""
    raise NotImplementedError("Dataset preparation pending — see data/README.md and design.md §2.1")


def select_drift_contents(
    harmful: dict[str, list[TextItem]], cfg: DriftConfig, seed: int
) -> list[TextItem]:
    """Fixed, category-balanced selection of the paired contents rendered into every
    family (design.md §2.2). Selection must be deterministic given the seed."""
    raise NotImplementedError("Dataset preparation pending — see design.md §2.2")
