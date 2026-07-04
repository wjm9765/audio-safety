"""Dataset manifests for the Audio-RDO gate.

The selected primary harmful source is FigStep/SafeBench. Geometry analysis does
not consume the harmful-only CSV directly: each harmful query must be paired with
a high-lexical-overlap benign counterpart and pass transcript controls.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import AudioRdoDatasetConfig, ConeConfig, DriftConfig


@dataclass(frozen=True)
class TextItem:
    text: str
    category: str  # harm category, or "benign" / "borderline"
    source: str  # originating benchmark
    item_id: str


@dataclass(frozen=True)
class AudioRdoPair:
    item_id: str
    category: str
    harmful_text: str
    benign_text: str
    source: str


@dataclass(frozen=True)
class AudioRdoSplit:
    train: list[AudioRdoPair]
    validation: list[AudioRdoPair]
    heldout: list[AudioRdoPair]


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return None


def load_harmful_seed_rows(path: Path, *, source: str) -> list[dict[str, str]]:
    """Load harmful-only seed rows from a public source such as FigStep SafeBench."""
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix == ".csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"unsupported harmful seed format {path.suffix!r}")

    normalized = []
    for idx, row in enumerate(rows):
        harmful = _first_present(row, ("harmful_text", "question", "goal", "instruction", "prompt"))
        if harmful is None:
            raise ValueError(f"row {idx} in {path} has no harmful text column")
        normalized.append(
            {
                "item_id": _first_present(row, ("item_id", "id", "idx")) or f"{source}_{idx:04d}",
                "category": _first_present(row, ("category", "topic", "type")) or "uncategorized",
                "harmful_text": harmful,
                "source": source,
            }
        )
    return normalized


def load_audio_rdo_pairs(data_dir: Path, cfg: AudioRdoDatasetConfig) -> list[AudioRdoPair]:
    """Load the curated harmful-benign pair manifest.

    Expected columns/keys: item_id, category, harmful_text, benign_text, source.
    ``cfg.seed_file`` is the harmful-only downloaded CSV; ``cfg.source_file`` is
    the curated pair manifest consumed by the experiment.
    """
    path = data_dir / cfg.source_file
    if not path.exists():
        raise FileNotFoundError(f"Audio-RDO pair manifest not found: {path}")

    rows = _read_jsonl(path) if path.suffix == ".jsonl" else _read_csv(path)
    pairs: list[AudioRdoPair] = []
    missing_benign = 0
    for idx, row in enumerate(rows):
        harmful = _first_present(row, ("harmful_text", "question", "goal", "instruction", "prompt"))
        benign = _first_present(row, ("benign_text", "safe_text", "benign_question"))
        if harmful is None:
            raise ValueError(f"row {idx} in {path} has no harmful text")
        if benign is None:
            missing_benign += 1
            continue
        item_id = _first_present(row, ("item_id", "id", "idx")) or (
            f"{cfg.harmful_source}_{idx:04d}"
        )
        pairs.append(
            AudioRdoPair(
                item_id=item_id,
                category=_first_present(row, ("category", "topic", "type")) or "uncategorized",
                harmful_text=harmful,
                benign_text=benign,
                source=_first_present(row, ("source",)) or cfg.harmful_source,
            )
        )

    if missing_benign:
        raise ValueError(
            f"{path} appears to be harmful-only ({missing_benign} rows lack benign_text). "
            "Create a curated harmful-benign pair manifest before running geometry analysis."
        )
    if len(pairs) < cfg.min_pairs:
        raise ValueError(f"need at least {cfg.min_pairs} pairs, found {len(pairs)} in {path}")
    return pairs


def split_audio_rdo_pairs(
    pairs: list[AudioRdoPair],
    cfg: AudioRdoDatasetConfig,
    seed: int,
) -> AudioRdoSplit:
    """Deterministic 40/20/40 split, capped at cfg.n_pairs."""
    import random

    selected = list(pairs)
    random.Random(seed).shuffle(selected)
    selected = selected[: cfg.n_pairs]

    n = len(selected)
    n_train = int(round(n * cfg.splits.train))
    n_validation = int(round(n * cfg.splits.validation))
    train = selected[:n_train]
    validation = selected[n_train : n_train + n_validation]
    heldout = selected[n_train + n_validation :]
    return AudioRdoSplit(train=train, validation=validation, heldout=heldout)


def load_harmful_texts(cfg: ConeConfig, data_dir: Path) -> dict[str, list[TextItem]]:
    """Category -> harmful texts for cone construction (design.md §2.1)."""
    raise NotImplementedError("Legacy cone-drift dataset loader is not used by exp1 Audio-RDO gate")


def load_benign_texts(cfg: ConeConfig, data_dir: Path) -> list[TextItem]:
    """Benign + borderline (XSTest) texts; borderline is required so the axes capture
    refusal rather than harmful-topic detection (design.md §2.1)."""
    raise NotImplementedError("Legacy cone-drift dataset loader is not used by exp1 Audio-RDO gate")


def select_drift_contents(
    harmful: dict[str, list[TextItem]], cfg: DriftConfig, seed: int
) -> list[TextItem]:
    """Fixed, category-balanced selection of the paired contents rendered into every
    family (design.md §2.2). Selection must be deterministic given the seed."""
    raise NotImplementedError("Legacy cone-drift selector is not used by exp1 Audio-RDO gate")
