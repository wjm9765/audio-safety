"""Pure helpers for the isolated ALMGuard wrapper.

Kept in ``src/`` (no torch, no ALMGuard imports) so the alignment and safety-guard
logic — the two places a silent gate corruption can hide — are unit-testable on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def staged_wav_name(index: int, width: int = 6) -> str:
    """Zero-padded staged filename so lexicographic order == numeric order.

    ALMGuard returns responses in its OWN filename-sort order. Unpadded names
    (``0,1,10,2,...``) sort lexicographically, so ``responses[i]`` would be
    misattributed to row ``i`` for >= 10 rows — a silent scramble of the survival
    set. Zero-padding makes lexicographic and numeric sorts identical, so
    positional alignment is safe regardless of ALMGuard's internal sort.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if index >= 10**width:
        raise ValueError(f"index {index} exceeds zero-pad width {width}")
    return f"{index:0{width}d}.wav"


def staged_mapping_row(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve manifest metadata while assigning the staged positional index.

    In particular, Run 9 separates vocal ``style`` from experimental
    ``condition``.  Dropping ``condition`` here would collapse clean and
    attacked rows to the same legacy style and corrupt gate evaluation.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    return {
        **dict(row),
        "index": index,
        "staging_index": index,
        "staged_wav_name": staged_wav_name(index),
    }


def align_responses(responses: Sequence[Any], mapping: Sequence[dict]) -> list[dict]:
    """Attach each staged row's metadata to its response BY POSITION.

    Sound only because :func:`staged_wav_name` zero-pads (sorted file order ==
    index order == ``mapping`` order). Missing responses become ``None``.
    """
    records = []
    for row in mapping:
        i = int(row["index"])
        records.append({**row, "output": responses[i] if i < len(responses) else None})
    return records


def excluded_training_files(adv_dirs: Sequence[Path | str], tag: str) -> list[str]:
    """Full paths under ``adv_dirs`` (including each dir itself) whose RESOLVED path
    contains ``tag``.

    The attack under test must never train the SAP (an attack-aware defense makes
    the gate meaningless). Matching the full resolved path — and the dir entries
    themselves, not just descendant basenames — catches the tag whether it is in a
    file name, a subdir, or the top-level adv-dir name.
    """
    if not tag:
        return []
    hits: list[str] = []
    for d in adv_dirs:
        base = Path(d)
        for candidate in [base, *base.glob("**/*")]:
            if tag in str(candidate.resolve()):
                hits.append(str(candidate))
    return hits
