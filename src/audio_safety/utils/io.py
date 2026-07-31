"""Run artifact IO: JSON/JSONL helpers and reproducibility snapshots."""

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def save_json(obj: Any, path: Path) -> None:
    """Write JSON via an atomic replace.

    Concurrent workers sharing a run directory (e.g. sharded GPU runs writing
    provenance.json) must never observe a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_git_commit() -> str | None:
    """Current commit hash, or None outside a git checkout (e.g. deployed container)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_git_dirty() -> bool | None:
    """Whether tracked or untracked worktree files differ from ``HEAD``."""

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def snapshot_config(cfg: BaseModel, run_dir: Path) -> Path:
    """Write the fully-resolved config + git commit so the run is reproducible
    even if configs/ change later (AGENTS.md '실험 문서 규약')."""
    snapshot = {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "config": cfg.model_dump(mode="json"),
    }
    path = run_dir / "config_snapshot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True))
    return path
