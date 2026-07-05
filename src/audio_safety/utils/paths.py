"""Workspace / cache / output path resolution.

Priority (AGENTS.md '경로와 캐시 정책'):
    1. explicit CLI argument
    2. project environment variable (AUDIO_SAFETY_*)
    3. config file value
    4. /workspace/audio_safety_data/{data,outputs,cache}

Never reference ~/.cache, /root, or personal absolute paths anywhere in code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from audio_safety.config.schema import PathsConfig

ENV_WORKSPACE = "AUDIO_SAFETY_WORKSPACE"
ENV_DATA_DIR = "AUDIO_SAFETY_DATA_DIR"
ENV_OUTPUT_DIR = "AUDIO_SAFETY_OUTPUT_DIR"
ENV_CACHE_DIR = "AUDIO_SAFETY_CACHE_DIR"

DEFAULT_WORKSPACE = Path("/workspace/audio_safety_data")


@dataclass(frozen=True)
class ResolvedPaths:
    workspace: Path
    data_dir: Path
    output_dir: Path
    cache_dir: Path


def _pick(cli: Path | str | None, env_var: str, config: Path | None, default: Path) -> Path:
    if cli is not None:
        return Path(cli)
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    if config is not None:
        return Path(config)
    return default


def resolve_paths(
    config: PathsConfig | None = None,
    *,
    workspace: Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> ResolvedPaths:
    """Resolve all project paths. Keyword arguments are CLI-level overrides."""
    config = config or PathsConfig()
    ws = _pick(workspace, ENV_WORKSPACE, config.workspace, DEFAULT_WORKSPACE)
    return ResolvedPaths(
        workspace=ws,
        data_dir=_pick(data_dir, ENV_DATA_DIR, config.data_dir, ws / "data"),
        output_dir=_pick(output_dir, ENV_OUTPUT_DIR, config.output_dir, ws / "outputs"),
        cache_dir=_pick(cache_dir, ENV_CACHE_DIR, config.cache_dir, ws / "cache"),
    )


def run_output_dir(output_dir: Path, run_name: str, *, create: bool = True) -> Path:
    """Directory for one run's artifacts (config snapshot, metrics, figures)."""
    run_dir = output_dir / run_name
    if create:
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir
