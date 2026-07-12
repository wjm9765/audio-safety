"""YAML config loading with file references and dotted CLI overrides.

Usage:
    cfg = load_experiment_config(
        "configs/experiments/exp1_refusal_cone_drift.yaml",
        overrides=["stats.n_permutations=1000", "seed=1"],
    )

An experiment config may reference other config files by path (relative to the
repo root / CWD, falling back to the referencing file's directory)::

    model: configs/models/qwen2_audio.yaml
    paths: configs/paths/default.yaml
    dataset: configs/datasets/audio_rdo_gate.yaml
"""

from pathlib import Path
from typing import Any

import yaml

from audio_safety.config.schema import ExperimentConfig
from audio_safety.utils.env import load_project_dotenv

# Top-level keys whose string value is treated as a path to another YAML file.
_FILE_REF_KEYS = ("model", "paths", "dataset")


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {path}, got {type(data)}")
    return data


def _resolve_ref(ref: str, base_dir: Path) -> Path:
    """Resolve a config file reference: CWD-relative first, then relative to the
    referencing config's directory."""
    candidates = [Path(ref), base_dir / ref]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config reference {ref!r} not found (tried {candidates})")


def _apply_override(raw: dict[str, Any], spec: str) -> None:
    """Apply a single ``dotted.key=value`` override in place. Values are parsed as YAML
    so ``seed=1`` is an int and ``drift.families=[plain,style]`` is a list."""
    key, sep, value = spec.partition("=")
    if not sep:
        raise ValueError(f"Override {spec!r} must have the form key=value")
    parts = key.strip().split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Override {spec!r}: no such config section {part!r}")
        node = node[part]
    node[parts[-1]] = yaml.safe_load(value)


def load_experiment_config(
    path: Path | str,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    # Load .env before any run so OPENROUTER_API_KEY (and other secrets) resolve
    # from the project .env instead of requiring a manual shell export. No-op when
    # .env is absent (CPU tests / CI); never overrides an already-set variable.
    load_project_dotenv()

    path = Path(path)
    raw = load_yaml(path)

    for key in _FILE_REF_KEYS:
        if isinstance(raw.get(key), str):
            raw[key] = load_yaml(_resolve_ref(raw[key], path.parent))

    for spec in overrides or []:
        _apply_override(raw, spec)

    return ExperimentConfig.model_validate(raw)
