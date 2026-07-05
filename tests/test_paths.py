from pathlib import Path

from audio_safety.config.schema import PathsConfig
from audio_safety.utils.paths import ENV_WORKSPACE, resolve_paths


def test_default_workspace(monkeypatch):
    monkeypatch.delenv(ENV_WORKSPACE, raising=False)
    paths = resolve_paths(PathsConfig())
    assert paths.workspace == Path("/workspace/audio_safety_data")
    assert paths.cache_dir == Path("/workspace/audio_safety_data/cache")


def test_env_beats_config(monkeypatch):
    monkeypatch.setenv(ENV_WORKSPACE, "/workspace/from_env")
    paths = resolve_paths(PathsConfig(workspace=Path("/workspace/from_config")))
    assert paths.workspace == Path("/workspace/from_env")
    assert paths.data_dir == Path("/workspace/from_env/data")


def test_cli_beats_env(monkeypatch):
    monkeypatch.setenv(ENV_WORKSPACE, "/workspace/from_env")
    paths = resolve_paths(PathsConfig(), workspace="/workspace/from_cli")
    assert paths.workspace == Path("/workspace/from_cli")


def test_config_beats_default(monkeypatch):
    monkeypatch.delenv(ENV_WORKSPACE, raising=False)
    paths = resolve_paths(PathsConfig(workspace=Path("/workspace/from_config")))
    assert paths.workspace == Path("/workspace/from_config")


def test_explicit_subdir_override(monkeypatch):
    monkeypatch.delenv(ENV_WORKSPACE, raising=False)
    paths = resolve_paths(PathsConfig(), output_dir="/workspace/other/outputs")
    assert paths.output_dir == Path("/workspace/other/outputs")
    assert paths.workspace == Path("/workspace/audio_safety_data")
