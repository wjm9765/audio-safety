"""Tests for the .env loader integration (config secrets like OPENROUTER_API_KEY)."""

import os

import audio_safety.utils.env as envmod
from audio_safety.config import load_experiment_config


def test_load_project_dotenv_loads_new_but_keeps_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "_LOADED", False)
    (tmp_path / ".env").write_text(
        "# a comment\nexport OPENROUTER_API_KEY=from_dotenv\nFOO_ENVTEST=\"bar\"\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "already_set")
    monkeypatch.delenv("FOO_ENVTEST", raising=False)

    assert envmod.load_project_dotenv() is True
    # an existing variable must win over .env
    assert os.environ["OPENROUTER_API_KEY"] == "already_set"
    # a variable only present in .env is loaded (quotes and `export ` handled)
    assert os.environ["FOO_ENVTEST"] == "bar"


def test_load_project_dotenv_missing_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "_LOADED", False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert envmod.load_project_dotenv() is False


def test_config_loader_triggers_dotenv(tmp_path, monkeypatch):
    # load_experiment_config should invoke the .env loader (no raise, no-op ok).
    called = {"n": 0}
    monkeypatch.setattr(
        "audio_safety.config.loader.load_project_dotenv",
        lambda: called.__setitem__("n", called["n"] + 1),
    )
    load_experiment_config("configs/experiments/run4_conversion_gap.yaml")
    assert called["n"] == 1
