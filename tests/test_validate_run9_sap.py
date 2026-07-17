from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "almguard" / "validate_run9_sap.py"
    spec = importlib.util.spec_from_file_location("_test_validate_run9_sap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_inspect_ptb_uses_child_cpu_and_parses_metrics(tmp_path, monkeypatch):
    python = tmp_path / "python"
    checkpoint = tmp_path / "sap.pth"
    mask = tmp_path / "mask.npz"
    for path in (python, checkpoint, mask):
        path.write_bytes(b"x")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "shape": [1, 128, 3000],
                    "dtype": "torch.float32",
                    "finite": True,
                    "max_abs": 0.5,
                    "outside_mask_max_abs": 0.0,
                    "nonzero": 12,
                }
            )
        )

    monkeypatch.setattr(SCRIPT.subprocess, "run", fake_run)
    result = SCRIPT._inspect_ptb(python, checkpoint, mask, 0.5)
    assert result["shape"] == [1, 128, 3000]
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["kwargs"]["capture_output"] is True
    assert str(checkpoint) in captured["command"]


def test_mask_contract_shape_and_count_fixture(tmp_path):
    mask = np.zeros(128)
    mask[:48] = 1
    path = tmp_path / "mask.npz"
    np.savez(path, avg_a=np.ones(128), avg_j=np.ones(128), mask=mask)
    loaded = np.load(path)
    assert set(loaded.files) == {"avg_a", "avg_j", "mask"}
    assert loaded["mask"].shape == (128,)
    assert np.count_nonzero(loaded["mask"]) == 48


def test_cli_rejects_invalid_epoch_or_tau():
    with pytest.raises(SystemExit, match="num_epochs must be"):
        SCRIPT.main(
            [
                "--prepared-root",
                "unused",
                "--data-dir",
                "unused",
                "--sap-dir",
                "unused",
                "--num-epochs",
                "0",
            ]
        )
    with pytest.raises(SystemExit, match="tau"):
        SCRIPT.main(
            [
                "--prepared-root",
                "unused",
                "--data-dir",
                "unused",
                "--sap-dir",
                "unused",
                "--tau",
                "0",
            ]
        )
