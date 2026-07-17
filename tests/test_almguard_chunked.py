from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "almguard" / "run_eval_chunked.py"
    spec = importlib.util.spec_from_file_location("_test_run_eval_chunked", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical_rows(tmp_path: Path, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        audio = tmp_path / f"{index}.wav"
        audio.write_bytes(b"wav")
        rows.append(
            {
                "record_id": f"record-{index}",
                "item_id": f"item-{index}",
                "safety_label": "harmful",
                "condition": "pv_standard",
                "sign": -3 if index % 2 == 0 else 3,
                "path": audio.name,
                "reference_text": f"request-{index}",
            }
        )
    return rows


def _install_fake_child(monkeypatch, calls: list[tuple[list[str], dict]], *, bad=None):
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        manifest = Path(command[command.index("--manifest") + 1])
        output = Path(command[command.index("--out") + 1])
        mode = command[command.index("--mode") + 1]
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        child = []
        for index, row in enumerate(rows):
            record = {
                **row,
                "index": index,
                "staging_index": index,
                "staged_wav_name": f"{index:06d}.wav",
                "defense": "none" if mode == "undefended" else "almguard",
                "output": f"response-{row['record_id']}",
            }
            if bad is not None:
                bad(record, index)
            child.append(record)
        _save(output, child)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(SCRIPT.subprocess, "run", fake_run)


def test_chunked_run_validates_and_checkpoints_in_canonical_order(tmp_path, monkeypatch):
    rows = _canonical_rows(tmp_path, 5)
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    cache = tmp_path / "cache"
    _save(manifest, rows)
    calls = []
    _install_fake_child(monkeypatch, calls)

    SCRIPT.main(
        [
            "--mode",
            "undefended",
            "--manifest",
            str(manifest),
            "--data-dir",
            str(tmp_path),
            "--out",
            str(output),
            "--cache-dir",
            str(cache),
            "--chunk-size",
            "2",
        ]
    )

    completed = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 3
    assert [row["record_id"] for row in completed] == [
        row["record_id"] for row in rows
    ]
    assert [row["index"] for row in completed] == list(range(5))
    assert [row["canonical_index"] for row in completed] == list(range(5))
    assert [row["invocation_index"] for row in completed] == [0, 1, 0, 1, 0]
    assert all(row["defense"] == "none" for row in completed)
    work_dirs = [
        Path(command[command.index("--work-dir") + 1])
        for command, _ in calls
    ]
    run_ids = {path.parents[1].name for path in work_dirs}
    assert len(run_ids) == 1
    assert len(next(iter(run_ids))) == 32


def test_resume_requires_exact_global_prefix_and_rejects_child_drift(tmp_path, monkeypatch):
    rows = _canonical_rows(tmp_path, 4)
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    perturb = tmp_path / "sap.pth"
    perturb.write_bytes(b"sap")
    _save(manifest, rows)
    prefix = [
        {
            **row,
            "index": index,
            "canonical_index": index,
            "invocation_index": index,
            "staging_index": index,
            "staged_wav_name": f"{index:06d}.wav",
            "defense": "almguard",
            "output": f"existing-{index}",
        }
        for index, row in enumerate(rows[:2])
    ]
    _save(output, prefix)
    calls = []
    _install_fake_child(monkeypatch, calls)

    argv = [
        "--mode",
        "defended",
        "--manifest",
        str(manifest),
        "--data-dir",
        str(tmp_path),
        "--out",
        str(output),
        "--perturb-path",
        str(perturb),
        "--chunk-size",
        "2",
        "--resume",
    ]
    SCRIPT.main(argv)
    completed = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
    assert [row["index"] for row in completed] == [0, 1, 2, 3]
    assert [row["invocation_index"] for row in completed] == [0, 1, 0, 1]
    assert completed[0]["output"] == "existing-0"

    completed[0]["canonical_index"] = 99
    _save(output, completed)
    calls.clear()
    with pytest.raises(SystemExit, match="global index/canonical_index"):
        SCRIPT.main(argv)
    assert calls == []

    with pytest.raises(SystemExit):
        SCRIPT.parse_args([*argv, "--overwrite"])
