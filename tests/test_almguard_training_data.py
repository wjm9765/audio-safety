"""CPU-only tests for fail-closed Run 9 ALMGuard SAP staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_safety.pipelines import almguard_training_data as td


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fake_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "ALMGuard"
    indices = {
        "advwave_p": [0, 1, 2, 3, 4],
        "advwave_suffix": [0, 1, 3, 4],
        "pair_audio": [0, 1, 2, 3, 4],
    }
    for family, source_indices in indices.items():
        directory = repo / "results" / family
        directory.mkdir(parents=True)
        for index in source_indices:
            (directory / f"{index}.wav").write_bytes(f"{family}-{index}".encode())

    clean = repo / "datasets" / "advbench_audios"
    clean.mkdir(parents=True)
    metadata = []
    for index in range(5):
        (clean / f"{index}.wav").write_bytes(f"clean-{index}".encode())
        metadata.append(
            {"audio": f"{index}.wav", "prompt": f"unique request {index}", "target": "x"}
        )
    metadata_path = repo / "datasets" / "AdvBench_Audio.json"
    _write_jsonl(metadata_path, metadata)

    data_dir = tmp_path / "data"
    eval_manifest = data_dir / "manifests" / "run9_fresh_clean.jsonl"
    _write_jsonl(
        eval_manifest,
        [
            {
                "item_id": "figstep_safebench_0000",
                "path": "audio/eval/0.wav",
                "reference_text": "different evaluation request",
                "source": "figstep_safebench",
            }
        ],
    )
    return repo, data_dir, eval_manifest


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo, data_dir, eval_manifest = _fake_inputs(tmp_path)
    output = data_dir / "almguard_sap_inputs"
    monkeypatch.setattr(td, "_git_commit", lambda _repo: "pinned-commit")
    td.prepare_data(
        almguard_repo=repo,
        data_dir=data_dir,
        output_root=output,
        metadata_path=repo / "datasets" / "AdvBench_Audio.json",
        eval_manifests=[eval_manifest],
        train_per_family=2,
        holdout_per_family=2,
        seed=0,
    )
    return output, data_dir


def test_prepare_equal_dirs_exact_zip_order_and_asr_alignment(tmp_path, monkeypatch):
    output, data_dir = _prepare(tmp_path, monkeypatch)
    contract = td.validate_prepared(output, data_dir=data_dir)

    assert contract["common_source_indices"] == [0, 1, 3, 4]
    assert contract["train_total"] == 6
    assert contract["positive_control_total"] == 6
    for split in ("train", "positive_control"):
        for family in td.FAMILIES:
            directory = output / split / family
            assert [path.name for path in td.numeric_wavs(directory)] == ["0.wav", "1.wav"]
            assert all(path.is_symlink() for path in directory.iterdir())

    rows = td.load_jsonl(output / "train_manifest.jsonl")
    assert [row["attack_family"] for row in rows] == list(td.FAMILIES) * 2
    assert [row["sequence_index"] for row in rows] == list(range(6))
    asr = td.load_jsonl(output / "train_asr_pairs.jsonl")
    assert [row["wav"] for row in asr] == [row["path"] for row in rows]
    assert [td.text_digest(row["txt"]) for row in asr] == [
        row["asr_reference_sha256"] for row in rows
    ]


def test_train_positive_control_are_item_path_and_text_disjoint(tmp_path, monkeypatch):
    output, _data_dir = _prepare(tmp_path, monkeypatch)
    train = td.load_jsonl(output / "train_manifest.jsonl")
    holdout = td.load_jsonl(output / "positive_control_manifest.jsonl")

    td.assert_train_holdout_disjoint(train, holdout)
    assert {row["source_index"] for row in train}.isdisjoint(row["source_index"] for row in holdout)


@pytest.mark.parametrize("leak_kind", ["source", "item", "text", "path"])
def test_eval_leakage_fails_closed(tmp_path, leak_kind):
    candidate = {
        "item_id": "advbench_audio_0000",
        "source_item_id": "advbench_audio_0000",
        "path": "staged/0.wav",
        "source_path": "/attacks/0.wav",
        "source_audio_path": "/clean/0.wav",
        "reference_text": "candidate request",
        "source": td.SOURCE_DATASET,
    }
    evaluation = {
        "item_id": "safe_0",
        "path": "eval/0.wav",
        "reference_text": "evaluation request",
        "source": "figstep_safebench",
    }
    if leak_kind == "source":
        evaluation["source"] = "AdvBench-Audio"
    elif leak_kind == "item":
        evaluation["item_id"] = candidate["item_id"]
    elif leak_kind == "text":
        evaluation["reference_text"] = candidate["reference_text"]
    else:
        evaluation["path"] = candidate["path"]
    manifest = tmp_path / "eval.jsonl"
    _write_jsonl(manifest, [evaluation])

    with pytest.raises(td.DataContractError, match="leaks into shared eval"):
        td.assert_no_eval_leakage([candidate], [manifest], tmp_path)


def test_phase_under_test_and_non_numeric_names_fail_closed(tmp_path):
    forbidden = tmp_path / "pv_standard" / "wavs"
    forbidden.mkdir(parents=True)
    (forbidden / "0.wav").write_bytes(b"x")
    with pytest.raises(td.DataContractError, match="phase-under-test"):
        td.numeric_wavs(forbidden, td.FORBIDDEN_TOKENS)

    invalid = tmp_path / "safe" / "wavs"
    invalid.mkdir(parents=True)
    (invalid / "clip.wav").write_bytes(b"x")
    with pytest.raises(td.DataContractError, match="integer WAV basenames"):
        td.numeric_wavs(invalid)


def test_asr_pair_reordering_is_detected(tmp_path, monkeypatch):
    output, data_dir = _prepare(tmp_path, monkeypatch)
    path = output / "train_asr_pairs.jsonl"
    rows = td.load_jsonl(path)
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(path, rows)

    with pytest.raises(td.DataContractError, match="ASR WAV order"):
        td.validate_prepared(output, data_dir=data_dir)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    output, data_dir = _prepare(tmp_path, monkeypatch)
    repo = tmp_path / "ALMGuard"
    with pytest.raises(td.DataContractError, match="refusing to overwrite"):
        td.prepare_data(
            almguard_repo=repo,
            data_dir=data_dir,
            output_root=output,
            metadata_path=repo / "datasets" / "AdvBench_Audio.json",
            eval_manifests=[data_dir / "manifests" / "run9_fresh_clean.jsonl"],
            train_per_family=2,
            holdout_per_family=2,
            seed=0,
        )
