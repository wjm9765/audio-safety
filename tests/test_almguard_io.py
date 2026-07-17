"""Tests for the ALMGuard wrapper's alignment + safety-guard logic (CPU, no torch).

These cover the two silent-corruption sites flagged in review: response<->row
alignment and the attack-under-test exclusion guard.
"""

from __future__ import annotations

import pytest

from audio_safety.pipelines.almguard_io import (
    align_responses,
    excluded_training_files,
    staged_wav_name,
)


def test_staged_names_sort_lexicographically_equals_index_order():
    # The alignment is positional; it is only sound if the staged filenames sort
    # the same way lexicographically as by index. Zero-padding guarantees this.
    names = [staged_wav_name(i) for i in range(123)]
    assert sorted(names) == names
    # The unpadded scheme this replaced would FAIL here:
    unpadded = [f"{i}.wav" for i in range(123)]
    assert sorted(unpadded) != unpadded


def test_staged_name_validation():
    assert staged_wav_name(0) == "000000.wav"
    with pytest.raises(ValueError):
        staged_wav_name(-1)
    with pytest.raises(ValueError):
        staged_wav_name(10**6)  # exceeds width 6


def test_align_responses_attaches_each_row_to_its_own_output():
    mapping = [
        {"index": i, "item_id": f"item{i}", "safety_label": "harmful", "style": "phase"}
        for i in range(12)
    ]
    # Responses in staged (index) order — guaranteed by zero-padding.
    responses = [f"resp{i}" for i in range(12)]
    records = align_responses(responses, mapping)
    assert len(records) == 12
    for i, rec in enumerate(records):
        assert rec["item_id"] == f"item{i}"
        assert rec["output"] == f"resp{i}"  # row i <-> its own audio's response


def test_align_responses_missing_response_is_none():
    mapping = [{"index": 0, "item_id": "a"}, {"index": 1, "item_id": "b"}]
    records = align_responses(["only-one"], mapping)
    assert records[0]["output"] == "only-one"
    assert records[1]["output"] is None


def test_excluded_guard_catches_tag_in_dir_name(tmp_path):
    # The attack tag is in the TOP-LEVEL dir name, not in any file name.
    adv = tmp_path / "phase_render"
    adv.mkdir()
    (adv / "clip0.wav").write_bytes(b"x")
    (adv / "clip1.wav").write_bytes(b"x")
    hits = excluded_training_files([adv], "phase")
    assert hits, "tag in the adv-dir name must be caught (attack-aware SAP guard)"


def test_excluded_guard_catches_tag_in_file_name(tmp_path):
    adv = tmp_path / "advwave_p"
    adv.mkdir()
    (adv / "phase_attack_0.wav").write_bytes(b"x")
    hits = excluded_training_files([adv], "phase")
    assert hits


def test_excluded_guard_clean_dirs_pass(tmp_path):
    adv = tmp_path / "advwave_suffix"
    adv.mkdir()
    (adv / "clip0.wav").write_bytes(b"x")
    assert excluded_training_files([adv], "phase") == []
    assert excluded_training_files([adv], "") == []
