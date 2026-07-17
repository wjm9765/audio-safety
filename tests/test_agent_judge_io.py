"""Alignment guards for local agent adjudication.

These tests exist because a silent label/row misalignment would not crash — it
would just move the vulnerable set S and change the gate verdict.
"""

from __future__ import annotations

import hashlib

import pytest

from audio_safety.evaluation.agent_judge_io import build_batches, merge_label_batches


def _row(record_id: str, **over):
    row = {
        "record_id": record_id,
        "item_id": f"item_{record_id}",
        "gate_role": "harmful_eval",
        "safety_label": "harmful",
        "condition": "clean",
        "style": "neutral",
        "sign": None,
        "reference_text": "q?",
        "undefended_output": f"u-{record_id}",
        "defended_output": f"d-{record_id}",
    }
    row.update(over)
    return row


def _label(record_id: str, undefended="policy_refusal", defended="policy_refusal"):
    return {
        "record_id": record_id,
        "undefended_reviewed_behavior_label": undefended,
        "defended_reviewed_behavior_label": defended,
    }


def test_build_batches_chunks_and_carries_both_arms():
    batches = build_batches([_row(str(i)) for i in range(5)], batch_size=2)
    assert [len(b) for b in batches] == [2, 2, 1]
    first = batches[0][0]
    assert first["record_id"] == "0"
    assert first["undefended_output"] == "u-0"
    assert first["defended_output"] == "d-0"


def test_build_batches_rejects_duplicate_record_ids():
    with pytest.raises(ValueError, match="duplicate record_id"):
        build_batches([_row("a"), _row("a")], batch_size=10)


def test_merge_requires_every_row_labelled():
    rows = [_row("a"), _row("b")]
    with pytest.raises(ValueError, match="never labelled"):
        merge_label_batches(rows, [_label("a")], resolution="agent")


def test_merge_rejects_unknown_and_duplicate_labels():
    rows = [_row("a")]
    with pytest.raises(ValueError, match="unknown record_id"):
        merge_label_batches(rows, [_label("a"), _label("zzz")], resolution="agent")
    with pytest.raises(ValueError, match="duplicate label"):
        merge_label_batches(rows, [_label("a"), _label("a")], resolution="agent")


def test_merge_rejects_off_taxonomy_label():
    with pytest.raises(ValueError, match="expected one of"):
        merge_label_batches([_row("a")], [_label("a", undefended="refused_ish")], resolution="x")


def test_merge_binds_label_to_the_exact_judged_text():
    rows = [_row("a", undefended_output="I cannot help.")]
    merged = merge_label_batches(rows, [_label("a")], resolution="claude_agent_local")
    assert len(merged) == 1
    got = merged[0]
    assert got["undefended_reviewed_behavior_label"] == "policy_refusal"
    assert got["undefended_output_sha256"] == hashlib.sha256(b"I cannot help.").hexdigest()
    assert got["undefended_label_resolution"] == "claude_agent_local"
    assert got["manual_local_no_external_judge"] is True
