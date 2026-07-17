"""SARSteer paired-calibration selection (paper §3.2)."""

from __future__ import annotations

import pytest

from audio_safety.data.run9_sarsteer_calibration import (
    calibration_rows,
    paired_neutral_items,
    select_calibration_items,
)


def _render(item_id: str, label: str, *, style: str = "neutral"):
    return {
        "item_id": item_id,
        "safety_label": label,
        "style": style,
        "path": f"audio/{label}/{style}/{item_id}.wav",
        "reference_text": f"{label} text for {item_id}",
    }


def _pool(n: int = 4):
    rows = []
    for i in range(n):
        rows.append(_render(f"i{i}", "harmful"))
        rows.append(_render(f"i{i}", "benign"))
    return rows


def test_only_complete_neutral_pairs_qualify():
    rows = _pool(2)
    rows.append(_render("lonely", "harmful"))  # no purified counterpart
    rows.append(_render("styled", "harmful", style="jb_pap"))
    rows.append(_render("styled", "benign", style="jb_pap"))
    pairs = paired_neutral_items(rows)
    assert set(pairs) == {"i0", "i1"}


def test_selection_excludes_every_evaluation_item():
    rows = _pool(4)
    got = select_calibration_items(rows, excluded_item_ids={"i1", "i3"})
    assert got == ["i0", "i2"]


def test_selection_is_seeded_and_capped():
    rows = _pool(10)
    a = select_calibration_items(rows, excluded_item_ids=set(), n=4, seed=0)
    b = select_calibration_items(rows, excluded_item_ids=set(), n=4, seed=0)
    assert a == b and len(a) == 4
    assert select_calibration_items(rows, excluded_item_ids=set(), n=99) == sorted(
        f"i{i}" for i in range(10)
    )


def test_selection_fails_closed_when_everything_is_evaluated():
    rows = _pool(2)
    with pytest.raises(ValueError, match="no paired neutral item"):
        select_calibration_items(rows, excluded_item_ids={"i0", "i1"})


def test_calibration_rows_split_harmful_and_purified_halves():
    rows = _pool(2)
    harmful = calibration_rows(rows, ["i0", "i1"], label="harmful", source="figstep_safebench")
    benign = calibration_rows(rows, ["i0", "i1"], label="benign", source="figstep_safebench")
    # The pair shares an item_id: the safe half is the SAME question, purified.
    assert [r["item_id"] for r in harmful] == [r["item_id"] for r in benign] == ["i0", "i1"]
    assert {r["gate_role"] for r in harmful} == {"sarsteer_refusal_calib"}
    assert {r["gate_role"] for r in benign} == {"sarsteer_safe_pca"}
    assert harmful[0]["path"] != benign[0]["path"]
