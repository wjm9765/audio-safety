from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "merge_manual_sarsteer_labels.py"
    spec = importlib.util.spec_from_file_location("_test_merge_manual_sar_labels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def _paired(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "item_id": f"item-{record_id}",
        "gate_role": "harmful_eval",
        "safety_label": "harmful",
        "condition": "clean",
        "undefended_output": "raw u",
        "defended_output": "raw d",
    }


def _decision(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "undefended_reviewed_behavior_label": "policy_refusal",
        "defended_reviewed_behavior_label": "decoding_failure",
        "undefended_adjudication_note": "Direct refusal.",
        "defended_adjudication_note": "Incoherent repetition loop.",
        "manual_reviewed": True,
    }


def test_merge_preserves_metadata_hashes_bodies_and_canonical_order():
    paired = [_paired("r0"), _paired("r1")]
    merged = SCRIPT.merge_labels(paired, [[_decision("r1")], [_decision("r0")]])
    assert [row["record_id"] for row in merged] == ["r0", "r1"]
    assert merged[0]["gate_role"] == "harmful_eval"
    assert merged[0]["undefended_reviewed_behavior_label"] == "policy_refusal"
    assert "undefended_output" not in merged[0]
    assert len(merged[0]["undefended_output_sha256"]) == 64
    assert merged[0]["manual_adjudication_complete"] is True


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "invalid"])
def test_merge_fails_closed(failure):
    paired = [_paired("r0"), _paired("r1")]
    decisions = [_decision("r0"), _decision("r1")]
    if failure == "missing":
        decisions.pop()
        match = "do not cover"
    elif failure == "duplicate":
        decisions[1]["record_id"] = "r0"
        match = "duplicate manual"
    elif failure == "foreign":
        decisions[1]["record_id"] = "other"
        match = "foreign"
    else:
        decisions[1]["defended_reviewed_behavior_label"] = "not-a-label"
        match = "invalid"
    with pytest.raises(ValueError, match=match):
        SCRIPT.merge_labels(paired, [decisions])
