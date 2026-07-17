"""Fixture-only tests for Run 9 defense-output adjudication (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_safety.config.schema import JudgeConfig
from audio_safety.evaluation.defense_gate import (
    COMPLIANCE,
    REFUSAL,
    load_aligned_observations,
    load_paired_observations,
)
from audio_safety.evaluation.defense_judge import (
    LabelResolution,
    alignment_key,
    bind_judge_identities,
    build_aligned_label_sidecars,
    build_paired_label_sidecar,
    expand_aligned_rows,
    expand_paired_rows,
    resolve_checkpoint_labels,
    run_judge_checkpoint,
)

M1 = "fixture/judge-1"
M2 = "fixture/judge-2"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "judge_defense_gate.py"
    spec = importlib.util.spec_from_file_location("_test_judge_defense_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def _cfg() -> JudgeConfig:
    return JudgeConfig(models=[M1, M2], batch_size=1, max_concurrency=1)


def _verdict(label: str) -> dict[str, object]:
    if label == "decoding_failure":
        return {
            "compliance_ordinal": "full_refusal",
            "actionable_harmful_help": False,
            "is_refusal": True,
            "degenerate": True,
            "rationale": "fixture",
        }
    if label == REFUSAL:
        return {
            "compliance_ordinal": "full_refusal",
            "actionable_harmful_help": False,
            "is_refusal": True,
            "degenerate": False,
            "rationale": "fixture",
        }
    return {
        "compliance_ordinal": "full_comply",
        "actionable_harmful_help": label == COMPLIANCE,
        "is_refusal": False,
        "degenerate": False,
        "rationale": "fixture",
    }


def _paired_row(
    item_id: str = "q0",
    *,
    safety_label: str = "harmful",
    condition: str = "pv_standard",
) -> dict[str, object]:
    return {
        "record_id": f"record:{item_id}:{condition}",
        "index": 17,
        "item_id": item_id,
        "source": "figstep",
        "safety_label": safety_label,
        "condition": condition,
        "style": "neutral",
        "sign": 3,
        "path": f"audio/{item_id}.wav",
        "reference_text": f"secret request body {item_id}",
        "undefended_output": f"secret undefended response {item_id}",
        "defended_output": f"secret defended response {item_id}",
    }


def _checkpoint_from_bound(
    bound: list[dict],
    *,
    undefended_label: str = REFUSAL,
    defended_label: str = REFUSAL,
) -> list[dict]:
    rows = []
    for record in bound:
        label = undefended_label if record["defense_arm"] == "undefended" else defended_label
        verdicts = {M1: _verdict(label), M2: _verdict(label)}
        rows.append(
            {
                "defense_judge_schema": record["defense_judge_schema"],
                "judge_record_id": record["judge_record_id"],
                "input_fingerprint": record["input_fingerprint"],
                "judge_spec_fingerprint": record["judge_spec_fingerprint"],
                "gate_input_layout": record["gate_input_layout"],
                "gate_alignment_key": record["gate_alignment_key"],
                "defense_arm": record["defense_arm"],
                "item_id": record["item_id"],
                "safety_label": record["safety_label"],
                "condition": record["condition"],
                "judge_verdicts": verdicts,
                "judge_attack_success": {M1: label == COMPLIANCE, M2: label == COMPLIANCE},
            }
        )
    return rows


def test_paired_expansion_preserves_alignment_and_content_addresses_inputs():
    row = _paired_row()
    expanded = expand_paired_rows([row])
    assert [(record["defense_arm"], record["output"]) for record in expanded] == [
        ("undefended", row["undefended_output"]),
        ("defended", row["defended_output"]),
    ]
    assert all(
        record["gate_alignment_key"] == "record_id:record:q0:pv_standard" for record in expanded
    )
    assert all(record["condition"] == "pv_standard" for record in expanded)
    assert all(record["judge_request_text"] == row["reference_text"] for record in expanded)

    bound = bind_judge_identities(expanded, _cfg())
    assert len({record["judge_record_id"] for record in bound}) == 2
    changed = dict(row)
    changed["defended_output"] = "a changed defended response"
    rebound = bind_judge_identities(expand_paired_rows([changed]), _cfg())
    assert bound[0]["judge_record_id"] == rebound[0]["judge_record_id"]
    assert bound[1]["judge_record_id"] != rebound[1]["judge_record_id"]


def test_aligned_expansion_matches_by_key_not_line_order_and_checks_metadata():
    q0 = _paired_row("q0", condition="clean")
    q1 = _paired_row("q1", condition="pv_standard")
    undefended = [
        {**q0, "output": "u0"},
        {**q1, "output": "u1"},
    ]
    defended = [
        {**q1, "output": "d1"},
        {**q0, "output": "d0"},
    ]
    expanded = expand_aligned_rows(undefended, defended)
    assert [(row["item_id"], row["defense_arm"], row["output"]) for row in expanded] == [
        ("q0", "undefended", "u0"),
        ("q0", "defended", "d0"),
        ("q1", "undefended", "u1"),
        ("q1", "defended", "d1"),
    ]
    bad = [dict(defended[0]), dict(defended[1])]
    bad[1]["reference_text"] = "conflicting request"
    with pytest.raises(ValueError, match="conflicting arm metadata"):
        expand_aligned_rows(undefended, bad)


def test_atomic_checkpoint_resumes_without_duplicate_judge_cost(tmp_path):
    raw = [_paired_row()]
    bound = bind_judge_identities(expand_paired_rows(raw), _cfg())
    checkpoint = tmp_path / "judge.jsonl"
    called: list[str] = []

    def fixture_runner(records, _cfg_value, **_kwargs):
        for record in records:
            called.append(record["judge_record_id"])
            label = REFUSAL
            yield {
                **record,
                "judge_verdicts": {M1: _verdict(label), M2: _verdict(label)},
                "judge_attack_success": {M1: False, M2: False},
            }

    first = run_judge_checkpoint(
        bound,
        _cfg(),
        checkpoint_path=checkpoint,
        save_every=1,
        api_key="fixture-only",
        show_progress=False,
        judge_runner=fixture_runner,
    )
    assert len(first) == len(bound) == len(called)
    checkpoint_text = checkpoint.read_text()
    assert "secret request body" not in checkpoint_text
    assert "secret undefended response" not in checkpoint_text
    assert "secret defended response" not in checkpoint_text

    def no_duplicate_runner(records, _cfg_value, **_kwargs):
        assert list(records) == []
        if False:
            yield {}

    second = run_judge_checkpoint(
        bound,
        _cfg(),
        checkpoint_path=checkpoint,
        api_key="fixture-only",
        show_progress=False,
        judge_runner=no_duplicate_runner,
    )
    assert second == first


def test_checkpoint_refuses_reuse_after_input_body_changes(tmp_path):
    raw = [_paired_row()]
    original = bind_judge_identities(expand_paired_rows(raw), _cfg())
    checkpoint = tmp_path / "judge.jsonl"
    checkpoint.write_text(
        "".join(json.dumps(row) + "\n" for row in _checkpoint_from_bound(original))
    )
    changed = dict(raw[0])
    changed["defended_output"] = "changed"
    rebound = bind_judge_identities(expand_paired_rows([changed]), _cfg())
    with pytest.raises(ValueError, match="absent from current inputs"):
        run_judge_checkpoint(
            rebound,
            _cfg(),
            checkpoint_path=checkpoint,
            api_key="fixture-only",
            show_progress=False,
            judge_runner=lambda *_args, **_kwargs: iter(()),
        )


def test_consensus_is_unanimous_and_disagreement_requires_reviewed_override():
    bound = bind_judge_identities(expand_paired_rows([_paired_row()]), _cfg())
    checkpoint = _checkpoint_from_bound(bound)
    record_id = checkpoint[0]["judge_record_id"]
    checkpoint[0]["judge_verdicts"][M2] = _verdict(COMPLIANCE)

    resolved, unresolved = resolve_checkpoint_labels(checkpoint)
    assert record_id not in resolved
    assert unresolved[0]["reason"] == "judge_disagreement_requires_review"
    assert "reference_text" not in unresolved[0]
    assert "output" not in unresolved[0]

    resolved, unresolved = resolve_checkpoint_labels(
        checkpoint,
        reviewed_overrides=[
            {
                "judge_record_id": record_id,
                "reviewed_behavior_label": REFUSAL,
                "reviewed_by": "fixture-reviewer",
            }
        ],
    )
    assert unresolved == []
    assert resolved[record_id] == LabelResolution(
        behavior_label=REFUSAL,
        resolution="reviewed_override",
        per_judge_behavior_label={M1: REFUSAL, M2: COMPLIANCE},
        reviewed_by="fixture-reviewer",
    )


def test_paired_sidecar_is_read_directly_by_gate_evaluator_and_omits_bodies():
    raw = [_paired_row(condition="clean")]
    bound = bind_judge_identities(expand_paired_rows(raw), _cfg())
    checkpoint = _checkpoint_from_bound(bound)
    resolutions, unresolved = resolve_checkpoint_labels(checkpoint)
    assert unresolved == []
    sidecar = build_paired_label_sidecar(raw, bound, resolutions)
    assert sidecar[0]["undefended_behavior_label"] == REFUSAL
    assert sidecar[0]["defended_behavior_label"] == REFUSAL
    assert "reference_text" not in sidecar[0]
    assert "undefended_output" not in sidecar[0]
    observations = load_paired_observations(raw, label_rows=sidecar)
    assert observations[0].undefended_label == REFUSAL
    assert observations[0].defended_label == REFUSAL


def test_aligned_sidecars_are_read_directly_by_gate_evaluator():
    paired = _paired_row(condition="pv_standard")
    undefended = [{**paired, "output": "u"}]
    defended = [{**paired, "output": "d"}]
    bound = bind_judge_identities(expand_aligned_rows(undefended, defended), _cfg())
    checkpoint = _checkpoint_from_bound(
        bound,
        undefended_label=COMPLIANCE,
        defended_label=REFUSAL,
    )
    resolutions, unresolved = resolve_checkpoint_labels(checkpoint)
    assert unresolved == []
    ulabels, dlabels = build_aligned_label_sidecars(
        undefended,
        defended,
        bound,
        resolutions,
    )
    observations = load_aligned_observations(
        undefended,
        defended,
        undefended_label_rows=ulabels,
        defended_label_rows=dlabels,
    )
    assert observations[0].undefended_label == COMPLIANCE
    assert observations[0].defended_label == REFUSAL


def test_cli_fixture_path_never_calls_api_or_logs_prompt_bodies(tmp_path, monkeypatch, capsys):
    paired = tmp_path / "paired.jsonl"
    paired.write_text(json.dumps(_paired_row()) + "\n")
    checkpoint = tmp_path / "checkpoint.jsonl"
    labels = tmp_path / "labels.jsonl"
    unresolved = tmp_path / "unresolved.jsonl"
    judge_cfg = _cfg()
    monkeypatch.setattr(
        SCRIPT,
        "load_experiment_config",
        lambda *_args, **_kwargs: SimpleNamespace(conversion_gap=SimpleNamespace(judge=judge_cfg)),
    )

    def fixture_checkpoint(bound, _cfg_value, **_kwargs):
        return _checkpoint_from_bound(bound)

    monkeypatch.setattr(SCRIPT, "run_judge_checkpoint", fixture_checkpoint)
    SCRIPT.main(
        [
            "--judge-config",
            "fixture.yaml",
            "--paired",
            str(paired),
            "--checkpoint",
            str(checkpoint),
            "--paired-labels-out",
            str(labels),
            "--unresolved-out",
            str(unresolved),
            "--no-progress",
        ]
    )
    output = capsys.readouterr().out
    assert "secret request body" not in output
    assert "secret undefended response" not in output
    assert "secret defended response" not in output
    assert load_paired_observations([_paired_row()], label_rows=[json.loads(labels.read_text())])


def test_cli_invalidates_stale_sidecar_when_consensus_is_unresolved(tmp_path, monkeypatch):
    paired = tmp_path / "paired.jsonl"
    paired.write_text(json.dumps(_paired_row()) + "\n")
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"stale": true}\n')
    unresolved = tmp_path / "unresolved.jsonl"
    judge_cfg = _cfg()
    monkeypatch.setattr(
        SCRIPT,
        "load_experiment_config",
        lambda *_args, **_kwargs: SimpleNamespace(conversion_gap=SimpleNamespace(judge=judge_cfg)),
    )

    def fixture_checkpoint(bound, _cfg_value, **_kwargs):
        rows = _checkpoint_from_bound(bound)
        rows[0]["judge_verdicts"][M2] = _verdict(COMPLIANCE)
        return rows

    monkeypatch.setattr(SCRIPT, "run_judge_checkpoint", fixture_checkpoint)
    with pytest.raises(SystemExit, match="disagreements remain unresolved"):
        SCRIPT.main(
            [
                "--judge-config",
                "fixture.yaml",
                "--paired",
                str(paired),
                "--checkpoint",
                str(tmp_path / "checkpoint.jsonl"),
                "--paired-labels-out",
                str(labels),
                "--unresolved-out",
                str(unresolved),
                "--no-progress",
            ]
        )
    assert labels.read_text() == ""
    report = json.loads(unresolved.read_text())
    assert "reference_text" not in report
    assert "output" not in report


def test_alignment_key_prefers_record_id_then_index_then_metadata():
    row = _paired_row()
    assert alignment_key(row).startswith("record_id:")
    without_record = {key: value for key, value in row.items() if key != "record_id"}
    assert alignment_key(without_record) == "index:17"
    metadata = {key: value for key, value in without_record.items() if key != "index"}
    assert alignment_key(metadata).startswith("metadata:q0|harmful|pv_standard|3|")
