from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "prepare_run9_sarsteer_adapted_gate.py"
    )
    spec = importlib.util.spec_from_file_location("_test_sarsteer_adapted_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_script()


def _touch(data_dir: Path, relative: str) -> str:
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())
    return relative


def _row(
    data_dir: Path,
    item_id: str,
    role: str,
    condition: str,
    *,
    category: int | None = None,
    sign: int | None = None,
    reference_text: str | None = None,
    eligible: bool = True,
) -> dict[str, object]:
    suffix = "none" if sign is None else str(sign)
    relative = f"fixture/{role}/{condition}/{suffix}/{item_id}.wav"
    return {
        "item_id": item_id,
        "source": (
            "librispeech_test_clean" if role == "utility_eval" else "figstep_safebench"
        ),
        "category_id": category,
        "category_name": f"category-{category}",
        "safety_label": (
            "harmful" if role in {"harmful_eval", "positive_control_eval"} else "benign"
        ),
        "gate_role": role,
        "style": condition,
        "condition": condition,
        "sign": sign,
        "path": _touch(data_dir, relative),
        "reference_text": reference_text or f"{role} request for {item_id}",
        "evaluation_eligible": eligible,
        "record_id": f"record-{role}-{item_id}-{condition}-{suffix}",
    }


def _triplet(
    data_dir: Path, item_id: str, category: int, *, eligible: bool = True
) -> list[dict[str, object]]:
    text = f"harmful request for {item_id}"
    return [
        _row(
            data_dir,
            item_id,
            "harmful_eval",
            "clean",
            category=category,
            reference_text=text,
            eligible=eligible,
        ),
        _row(
            data_dir,
            item_id,
            "harmful_eval",
            "pv_standard",
            category=category,
            sign=-3,
            reference_text=text,
            eligible=eligible,
        ),
        _row(
            data_dir,
            item_id,
            "harmful_eval",
            "pv_standard",
            category=category,
            sign=3,
            reference_text=text,
            eligible=eligible,
        ),
    ]


def _pc_category(item_id: str) -> int:
    number = int(item_id.rsplit("_", 1)[1])
    return number // 50 + 1


def _fixture(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    known_by_category: dict[int, list[str]] = defaultdict(list)
    for item_id, _sign_value in GATE.KNOWN_DEV_VULNERABLE_ATTACKS:
        category = _pc_category(item_id)
        if item_id not in known_by_category[category]:
            known_by_category[category].append(item_id)

    core: list[dict[str, object]] = []
    core_harmful_ids: list[str] = []
    for category in GATE.HARMFUL_CATEGORIES:
        item_ids = list(known_by_category[category])
        item_ids.extend(
            f"core_h_{category}_{index:02d}"
            for index in range(10 - len(item_ids))
        )
        assert len(item_ids) == 10
        core_harmful_ids.extend(item_ids)
        for item_id in item_ids:
            core.extend(_triplet(data_dir, item_id, category))

    for category in GATE.SOFT_CATEGORIES:
        core.extend(
            _row(
                data_dir,
                f"core_soft_{category}_{index:02d}",
                "soft_overrefusal",
                "clean",
                category=category,
            )
            for index in range(10)
        )
    core.extend(
        _row(
            data_dir,
            f"core_utility_{index:02d}",
            "utility_eval",
            "clean",
        )
        for index in range(30)
    )
    core.extend(
        _row(
            data_dir,
            f"core_adv_pc_{index:02d}",
            "positive_control_eval",
            "positive_control",
        )
        for index in range(30)
    )
    assert len(core) == 300

    full = list(core)
    final_harmful_ids: dict[int, list[str]] = defaultdict(list)
    for category in GATE.HARMFUL_CATEGORIES:
        for index in range(12):
            item_id = f"final_h_{category}_{index:02d}"
            final_harmful_ids[category].append(item_id)
            full.extend(_triplet(data_dir, item_id, category))
    for category in GATE.SOFT_CATEGORIES:
        full.extend(
            _row(
                data_dir,
                f"final_soft_{category}_{index:02d}",
                "soft_overrefusal",
                "clean",
                category=category,
            )
            for index in range(12)
        )
    full.extend(
        _row(
            data_dir,
            f"final_utility_{index:02d}",
            "utility_eval",
            "clean",
        )
        for index in range(35)
    )

    # Fixed PAP IDs need metadata, but they are not candidates for the paired
    # final harmful cell because they intentionally lack complete PV variants.
    for item_id in GATE.FINAL_PAP_POSITIVE_CONTROL_IDS:
        full.append(
            _row(
                data_dir,
                item_id,
                "harmful_eval",
                "clean",
                category=_pc_category(item_id),
                eligible=False,
            )
        )

    pap = []
    pap_ids = set(core_harmful_ids) | set(GATE.FINAL_PAP_POSITIVE_CONTROL_IDS)
    for item_id in sorted(pap_ids):
        relative = f"audio_attack_flip/harmful/jb_pap/{item_id}.wav"
        pap.append(
            {
                "item_id": item_id,
                "style": "jb_pap",
                "safety_label": "harmful",
                "text": f"PAP prefixed request for {item_id}",
                "output_path": str((data_dir / _touch(data_dir, relative)).resolve()),
            }
        )

    vector = []
    for index, role in enumerate(("sarsteer_refusal_calib", "sarsteer_pca")):
        item_id = f"vector_{index}"
        vector.append(
            {
                "item_id": item_id,
                "path": _touch(data_dir, f"vector/{item_id}.wav"),
                "reference_text": f"vector calibration text {index}",
                "gate_role": role,
            }
        )
    return data_dir, full, core, pap, vector, final_harmful_ids


def test_builds_deterministic_core_dev_and_disjoint_heldout_final(tmp_path: Path):
    data_dir, full, core, pap, vector, final_harmful_ids = _fixture(tmp_path)
    first = GATE.build_adapted_gate_plan(full, core, pap, vector, data_dir=data_dir)
    second = GATE.build_adapted_gate_plan(full, core, pap, vector, data_dir=data_dir)

    assert first.dev_rows == second.dev_rows
    assert first.final_rows == second.final_rows
    assert len(first.dev_rows) == 76
    assert len(first.final_rows) == 300
    assert Counter(row["gate_role"] for row in first.dev_rows) == GATE.DEV_ROLE_COUNTS
    assert Counter(row["gate_role"] for row in first.final_rows) == GATE.FINAL_ROLE_COUNTS
    assert Counter(row["condition"] for row in first.final_rows) == {
        "clean": 130,
        "pv_standard": 140,
        "positive_control": 30,
    }
    assert all(row["alpha_selection_only"] is True for row in first.dev_rows)
    assert all(row["final_gate_eligible"] is False for row in first.dev_rows)
    assert all(row["gate_partition"] == "sarsteer_alpha_dev" for row in first.dev_rows)
    assert all(row["alpha_selection_only"] is False for row in first.final_rows)
    assert all(row["final_gate_eligible"] is True for row in first.final_rows)
    assert all(row["gate_partition"] == "sarsteer_final_gate" for row in first.final_rows)

    selected_final_harmful = {
        row["item_id"]
        for row in first.final_rows
        if row["gate_role"] == "harmful_eval"
    }
    assert selected_final_harmful == {
        item_id
        for category in GATE.HARMFUL_CATEGORIES
        for item_id in final_harmful_ids[category][:10]
    }
    core_ids = {row["item_id"] for row in core}
    final_ids = {row["item_id"] for row in first.final_rows}
    vector_ids = {row["item_id"] for row in vector}
    assert not (core_ids & final_ids)
    assert not (vector_ids & final_ids)
    assert first.summary["selection_used_model_outcomes"] is False
    assert first.summary["historical_openrouter_labels_read"] is False
    assert first.summary["leakage"]["all_checks_passed"] is True


def test_dev_contains_exact_frozen_s_cells_and_outcome_agnostic_pap_controls(tmp_path: Path):
    data_dir, full, core, pap, vector, _final_harmful_ids = _fixture(tmp_path)
    plan = GATE.build_adapted_gate_plan(full, core, pap, vector, data_dir=data_dir)

    known_clean = [
        row for row in plan.dev_rows if row.get("adapted_gate_component") == "known_s_clean"
    ]
    known_attacks = [
        row for row in plan.dev_rows if row.get("adapted_gate_component") == "known_s_attack"
    ]
    assert len(known_clean) == 6
    assert len(known_attacks) == 10
    assert {
        (row["item_id"], int(row["sign"])) for row in known_attacks
    } == set(GATE.KNOWN_DEV_VULNERABLE_ATTACKS)

    controls = [
        row for row in plan.dev_rows if row["gate_role"] == "positive_control_eval"
    ]
    assert len(controls) == 30
    assert all(row["condition"] == "positive_control" for row in controls)
    assert all(row["attack_family"] == "jb_pap" for row in controls)
    assert all(row["historical_outcomes_used_for_selection"] is False for row in controls)
    assert not (
        {row["item_id"] for row in controls}
        & {item_id for item_id, _sign_value in GATE.KNOWN_DEV_VULNERABLE_ATTACKS}
    )
    final_controls = [
        row for row in plan.final_rows if row["gate_role"] == "positive_control_eval"
    ]
    assert [row["item_id"] for row in final_controls] == list(
        GATE.FINAL_PAP_POSITIVE_CONTROL_IDS
    )


def test_fails_closed_when_frozen_pap_asset_is_missing(tmp_path: Path):
    data_dir, full, core, pap, vector, _final_harmful_ids = _fixture(tmp_path)
    missing_id = GATE.FINAL_PAP_POSITIVE_CONTROL_IDS[-1]
    pap = [row for row in pap if row["item_id"] != missing_id]
    with pytest.raises(ValueError, match="frozen final PAP IDs missing"):
        GATE.build_adapted_gate_plan(full, core, pap, vector, data_dir=data_dir)


def test_fails_closed_on_final_vector_reference_text_leakage(tmp_path: Path):
    data_dir, full, core, pap, vector, final_harmful_ids = _fixture(tmp_path)
    leaked_id = final_harmful_ids[1][0]
    leaked_text = next(
        row["reference_text"]
        for row in full
        if row["item_id"] == leaked_id and row["condition"] == "clean"
    )
    vector[0] = {**vector[0], "reference_text": leaked_text}
    with pytest.raises(ValueError, match="held-out final/vector calibration leakage"):
        GATE.build_adapted_gate_plan(full, core, pap, vector, data_dir=data_dir)
