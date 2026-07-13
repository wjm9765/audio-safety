import pytest

from audio_safety.evaluation.attack_flip import harmful_specific_interaction
from audio_safety.evaluation.causal_trace import (
    assert_unique_trace_ids,
    identity_invariance,
    make_trace_id,
    plan_primary_conditions,
    primary_contrast,
    reproduced_flip_accounting,
    summarize,
)

MODEL = "J"
CLEAN, ATTACK = "neutral", "jb_pap"
P2 = "first_generation_prelogit"


def _v(ordinal, is_refusal, *, degenerate=False, actionable=False):
    return {
        "compliance_ordinal": ordinal,
        "actionable_harmful_help": actionable,
        "is_refusal": is_refusal,
        "degenerate": degenerate,
        "rationale": "",
    }


COMPLY = _v("full_comply", False)   # harmful: attack success / benign: answered
REFUSE = _v("full_refusal", True)   # genuine refusal on either side


def _row(item, modality, style, safety, verdict):
    return {
        "item_id": item,
        "modality": modality,
        "style": style,
        "safety_label": safety,
        "judge_verdicts": {MODEL: verdict},
    }


def _eight(item, *, ah_clean, ah_att, th_clean, th_att, ab_clean, ab_att, tb_clean, tb_att):
    return [
        _row(item, "audio", CLEAN, "harmful", ah_clean),
        _row(item, "audio", ATTACK, "harmful", ah_att),
        _row(item, "text", CLEAN, "harmful", th_clean),
        _row(item, "text", ATTACK, "harmful", th_att),
        _row(item, "audio", CLEAN, "benign", ab_clean),
        _row(item, "audio", ATTACK, "benign", ab_att),
        _row(item, "text", CLEAN, "benign", tb_clean),
        _row(item, "text", ATTACK, "benign", tb_att),
    ]


# --- tau (unconditional harmful-specific audio x attack interaction) -----------


def test_tau_cancels_to_zero_for_generic_compliance():
    # Attack flips refusal->comply EQUALLY in audio and text (harmful), benign flat.
    rows = _eight(
        "a",
        ah_clean=REFUSE, ah_att=COMPLY, th_clean=REFUSE, th_att=COMPLY,
        ab_clean=COMPLY, ab_att=COMPLY, tb_clean=COMPLY, tb_att=COMPLY,
    )
    res = harmful_specific_interaction(
        rows, model=MODEL, clean_style=CLEAN, attack_style=ATTACK, n_bootstrap=200
    )
    assert res["insufficient"] is False
    assert res["n_complete"] == 1
    assert res["tau_pp"] == pytest.approx(0.0, abs=1e-9)


def test_tau_positive_for_audio_specific_harmful_effect():
    # Attack flips harmful audio but NOT harmful text; benign unaffected -> tau > 0.
    rows = _eight(
        "a",
        ah_clean=REFUSE, ah_att=COMPLY, th_clean=REFUSE, th_att=REFUSE,
        ab_clean=COMPLY, ab_att=COMPLY, tb_clean=COMPLY, tb_att=COMPLY,
    )
    res = harmful_specific_interaction(
        rows, model=MODEL, clean_style=CLEAN, attack_style=ATTACK, n_bootstrap=200
    )
    assert res["tau_pp"] == pytest.approx(100.0)


def test_tau_excludes_items_missing_a_cell():
    rows = _eight(
        "a",
        ah_clean=REFUSE, ah_att=COMPLY, th_clean=REFUSE, th_att=REFUSE,
        ab_clean=COMPLY, ab_att=COMPLY, tb_clean=COMPLY, tb_att=COMPLY,
    )
    rows = [
        r for r in rows
        if not (r["modality"] == "text" and r["style"] == ATTACK and r["safety_label"] == "benign")
    ]
    res = harmful_specific_interaction(rows, model=MODEL, clean_style=CLEAN, attack_style=ATTACK)
    assert res["insufficient"] is True  # the only item is now incomplete


# --- trace planning / ids ------------------------------------------------------


def test_make_trace_id_distinguishes_conditions():
    kw = dict(recipient_item="x", position=P2, seed=0)
    a = make_trace_id(condition="same_item", donor_item="x", layer=16, **kw)
    b = make_trace_id(condition="wrong_item", donor_item="y", layer=16, **kw)
    c = make_trace_id(condition="same_item", donor_item="x", layer=12, **kw)
    assert len({a, b, c}) == 3


def test_plan_primary_conditions_unique_and_complete():
    specs = plan_primary_conditions(
        flip_item="x", wrong_item="y", benign_item="b", layer=16, position=P2, seed=0
    )
    assert_unique_trace_ids(specs)
    conds = {s["condition"] for s in specs}
    expected = {"no_patch", "identity", "same_item", "wrong_item", "reverse", "benign_same_item"}
    assert expected <= conds


def test_plan_skips_absent_wrong_and_benign():
    specs = plan_primary_conditions(
        flip_item="x", wrong_item=None, benign_item=None, layer=16, position=P2, seed=0
    )
    conds = {s["condition"] for s in specs}
    assert "wrong_item" not in conds
    assert not any(c.startswith("benign") for c in conds)


def test_assert_unique_trace_ids_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate trace_id"):
        assert_unique_trace_ids([{"trace_id": "a"}, {"trace_id": "a"}])


# --- aggregation ---------------------------------------------------------------


def _trace(cond, item, safety, verdict, *, output="out"):
    return {
        "condition": cond,
        "layer": 16,
        "position": P2,
        "recipient_item": item,
        "recipient_safety": safety,
        "output": output,
        "judge_verdicts": {MODEL: verdict},
    }


def test_summarize_reports_alive_signal():
    records = [
        _trace("no_patch", "x1", "harmful", COMPLY, output="np-x1"),
        _trace("no_patch", "x2", "harmful", COMPLY, output="np-x2"),
        _trace("identity", "x1", "harmful", COMPLY, output="np-x1"),
        _trace("identity", "x2", "harmful", COMPLY, output="np-x2"),
        _trace("same_item", "x1", "harmful", REFUSE),
        _trace("same_item", "x2", "harmful", REFUSE),
        _trace("wrong_item", "x1", "harmful", COMPLY),
        _trace("wrong_item", "x2", "harmful", COMPLY),
        _trace("benign_no_patch", "b1", "benign", COMPLY),
        _trace("benign_same_item", "b1", "benign", COMPLY),
    ]
    s = summarize(
        records, judge_models=[MODEL], primary_layer=16, primary_position=P2,
        original_flip_items=["x1", "x2"],
    )
    assert s["identity_invariance"]["invariant"] is True
    blk = s["per_judge"][MODEL]
    assert blk["reproduced_flips"]["n_reproduced_flips"] == 2
    pc = blk["primary_contrast"]
    assert pc["harmful_rescue"] == pytest.approx(1.0)
    assert pc["benign_overrefusal"] == pytest.approx(0.0)
    assert pc["contrast_C"] == pytest.approx(1.0)
    assert pc["wrong_item_refusal"] == pytest.approx(0.0)


def test_identity_mismatch_is_flagged():
    records = [
        _trace("no_patch", "x1", "harmful", COMPLY, output="A"),
        _trace("identity", "x1", "harmful", COMPLY, output="B"),
    ]
    inv = identity_invariance(records, layer=16, position=P2)
    assert inv["invariant"] is False
    assert inv["n_mismatch"] == 1


def test_primary_contrast_none_when_benign_absent():
    rates = {"same_item": {"refusal_rate": 0.5}, "no_patch": {"refusal_rate": 0.0}}
    pc = primary_contrast(rates)
    assert pc["harmful_rescue"] == pytest.approx(0.5)
    assert pc["benign_overrefusal"] is None
    assert pc["contrast_C"] is None


def test_reproduced_flip_accounting_counts_only_still_complying():
    records = [
        _trace("no_patch", "x1", "harmful", COMPLY),
        _trace("no_patch", "x2", "harmful", REFUSE),  # no longer reproduces the flip
    ]
    acc = reproduced_flip_accounting(
        records, model=MODEL, layer=16, position=P2, original_flip_items=["x1", "x2", "x3"]
    )
    assert acc["n_original_flips"] == 3
    assert acc["n_no_patch_scored"] == 2
    assert acc["n_reproduced_flips"] == 1
