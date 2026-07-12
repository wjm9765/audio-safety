"""Run 4 Stage B mechanism-adjudication tests on synthetic activations.

CPU-only. Builds controlled residual activations where r_A = e_0, so the refusal
coordinate c_R is coordinate 0 of ``cr_hidden`` and the harmfulness signal is
coordinate 0 of ``ch_hidden``. Each mechanism (CONVERSION / PERCEPTION / DRIFT /
READOUT) is constructed and the adjudicator must recover it.
"""

import numpy as np
import pytest

from audio_safety.evaluation.conversion_probe import (
    adjudicate_conversion,
    block_writer_gap,
    cross_fit_dim,
    readout_auroc,
)

D = 64
R_A = np.eye(D)[0]
CELLS = [("text", "harmful"), ("text", "benign"), ("audio", "harmful"), ("audio", "benign")]


def _build(cr_mu, ch_mu, *, cr1_harmful=3.0, n_items=60, seed=0):
    """Synthetic activations. coord0 = r_A (refusal); coord1 = a harmfulness
    direction at P2 (harmful vs benign, no modality gap) that is ORTHOGONAL to
    r_A, so r_H@P2 does not trivially coincide with the refusal axis; coords 2..D
    are matched noise."""
    rng = np.random.default_rng(seed)
    cr_rows, ch_rows, meta = [], [], []
    for i in range(n_items):
        for mod, saf in CELLS:
            cr = rng.normal(0, 2.5, D)
            cr[0] = rng.normal(cr_mu[(mod, saf)], 1.5)
            cr[1] = rng.normal(cr1_harmful if saf == "harmful" else 0.0, 0.5)
            ch = rng.normal(0, 0.4, D)
            ch[0] = rng.normal(ch_mu[(mod, saf)], 0.5)
            cr_rows.append(cr)
            ch_rows.append(ch)
            meta.append(
                {"item_id": f"q{i}", "modality": mod, "safety_label": saf, "behavior_label": None}
            )
    cr_hidden = np.asarray(cr_rows, dtype=float)
    ch_hidden = np.asarray(ch_rows, dtype=float)
    # behavior: harmful rows split within modality by the refusal coordinate (so
    # r_A separates refusal vs compliance in-modality, satisfying the readout gate);
    # benign rows are benign answers.
    for mod in ("text", "audio"):
        sel = [
            j
            for j, m in enumerate(meta)
            if m["modality"] == mod and m["safety_label"] == "harmful"
        ]
        med = np.median(cr_hidden[sel, 0])
        for j in sel:
            meta[j]["behavior_label"] = (
                "policy_refusal" if cr_hidden[j, 0] >= med else "harmful_compliance"
            )
    for m in meta:
        if m["safety_label"] == "benign":
            m["behavior_label"] = "benign_answer"
    return cr_hidden, ch_hidden, meta


def _adj(cr, ch, meta):
    return adjudicate_conversion(
        cr, ch, meta, R_A, n_random_directions=120, n_boot=300, seed=0
    )


def test_readout_auroc_perfect():
    auroc = readout_auroc(np.array([0.1, 0.2, 0.9, 1.0]), np.array([0, 0, 1, 1]))
    assert auroc == pytest.approx(1.0)


def test_cross_fit_dim_separates_without_self_leakage():
    cr, ch, meta = _build(
        cr_mu={c: 0.0 for c in CELLS},
        ch_mu={
            ("text", "harmful"): 2, ("audio", "harmful"): 2,
            ("text", "benign"): 0, ("audio", "benign"): 0,
        },
    )
    item_ids = [m["item_id"] for m in meta]
    harmful = np.array([m["safety_label"] == "harmful" for m in meta])
    scores = cross_fit_dim(ch, harmful, item_ids, k=5, seed=0)
    assert scores[harmful].mean() > scores[~harmful].mean() + 1.0


def test_conversion_mechanism():
    # harmfulness preserved (ch harmful high both modalities); refusal under-driven
    # in audio (cr harmful: text high, audio low); benign has no modality gap.
    cr, ch, meta = _build(
        cr_mu={("text", "harmful"): 3.0, ("audio", "harmful"): -3.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
        ch_mu={("text", "harmful"): 2.0, ("audio", "harmful"): 2.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
    )
    report = _adj(cr, ch, meta)
    assert report["status"] == "CONVERSION"
    assert report["refusal_underactivation"]["d_R_sd"] > 0.3
    assert report["specificity"]["specificity_ratio"] >= 2.0


def test_perception_mechanism():
    # audio harmfulness degraded (ch harmful audio ~ benign), so d_H is large.
    cr, ch, meta = _build(
        cr_mu={("text", "harmful"): 3.0, ("audio", "harmful"): -3.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
        ch_mu={("text", "harmful"): 2.0, ("audio", "harmful"): 0.1,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
    )
    report = _adj(cr, ch, meta)
    assert report["status"] == "PERCEPTION"


def test_drift_mechanism():
    # same modality offset in harmful AND benign -> benign-centered gap cancels.
    cr, ch, meta = _build(
        cr_mu={("text", "harmful"): 3.0, ("audio", "harmful"): -3.0,
               ("text", "benign"): 3.0, ("audio", "benign"): -3.0},
        ch_mu={("text", "harmful"): 2.0, ("audio", "harmful"): 2.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
    )
    report = _adj(cr, ch, meta)
    assert report["status"] == "DRIFT"


def test_readout_mechanism():
    # harmfulness preserved and c_R NOT under-driven (audio ~ text on harmful).
    cr, ch, meta = _build(
        cr_mu={("text", "harmful"): 3.0, ("audio", "harmful"): 3.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
        ch_mu={("text", "harmful"): 2.0, ("audio", "harmful"): 2.0,
               ("text", "benign"): 0.0, ("audio", "benign"): 0.0},
    )
    report = _adj(cr, ch, meta)
    assert report["status"] == "READOUT"


def test_block_writer_telescoping():
    # c_R(l) is a cumulative sum, so per-block deltas telescope back to out(L-1).
    text = np.cumsum(np.array([[1.0, 0.5, 0.5, 2.0]]), axis=1)
    audio = np.cumsum(np.array([[1.0, 0.1, 0.1, 0.2]]), axis=1)
    input_proj = np.array([0.0])
    out = block_writer_gap(text, audio, input_proj_text=input_proj, input_proj_audio=input_proj)
    assert out["telescoping_residual"] == pytest.approx(0.0, abs=1e-9)
    # audio writes less onto r_A at the later blocks
    assert out["delta_text_minus_audio"][-1] > 0
