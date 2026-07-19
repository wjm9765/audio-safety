"""CPU unit tests for the Run 10 channel-patching orchestration helpers (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from audio_safety.pipelines.channel_patching import (
    assert_pair_alignment,
    delta_heard,
    forced_choice_margin,
    freeze_tau,
    recognized_both_mask,
    refusal_margin,
)

# Layout: [sys, AUDIO, AUDIO, instr, gen] with audio_token_id = 7, t_ab = 4.
AID = 7
_CLEAN = np.array([1, 7, 7, 3, 9])
_ATTACK = np.array([1, 7, 7, 3, 9])
_MASK = np.array([1, 1, 1, 1, 1])


def test_assert_pair_alignment_happy_path():
    positions, t_ab = assert_pair_alignment(
        _CLEAN, _MASK, _ATTACK, _MASK, audio_token_id=AID, clean_t_ab=4, attack_t_ab=4
    )
    assert positions == [1, 2]
    assert t_ab == 4


def test_assert_pair_alignment_rejects_span_mismatch():
    attack = np.array([1, 7, 3, 3, 9])  # only one audio token now
    with pytest.raises(ValueError, match="audio-token positions differ"):
        assert_pair_alignment(
            _CLEAN, _MASK, attack, _MASK, audio_token_id=AID, clean_t_ab=4, attack_t_ab=4
        )


def test_assert_pair_alignment_rejects_prompt_and_tab_mismatch():
    attack = np.array([1, 7, 7, 99, 9])  # non-audio (instruction) token differs
    with pytest.raises(ValueError, match="non-audio tokens differ"):
        assert_pair_alignment(
            _CLEAN, _MASK, attack, _MASK, audio_token_id=AID, clean_t_ab=4, attack_t_ab=4
        )
    with pytest.raises(ValueError, match="t_AB differs"):
        assert_pair_alignment(
            _CLEAN, _MASK, _ATTACK, _MASK, audio_token_id=AID, clean_t_ab=4, attack_t_ab=3
        )


def test_assert_pair_alignment_rejects_span_after_tab_and_length_mismatch():
    # t_ab inside the audio span is illegal (audio must precede the readout)
    with pytest.raises(ValueError, match="audio span must precede"):
        assert_pair_alignment(
            _CLEAN, _MASK, _ATTACK, _MASK, audio_token_id=AID, clean_t_ab=2, attack_t_ab=2
        )
    with pytest.raises(ValueError, match="input_ids length differ"):
        assert_pair_alignment(
            _CLEAN, _MASK, np.array([1, 7, 7, 3]), np.array([1, 1, 1, 1]),
            audio_token_id=AID, clean_t_ab=4, attack_t_ab=4,
        )


def test_refusal_margin_and_guards():
    logits = np.array([0.0, 2.0, 1.0, -1.0, 0.5])
    m = refusal_margin(logits, refusal_ids=[1], compliance_ids=[3])
    assert m == pytest.approx(2.0 - (-1.0))
    with pytest.raises(ValueError, match="non-empty"):
        refusal_margin(logits, refusal_ids=[], compliance_ids=[3])
    with pytest.raises(ValueError, match="disjoint"):
        refusal_margin(logits, refusal_ids=[1, 3], compliance_ids=[3])


def test_forced_choice_margin():
    logits = np.array([0.0, 5.0, 1.0, 3.0])  # options map to letter token ids
    # options A,B,C -> token ids 1,2,3 ; correct = A (index 0), logit 5 highest
    h, pred = forced_choice_margin(logits, option_token_ids=[1, 2, 3], correct_index=0)
    assert pred == 0
    assert h == pytest.approx(5.0 - 3.0)  # correct minus best distractor
    # a misheard case: correct is C (option idx 2, logit 3) but A (logit 5) wins -> H<0
    h2, pred2 = forced_choice_margin(logits, option_token_ids=[1, 2, 3], correct_index=2)
    assert pred2 == 0  # option A (token id 1 -> logit 5.0) is the argmax
    assert h2 < 0
    with pytest.raises(ValueError, match="distinct"):
        forced_choice_margin(logits, option_token_ids=[1, 1, 3], correct_index=0)


def test_recognized_both_mask_and_delta_heard():
    h_clean = np.array([2.0, 1.0, 3.0, -0.5])
    h_attack = np.array([1.5, -1.0, 2.0, 0.7])
    mask = recognized_both_mask(h_clean, h_attack, tau=0.0)
    # pair 1: both >0 True; pair 2: attack<0 False; pair 3: both>0 True; pair 4: clean<0 False
    assert mask.tolist() == [True, False, True, False]

    m_clean = np.array([3.0, 3.0, 2.0, 2.0])
    m_attack = np.array([1.0, 9.0, 0.0, 9.0])
    out = delta_heard(m_clean, m_attack, mask)
    assert out["n"] == 2
    # only recognized pairs 1 and 3 count: mean([1-3, 0-2]) = mean([-2, -2]) = -2
    assert out["delta_heard"] == pytest.approx(-2.0)
    # empty mask -> nan, n=0
    empty = delta_heard(m_clean, m_attack, np.zeros(4, bool))
    assert empty["n"] == 0 and np.isnan(empty["delta_heard"])


def test_freeze_tau_uses_clean_dev_quantile_only():
    h_clean_dev = np.arange(0.0, 100.0)  # 0..99
    # want 90% of clean-dev to pass -> tau = 10th percentile
    tau = freeze_tau(h_clean_dev, recognized_fraction=0.9)
    assert tau == pytest.approx(np.quantile(h_clean_dev, 0.1))
    frac_pass = float((h_clean_dev > tau).mean())
    assert frac_pass == pytest.approx(0.9, abs=0.02)
    with pytest.raises(ValueError, match="recognized_fraction"):
        freeze_tau(h_clean_dev, recognized_fraction=1.5)
