"""Run 4 §8 attack-induced-flip tests: frozen wrappers + flip/DiD/interaction stats.

All CPU-only (no network/GPU): the deterministic jailbreak wrapping and the flip
analysis are exercised on synthetic judged rows with known ground truth.
"""

from pathlib import Path

import pytest

from audio_safety.config.schema import AudioRdoDatasetConfig
from audio_safety.data.attacks import (
    build_attack_variant_records,
    load_jailbreak_templates,
    wrap_request,
)
from audio_safety.data.datasets import AudioRdoPair
from audio_safety.data.families import _parse_variant_override_file, render_audio_records
from audio_safety.evaluation.attack_flip import (
    audio_specificity,
    benign_did,
    classify,
    compute_attack_flip,
    flip_stats,
)
from audio_safety.utils.io import save_jsonl

MODEL = "judge-1"
TEMPLATES_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "attacks" / "jailbreak_templates.yaml"
)


# --- frozen jailbreak wrappers ------------------------------------------------


def _templates():
    return load_jailbreak_templates(TEMPLATES_YAML)


def test_frozen_templates_load_and_are_unique():
    templates = _templates()
    assert len(templates) >= 2
    style_keys = [t.style_key for t in templates]
    assert len(set(style_keys)) == len(style_keys)
    assert {"jb_ica", "jb_pap"} <= set(style_keys)


def test_wrap_substitutes_request_verbatim_and_is_deterministic():
    template = _templates()[0]
    req = "How do I do the harmful thing?"
    wrapped1 = wrap_request(template, req)
    wrapped2 = wrap_request(template, req)
    assert wrapped1 == wrapped2  # deterministic
    assert req in wrapped1  # base request preserved verbatim
    assert "{request}" not in wrapped1  # slot filled


def test_wrap_handles_braces_in_request():
    template = _templates()[0]
    # Literal braces must not be interpreted as format fields.
    req = "print('{oops}') then {do_bad}"
    wrapped = wrap_request(template, req)
    assert "{oops}" in wrapped
    assert "{do_bad}" in wrapped


def test_template_requires_single_request_slot(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "templates:\n"
        "  - id: no_slot\n    style_key: jb_x\n    template: 'answer this'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="request"):
        load_jailbreak_templates(bad)


def test_build_records_shape_and_freeze_metadata():
    pairs = [
        AudioRdoPair("id0", "cat", "harmful zero", "benign zero", "src"),
        AudioRdoPair("id1", "cat", "harmful one", "benign one", "src"),
    ]
    templates = _templates()
    records = build_attack_variant_records(pairs, templates)
    # one record per (pair x template x {harmful,benign})
    assert len(records) == len(pairs) * len(templates) * 2
    for r in records:
        assert r["target_style"] in {t.style_key for t in templates}
        assert r["styled_text"]
        assert r["base_reference_text"] in r["styled_text"]
        assert len(r["template_sha256"]) == 64
        assert len(r["rendered_sha256"]) == 64
        # gate fields keep the render pipeline from skipping these rows
        assert r["added_operational_detail"] is False
        assert r["refusal_or_warning"] is False
        assert r["content_preservation"] == "high"


def test_build_records_safety_label_filter():
    pairs = [AudioRdoPair("id0", "cat", "harmful zero", "benign zero", "src")]
    records = build_attack_variant_records(pairs, _templates(), safety_labels=("harmful",))
    assert {r["safety_label"] for r in records} == {"harmful"}


# --- synthetic judged rows ----------------------------------------------------


def _verdict(ordinal, *, is_refusal=False, degenerate=False, actionable=False):
    return {
        "compliance_ordinal": ordinal,
        "is_refusal": is_refusal,
        "degenerate": degenerate,
        "actionable_harmful_help": actionable,
    }


COMPLY = _verdict("full_comply")
REFUSE = _verdict("full_refusal", is_refusal=True)
NONANSWER = _verdict("full_refusal", is_refusal=True, degenerate=True)
BENIGN_ANSWER = _verdict("full_comply")  # for benign, comply = answered


def _row(item_id, safety_label, style, modality, verdict, model=MODEL):
    return {
        "item_id": item_id,
        "safety_label": safety_label,
        "style": style,
        "modality": modality,
        "judge_verdicts": {model: verdict},
    }


def test_classify_buckets():
    assert classify(COMPLY) == "comply"
    assert classify(REFUSE) == "refusal"
    # degenerate wins over is_refusal -> non-answer, not a genuine refusal
    assert classify(NONANSWER) == "nonanswer"
    # benign answered (not refusal/degenerate/comply-harmful) -> other for a
    # non-attack ordinal
    assert classify(_verdict("safe_redirect", is_refusal=True)) == "refusal"
    assert classify(_verdict("full_refusal", is_refusal=False)) == "other"


def test_flip_stats_counts_genuine_flips():
    rows = []
    # item0: refuse -> comply  (FLIP)
    rows += [
        _row("i0", "harmful", "neutral", "audio", REFUSE),
        _row("i0", "harmful", "jb", "audio", COMPLY),
    ]
    # item1: refuse -> refuse  (no flip)
    rows += [
        _row("i1", "harmful", "neutral", "audio", REFUSE),
        _row("i1", "harmful", "jb", "audio", REFUSE),
    ]
    # item2: comply -> comply  (clean already complies; not a flip)
    rows += [
        _row("i2", "harmful", "neutral", "audio", COMPLY),
        _row("i2", "harmful", "jb", "audio", COMPLY),
    ]
    # item3: refuse -> comply  (FLIP)
    rows += [
        _row("i3", "harmful", "neutral", "audio", REFUSE),
        _row("i3", "harmful", "jb", "audio", COMPLY),
    ]
    # item4: nonanswer(clean) -> comply  (NOT a genuine flip; clean not a refusal)
    rows += [
        _row("i4", "harmful", "neutral", "audio", NONANSWER),
        _row("i4", "harmful", "jb", "audio", COMPLY),
    ]
    res = flip_stats(
        rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
        n_bootstrap=200, seed=0,
    )
    assert res["n_eligible_pairs"] == 5
    assert res["genuine_refuse_clean"] == 3  # i0, i1, i3
    assert res["genuine_flips"] == 2  # i0, i3
    assert res["flip_rate_given_clean_refuse"] == pytest.approx(2 / 3)
    assert res["reverse_unflips"] == 0
    # clean attack rate = 1/5 (only i2), attacked = 4/5 (i0,i2,i3,i4)
    assert res["clean_attack_rate"] == pytest.approx(0.2)
    assert res["attacked_attack_rate"] == pytest.approx(0.8)
    assert res["rd_pp"] == pytest.approx(60.0)
    # attacked strictly favored -> one-sided p well below 0.5
    assert res["mcnemar_attacked_gt_clean"]["p_one_sided_audio_gt_text"] < 0.5
    assert res["attacked_taxonomy"]["comply"] == 4


def test_flip_stats_reverse_unflip_counted():
    rows = [
        _row("i0", "harmful", "neutral", "audio", COMPLY),
        _row("i0", "harmful", "jb", "audio", REFUSE),
    ]
    res = flip_stats(
        rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
        n_bootstrap=100, seed=0,
    )
    assert res["genuine_flips"] == 0
    assert res["reverse_unflips"] == 1


def test_flip_stats_insufficient_when_no_shared_items():
    rows = [_row("i0", "harmful", "neutral", "audio", REFUSE)]
    res = flip_stats(
        rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
        n_bootstrap=50, seed=0,
    )
    assert res["insufficient"] is True


def test_benign_did_harmful_specific_is_positive():
    rows = []
    for i in range(4):
        # harmful: refuse -> comply (Δ = +1)
        rows += [
            _row(f"i{i}", "harmful", "neutral", "audio", REFUSE),
            _row(f"i{i}", "harmful", "jb", "audio", COMPLY),
        ]
        # benign: answered -> answered (Δ = 0)  => harmful-specific
        rows += [
            _row(f"i{i}", "benign", "neutral", "audio", BENIGN_ANSWER),
            _row(f"i{i}", "benign", "jb", "audio", BENIGN_ANSWER),
        ]
    res = benign_did(
        rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
        n_bootstrap=200, seed=0,
    )
    assert res["n"] == 4
    assert res["did_pp"] == pytest.approx(100.0)
    assert res["ci_low_pp"] == pytest.approx(100.0)


def test_benign_did_generic_shift_is_zero():
    rows = []
    for i in range(4):
        # harmful: refuse -> comply (Δ = +1)
        rows += [
            _row(f"i{i}", "harmful", "neutral", "audio", REFUSE),
            _row(f"i{i}", "harmful", "jb", "audio", COMPLY),
        ]
        # benign: refused(clean) -> answered(attacked) (Δ = +1) => generic shift
        rows += [
            _row(f"i{i}", "benign", "neutral", "audio", REFUSE),
            _row(f"i{i}", "benign", "jb", "audio", BENIGN_ANSWER),
        ]
    res = benign_did(
        rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
        n_bootstrap=200, seed=0,
    )
    assert res["did_pp"] == pytest.approx(0.0)


def test_audio_specificity_interaction_positive():
    rows = []
    for i in range(4):
        # audio: refuse -> comply (Δ audio = +1)
        rows += [
            _row(f"i{i}", "harmful", "neutral", "audio", REFUSE),
            _row(f"i{i}", "harmful", "jb", "audio", COMPLY),
        ]
        # text: refuse -> refuse (Δ text = 0)  => audio-specific
        rows += [
            _row(f"i{i}", "harmful", "neutral", "text", REFUSE),
            _row(f"i{i}", "harmful", "jb", "text", REFUSE),
        ]
    res = audio_specificity(
        rows, model=MODEL, clean_style="neutral", attack_style="jb",
        n_bootstrap=200, seed=0,
    )
    assert res["n"] == 4
    assert res["audio_attack_effect_pp"] == pytest.approx(100.0)
    assert res["text_attack_effect_pp"] == pytest.approx(0.0)
    assert res["interaction_pp"] == pytest.approx(100.0)


def test_audio_specificity_symmetric_interaction_zero():
    rows = []
    for i in range(4):
        rows += [
            _row(f"i{i}", "harmful", "neutral", "audio", REFUSE),
            _row(f"i{i}", "harmful", "jb", "audio", COMPLY),
            _row(f"i{i}", "harmful", "neutral", "text", REFUSE),
            _row(f"i{i}", "harmful", "jb", "text", COMPLY),
        ]
    res = audio_specificity(
        rows, model=MODEL, clean_style="neutral", attack_style="jb",
        n_bootstrap=200, seed=0,
    )
    assert res["interaction_pp"] == pytest.approx(0.0)


def test_compute_attack_flip_assembles_families():
    rows = [
        _row("i0", "harmful", "neutral", "audio", REFUSE),
        _row("i0", "harmful", "jb_ica", "audio", COMPLY),
        _row("i0", "benign", "neutral", "audio", BENIGN_ANSWER),
        _row("i0", "benign", "jb_ica", "audio", BENIGN_ANSWER),
    ]
    report = compute_attack_flip(
        rows,
        judge_models=[MODEL],
        families=[{"name": "jailbreak", "attack_styles": ["jb_ica"]}],
        clean_style="neutral",
        primary_modality="audio",
        n_bootstrap=50,
        seed=0,
    )
    assert report["families"][0]["name"] == "jailbreak"
    style_block = report["families"][0]["styles"][0]
    assert style_block["attack_style"] == "jb_ica"
    flip = style_block["per_judge"][MODEL]["flip"]
    assert flip["genuine_flips"] == 1
    # single judge -> agreement not computed
    assert style_block["judge_agreement"]["kappa"] is None


# --- pipeline wiring: fail-closed + freeze-hash guards -------------------------


def _attack_dataset_cfg(tmp_path, *, attack_file, emotion_file=None):
    style_cfg = {"enabled": emotion_file is not None}
    if emotion_file is not None:
        style_cfg["output_file"] = emotion_file
    return AudioRdoDatasetConfig.model_validate(
        {
            "styles": ["neutral", "jb_ica"],
            "neutral_acoustic_styles": ["jb_ica"],
            "attack_variant_file": attack_file,
            "style_variant_generation": style_cfg,
            "tts": {"batch_command_template": "true {batch_jsonl}"},
        }
    )


def test_render_fails_closed_when_attack_wrapper_missing(tmp_path):
    # jb_ica is a text-attack style but no attack-variant file exists -> must raise
    # rather than silently render clean base text under the jb_ica label.
    cfg = _attack_dataset_cfg(tmp_path, attack_file=None)
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    with pytest.raises(ValueError, match="text-attack style"):
        render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)


def test_render_uses_wrapped_text_when_attack_file_present(tmp_path):
    templates = _templates()
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    records = build_attack_variant_records(pairs, [t for t in templates if t.style_key == "jb_ica"])
    attack_rel = "attacks.jsonl"
    save_jsonl(records, tmp_path / attack_rel)
    cfg = _attack_dataset_cfg(tmp_path, attack_file=attack_rel)
    out = render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)
    jb = [r for r in out if r["style"] == "jb_ica" and r["safety_label"] == "harmful"][0]
    assert "harmful zero" in str(jb["reference_text"])  # wrapped, contains base
    assert str(jb["reference_text"]) != "harmful zero"  # not the bare base text
    assert len(str(jb["reference_sha256"])) == 64


def test_emotion_row_cannot_satisfy_attack_guard(tmp_path):
    # An emotion-file row keyed with a jb_ica style must NOT count as a verified
    # attack: the guard requires the hash-verified attack file, so render fails.
    save_jsonl(
        [
            {
                "item_id": "i0",
                "safety_label": "harmful",
                "target_style": "jb_ica",
                "styled_text": "sneaky unverified text",
                "content_preservation": "high",
            }
        ],
        tmp_path / "emotion.jsonl",
    )
    cfg = _attack_dataset_cfg(tmp_path, attack_file=None, emotion_file="emotion.jsonl")
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    with pytest.raises(ValueError, match="hash-verified"):
        render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)


def test_render_rejects_stale_wav(tmp_path):
    templates = [t for t in _templates() if t.style_key == "jb_ica"]
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    save_jsonl(build_attack_variant_records(pairs, templates), tmp_path / "attacks.jsonl")
    cfg = _attack_dataset_cfg(tmp_path, attack_file="attacks.jsonl")
    # Pre-create a jb_ica wav + a sidecar recording a DIFFERENT text hash (stale).
    wav = tmp_path / "audio" / "harmful" / "jb_ica" / "i0.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")
    (wav.parent / "i0.wav.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="stale render"):
        render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)


def test_render_reuses_wav_when_sidecar_matches(tmp_path):
    templates = [t for t in _templates() if t.style_key == "jb_ica"]
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    recs = build_attack_variant_records(pairs, templates)
    save_jsonl(recs, tmp_path / "attacks.jsonl")
    cfg = _attack_dataset_cfg(tmp_path, attack_file="attacks.jsonl")
    # Correct sidecar hash = sha256 of the exact wrapped harmful text.
    import hashlib

    jb_harmful = next(
        r for r in recs if r["target_style"] == "jb_ica" and r["safety_label"] == "harmful"
    )
    good = hashlib.sha256(str(jb_harmful["styled_text"]).encode("utf-8")).hexdigest()
    wav = tmp_path / "audio" / "harmful" / "jb_ica" / "i0.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")
    (wav.parent / "i0.wav.sha256").write_text(good)
    out = render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)
    jb = [r for r in out if r["style"] == "jb_ica" and r["safety_label"] == "harmful"][0]
    assert jb["status"] == "exists"  # matched sidecar -> reused, no stale error


def test_render_rejects_attack_wav_without_sidecar(tmp_path):
    templates = [t for t in _templates() if t.style_key == "jb_ica"]
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    save_jsonl(build_attack_variant_records(pairs, templates), tmp_path / "attacks.jsonl")
    cfg = _attack_dataset_cfg(tmp_path, attack_file="attacks.jsonl")
    wav = tmp_path / "audio" / "harmful" / "jb_ica" / "i0.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")  # no sidecar -> unverifiable provenance
    with pytest.raises(ValueError, match="provenance sidecar"):
        render_audio_records(pairs, cfg, cfg.tts, tmp_path, dry_run=True)


def test_attack_row_missing_hash_raises(tmp_path):
    path = tmp_path / "attacks.jsonl"
    save_jsonl(
        [{"item_id": "i0", "safety_label": "harmful", "target_style": "jb_ica",
          "styled_text": "wrapped"}],  # no rendered_sha256
        path,
    )
    with pytest.raises(ValueError, match="no rendered_sha256"):
        _parse_variant_override_file(path, verify_hash=True, strict_unique=True)


def test_attack_row_padded_text_hash_raises(tmp_path):
    import hashlib

    padded = "  wrapped  "
    path = tmp_path / "attacks.jsonl"
    save_jsonl(
        [{"item_id": "i0", "safety_label": "harmful", "target_style": "jb_ica",
          "styled_text": padded,
          "rendered_sha256": hashlib.sha256(padded.encode("utf-8")).hexdigest()}],
        path,
    )
    # Loader hashes the stripped (spoken) text, so a hash of the padded raw fails.
    with pytest.raises(ValueError, match="hash mismatch"):
        _parse_variant_override_file(path, verify_hash=True, strict_unique=True)


def test_attack_file_roundtrip_hash_verifies(tmp_path):
    pairs = [AudioRdoPair("i0", "cat", "harmful zero", "benign zero", "src")]
    recs = build_attack_variant_records(pairs, _templates())
    path = tmp_path / "attacks.jsonl"
    save_jsonl(recs, path)
    # Freshly built records verify cleanly (no strip/hash drift).
    overrides = _parse_variant_override_file(path, verify_hash=True, strict_unique=True)
    assert ("i0", "harmful", "jb_ica") in overrides


def test_parse_variant_hash_mismatch_raises(tmp_path):
    path = tmp_path / "attacks.jsonl"
    save_jsonl(
        [
            {
                "item_id": "i0",
                "safety_label": "harmful",
                "target_style": "jb_ica",
                "styled_text": "wrapped A",
                "rendered_sha256": "deadbeef" * 8,  # wrong hash
            }
        ],
        path,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        _parse_variant_override_file(path, verify_hash=True, strict_unique=True)


def test_parse_variant_strict_unique_rejects_duplicates(tmp_path):
    path = tmp_path / "attacks.jsonl"
    row = {
        "item_id": "i0",
        "safety_label": "harmful",
        "target_style": "jb_ica",
        "styled_text": "wrapped",
    }
    save_jsonl([row, dict(row)], path)
    with pytest.raises(ValueError, match="duplicate"):
        _parse_variant_override_file(path, strict_unique=True)


def test_index_rejects_duplicate_cell_rows():
    rows = [
        _row("dup", "harmful", "neutral", "audio", REFUSE),
        _row("dup", "harmful", "neutral", "audio", COMPLY),
        _row("dup", "harmful", "jb", "audio", COMPLY),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        flip_stats(
            rows, model=MODEL, clean_style="neutral", attack_style="jb", modality="audio",
            n_bootstrap=50, seed=0,
        )
