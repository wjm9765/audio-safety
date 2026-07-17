from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script(filename: str, module_name: str):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load_script("build_sarsteer_defense.py", "_test_build_sarsteer_defense")
APPLY = _load_script("apply_sarsteer_defense.py", "_test_apply_sarsteer_defense")
ROOT = Path(__file__).parents[1]


def _refusal(item_id: str, *, source: str = "advbench") -> dict[str, object]:
    return {
        "item_id": item_id,
        "reference_text": f"harmful text {item_id}",
        "source": source,
    }


def _benign(item_id: str, path: str, *, source: str = "librispeech") -> dict[str, object]:
    return {"item_id": item_id, "path": path, "source": source}


def test_explicit_calibration_exact_counts_and_path_resolution(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    relative = data_dir / "relative.flac"
    absolute = tmp_path / "absolute.flac"
    extra = data_dir / "extra.flac"
    for path in (relative, absolute, extra):
        path.write_bytes(b"audio")

    selection = BUILD.select_explicit_calibration(
        [_refusal("r0"), _refusal("r1"), _refusal("unused")],
        [
            _benign("b0", "relative.flac"),
            _benign("b1", str(absolute)),
            _benign("unused", "extra.flac"),
        ],
        data_dir=data_dir,
        n_refusal=2,
        n_benign=2,
    )

    assert selection.harmful_texts == ["harmful text r0", "harmful text r1"]
    assert selection.benign_paths == [relative.resolve(), absolute.resolve()]
    assert selection.refusal_item_ids == {"r0", "r1"}
    assert selection.benign_item_ids == {"b0", "b1"}


def test_explicit_calibration_rejects_shortfall_prompt_alias_and_duplicates(tmp_path):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    with pytest.raises(SystemExit, match="requested 2.*only 1"):
        BUILD.select_explicit_calibration(
            [_refusal("r0")],
            [_benign("b0", str(audio)), _benign("b1", str(audio) + "-missing")],
            data_dir=tmp_path,
            n_refusal=2,
            n_benign=2,
        )
    with pytest.raises(SystemExit, match="reference_text"):
        BUILD.select_explicit_calibration(
            [{"item_id": "r0", "prompt": "must not be accepted"}],
            [_benign("b0", str(audio))],
            data_dir=tmp_path,
            n_refusal=1,
            n_benign=1,
        )
    with pytest.raises(SystemExit, match="duplicate refusal item_id"):
        BUILD.select_explicit_calibration(
            [_refusal("same"), _refusal("same")],
            [_benign("b0", str(audio))],
            data_dir=tmp_path,
            n_refusal=1,
            n_benign=1,
        )


def test_eval_overlap_fails_by_item_path_or_text_but_not_source(tmp_path):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    selection = BUILD.select_explicit_calibration(
        [_refusal("r0")],
        [_benign("b0", str(audio))],
        data_dir=tmp_path,
        n_refusal=1,
        n_benign=1,
    )
    def check(rows):
        return BUILD.validate_eval_disjoint(selection, rows, data_dir=tmp_path)

    # SARSteer itself calibrates and evaluates on the same corpus (§3.2), so a
    # shared source with disjoint items must PASS.
    check([{"item_id": "eval0", "source": "figstep"}, {"item_id": "eval0", "source": "figstep"}])
    check([{"item_id": "eval0", "source": "librispeech"}])

    with pytest.raises(SystemExit, match="item_id=.*r0"):
        check([{"item_id": "r0", "source": "figstep"}])
    with pytest.raises(SystemExit, match="path="):
        check([{"item_id": "eval0", "source": "figstep", "path": str(audio)}])


def test_eval_overlap_catches_rekeyed_item_by_reference_text(tmp_path):
    # An eval row re-keyed under a fresh item_id still leaks if it is the same
    # question, so identity is also checked on normalized reference_text.
    harmful = tmp_path / "h0.wav"
    harmful.write_bytes(b"h")
    benign = tmp_path / "b0.wav"
    benign.write_bytes(b"b")
    selection = BUILD.select_paper_audio_calibration(
        [{"item_id": "c0", "path": str(harmful), "reference_text": "How do I pick a lock?"}],
        [{"item_id": "c1", "path": str(benign), "reference_text": "How do locksmiths work?"}],
        data_dir=tmp_path,
        n_refusal=1,
        n_benign=1,
    )
    with pytest.raises(SystemExit, match="reference_text="):
        BUILD.validate_eval_disjoint(
            selection,
            [{"item_id": "fresh", "reference_text": "  how do i PICK a lock?  "}],
            data_dir=tmp_path,
        )


def _eval_row(item_id: str, path: str, *, sign: float) -> dict[str, object]:
    return {
        "item_id": item_id,
        "source": "figstep",
        "safety_label": "harmful",
        "condition": "pv_standard",
        "style": f"pv_standard_{sign}",
        "sign": sign,
        "path": path,
        "reference_text": f"reference {item_id}",
    }


def test_prepare_eval_rows_supports_absolute_relative_and_rejects_duplicate(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    relative = data_dir / "relative.wav"
    absolute = tmp_path / "absolute.wav"
    relative.write_bytes(b"wav")
    absolute.write_bytes(b"wav")
    rows = [
        _eval_row("a", "relative.wav", sign=-3),
        _eval_row("a", str(absolute), sign=3),
    ]

    prepared = APPLY.prepare_eval_rows(rows, data_dir=data_dir)
    assert [row.audio_path for row in prepared] == [relative.resolve(), absolute.resolve()]
    assert prepared[0].key == ("a", "harmful", "pv_standard", "-3")
    assert prepared[1].key == ("a", "harmful", "pv_standard", "3")
    legacy_pair = [
        {**rows[0], "sign": None, "safety_label": "harmful"},
        {**rows[0], "sign": None, "safety_label": "benign"},
    ]
    legacy_prepared = APPLY.prepare_eval_rows(legacy_pair, data_dir=data_dir)
    assert len({row.key for row in legacy_prepared}) == 2
    with pytest.raises(SystemExit, match="duplicate input row key"):
        APPLY.prepare_eval_rows([rows[0], dict(rows[0])], data_dir=data_dir)


def test_completed_output_validation_rejects_incomplete_and_foreign_keys(tmp_path):
    output = tmp_path / "out.jsonl"
    key = ("a", "harmful", "pv_standard", "-3")
    incomplete = _eval_row("a", "unused.wav", sign=-3)
    output.write_text(json.dumps(incomplete) + "\n")
    with pytest.raises(SystemExit, match="undefended_output"):
        APPLY.load_completed_rows(output, valid_keys={key})

    foreign = _eval_row("other", "unused.wav", sign=-3)
    foreign.update(undefended_output="u", defended_output="d")
    output.write_text(json.dumps(foreign) + "\n")
    with pytest.raises(SystemExit, match="absent from the current input"):
        APPLY.load_completed_rows(output, valid_keys={key})


def test_main_flushes_each_pair_then_resumes_and_overwrites(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        (data_dir / name).write_bytes(b"wav")
    manifest = data_dir / "fresh.jsonl"
    rows = [
        _eval_row("a", "a.wav", sign=-3),
        _eval_row("b", "b.wav", sign=3),
        _eval_row("c", "c.wav", sign=-3),
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    vectors = tmp_path / "vectors.npz"
    vectors.write_bytes(b"stub")
    output = tmp_path / "paired.jsonl"

    monkeypatch.setattr(APPLY, "load_sarsteer_vectors", lambda _: {0: object()})
    monkeypatch.setattr(
        APPLY, "load_sarsteer_metadata", lambda _: {"implementation": "paper_faithful"}
    )
    import audio_safety.models.qwen2_audio as qwen

    monkeypatch.setattr(qwen, "load_qwen2_audio", lambda *_args, **_kwargs: ("m", "p"))
    monkeypatch.setattr(
        qwen,
        "generate_audio_response",
        lambda _m, _p, path, *_args, **_kwargs: f"undefended:{path.name}",
    )
    calls = {"defended": 0}

    def fail_second(_m, _p, path, *_args, **_kwargs):
        calls["defended"] += 1
        if calls["defended"] == 2:
            raise RuntimeError("simulated interruption")
        return f"defended:{path.name}"

    monkeypatch.setattr(APPLY, "generate_audio_response_with_sarsteer", fail_second)
    argv = [
        "--config",
        str(ROOT / "configs/experiments/run9_defense_gate.yaml"),
        "--run-name",
        "test",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--manifest",
        str(manifest),
        "--vectors",
        str(vectors),
        "--output",
        str(output),
    ]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        APPLY.main(argv)

    first_pass = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(first_pass) == 1
    assert first_pass[0]["undefended_output"] == "undefended:a.wav"
    assert first_pass[0]["defended_output"] == "defended:a.wav"

    monkeypatch.setattr(
        APPLY,
        "generate_audio_response_with_sarsteer",
        lambda _m, _p, path, *_args, **_kwargs: f"resumed:{path.name}",
    )
    APPLY.main([*argv, "--limit", "1"])
    resumed = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(resumed) == 2
    assert resumed[0]["defended_output"] == "defended:a.wav"
    assert resumed[1]["defended_output"] == "resumed:b.wav"
    APPLY.main([*argv, "--limit", "1"])
    resumed = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(resumed) == 3
    assert resumed[2]["defended_output"] == "resumed:c.wav"
    assert len({row["record_id"] for row in resumed}) == 3

    monkeypatch.setattr(
        APPLY,
        "generate_audio_response_with_sarsteer",
        lambda _m, _p, path, *_args, **_kwargs: f"overwrite:{path.name}",
    )
    APPLY.main([*argv, "--overwrite"])
    overwritten = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(overwritten) == 3
    assert all(row["defended_output"].startswith("overwrite:") for row in overwritten)


def test_paper_audio_calibration_reads_refusal_paths_and_allows_paired_ids(tmp_path):
    harmful = tmp_path / "harm.wav"
    benign = tmp_path / "safe.wav"
    harmful.write_bytes(b"harm")
    benign.write_bytes(b"safe")
    selection = BUILD.select_paper_audio_calibration(
        [{"item_id": "paired", "path": str(harmful), "source": "figstep_harm_train"}],
        [{"item_id": "paired", "path": str(benign), "source": "figstep_safe_train"}],
        data_dir=tmp_path,
        n_refusal=1,
        n_benign=1,
    )

    assert selection.harmful_texts == []
    assert selection.harmful_paths == [harmful.resolve()]
    assert selection.benign_paths == [benign.resolve()]
    assert selection.refusal_item_ids == {"paired"}
    assert selection.benign_item_ids == {"paired"}

    with pytest.raises(SystemExit, match="requires a non-empty string field 'path'"):
        BUILD.select_paper_audio_calibration(
            [{"item_id": "r0", "reference_text": "text is not an audio input"}],
            [{"item_id": "b0", "path": str(benign)}],
            data_dir=tmp_path,
            n_refusal=1,
            n_benign=1,
        )


def test_paper_audio_calibration_rejects_duplicate_or_shared_paths(tmp_path):
    audio = tmp_path / "same.wav"
    other = tmp_path / "other.wav"
    audio.write_bytes(b"audio")
    other.write_bytes(b"audio")
    with pytest.raises(SystemExit, match="duplicate paper refusal audio path"):
        BUILD.select_paper_audio_calibration(
            [
                {"item_id": "r0", "path": str(audio)},
                {"item_id": "r1", "path": str(audio)},
            ],
            [{"item_id": "b0", "path": str(other)}],
            data_dir=tmp_path,
            n_refusal=2,
            n_benign=1,
        )
    with pytest.raises(SystemExit, match="audio path overlap"):
        BUILD.select_paper_audio_calibration(
            [{"item_id": "r0", "path": str(audio)}],
            [{"item_id": "b0", "path": str(audio)}],
            data_dir=tmp_path,
            n_refusal=1,
            n_benign=1,
        )


def test_undefended_cache_rejects_rerendered_audio(tmp_path):
    # The undefended cache is reused across the alpha sweep and the held-out gate.
    # RowKey excludes the audio path, so a re-render must be caught here or the
    # cached (old-audio) undefended text would be paired with new-audio defended.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "new.wav").write_bytes(b"new-audio")
    prepared = APPLY.prepare_eval_rows([_eval_row("a", "new.wav", sign=3)], data_dir=data_dir)

    cache_path = tmp_path / "baseline.jsonl"
    stale = dict(_eval_row("a", "old.wav", sign=3))
    stale.update({"undefended_output": "cached", "sarsteer_implementation": "paper_faithful"})
    cache_path.write_text(json.dumps(stale) + "\n", "utf-8")

    with pytest.raises(SystemExit, match="re-rendered"):
        APPLY.load_undefended_cache(
            cache_path,
            prepared=prepared,
            implementation="paper_faithful",
            data_dir=data_dir,
        )


def test_undefended_cache_accepts_matching_audio_and_impl(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.wav").write_bytes(b"audio")
    prepared = APPLY.prepare_eval_rows([_eval_row("a", "a.wav", sign=3)], data_dir=data_dir)

    cache_path = tmp_path / "baseline.jsonl"
    row = dict(_eval_row("a", "a.wav", sign=3))
    row.update({"undefended_output": "ok", "sarsteer_implementation": "paper_faithful"})
    cache_path.write_text(json.dumps(row) + "\n", "utf-8")

    cache = APPLY.load_undefended_cache(
        cache_path, prepared=prepared, implementation="paper_faithful", data_dir=data_dir
    )
    assert cache[prepared[0].key] == "ok"

    with pytest.raises(SystemExit, match="implementation"):
        APPLY.load_undefended_cache(
            cache_path, prepared=prepared, implementation="legacy_reconstruction", data_dir=data_dir
        )
