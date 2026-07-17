import hashlib

from audio_safety.data.run9_fresh import build_run9_fresh_plan, write_job_shards


def _row(category_id: int, question: str) -> dict[str, str]:
    return {
        "category_id": str(category_id),
        "category_name": f"category-{category_id}",
        "question": question,
    }


def test_plan_reuses_only_hash_verified_harmful_audio(tmp_path):
    data_dir = tmp_path / "data"
    reuse = data_dir / "old"
    reuse.mkdir(parents=True)
    wav = reuse / "figstep_safebench_0000.wav"
    wav.write_bytes(b"wav")
    digest = hashlib.sha256(b"harmful one").hexdigest()
    wav.with_suffix(".wav.sha256").write_text(digest)

    plan = build_run9_fresh_plan(
        [_row(1, "harmful one"), _row(2, "harmful two"), _row(8, "soft control")],
        data_dir=data_dir,
        output_audio_root=data_dir / "new",
        reuse_harmful_root=reuse,
    )

    assert len(plan.manifest_rows) == 3
    assert plan.reused_count == 1
    assert plan.pending_count == 2
    assert len(plan.tts_jobs) == 2
    assert plan.manifest_rows[0]["render_status"] == "reused_hash_verified"
    assert plan.manifest_rows[2]["safety_label"] == "benign"
    assert plan.manifest_rows[2]["gate_role"] == "soft_overrefusal"
    assert plan.manifest_rows[2]["asr_required"] is True


def test_job_shards_are_complete_and_balanced(tmp_path):
    jobs = [{"item_id": str(index)} for index in range(8)]
    paths = write_job_shards(jobs, tmp_path / "jobs.jsonl", workers=3)

    lines = [path.read_text().splitlines() for path in paths]
    assert sorted(map(len, lines)) == [2, 3, 3]
    assert sum(map(len, lines)) == len(jobs)
