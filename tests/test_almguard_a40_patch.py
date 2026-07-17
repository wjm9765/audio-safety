from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
# The single consolidated Run 9 patch that setup applies (supersedes the piecemeal
# argparse/--prefix/bf16 edits; the older a40_bf16_cached_mask.patch is kept for
# provenance only). It captures the full upstream->current transformation.
PATCH = ROOT / "scripts" / "almguard" / "patches" / "run9_almguard_runtime.patch"
SETUP = ROOT / "scripts" / "almguard" / "setup_almguard_env.sh"


def test_runtime_patch_is_parseable_and_preserves_fp32_and_bf16_plumbing():
    text = PATCH.read_text(encoding="utf-8")
    parsed = subprocess.run(
        ["git", "apply", "--numstat", str(PATCH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert "main.py" in parsed.stdout
    assert "eval_qwen.py" in parsed.stdout
    # A40 device/dtype plumbing (fidelity-neutral) is preserved.
    assert "torch_dtype=torch.bfloat16" in text
    assert 'attn_implementation="sdpa"' in text
    assert "model.eval()" in text
    assert "model.requires_grad_(False)" in text
    assert "if os.path.exists(saliency_cache):" in text
    assert "asr_model = None" in text
    assert "dtype=torch.float32" in text
    assert text.count("perturb.to(dtype=audio_feat.dtype)") == 2
    assert "device=model.device, dtype=model.dtype" in text
    assert "gradient_checkpointing" not in text


def test_runtime_patch_adds_budget_resume_watchdog_and_optional_sroa():
    text = PATCH.read_text(encoding="utf-8")
    # Budget knobs + resume + per-audio watchdog (schedule only; SAP objective unchanged).
    # (--num_epochs is a pre-existing upstream arg, so it is not part of this diff.)
    assert "--max_iter" in text
    assert "--max_seconds" in text
    assert "--no_resume" in text
    assert "_latest_resume_checkpoint" in text
    assert "watchdog" in text
    assert "resume=not args.no_resume" in text
    # HIGH-finding fix: resume is bound to the ordered audio set + reduced schedule.
    assert "sap_run_config" in text
    assert "audioset_sha256" in text
    # Upstream SorryBench SRoA scorer made optional so a missing judge never blocks generation.
    assert "score_sroa" in text
    # The SAP objective knobs must NOT be net-removed by this patch. A pure
    # reformat (e.g. adding a trailing comma to `tau=0.5`) shows the term on both a
    # removed and an added line; a real deletion would remove it without re-adding.
    lines = text.splitlines()
    for frozen in ("tau=0.5", "k=48", "lr=3e-4"):
        removed = sum(
            1 for ln in lines if ln.startswith("-") and not ln.startswith("---") and frozen in ln
        )
        added = sum(
            1 for ln in lines if ln.startswith("+") and not ln.startswith("+++") and frozen in ln
        )
        assert added >= removed, (
            f"patch net-removes frozen objective term {frozen!r} (added {added} < removed {removed})"
        )


def test_setup_reapplies_runtime_patch_idempotently_before_compile_check():
    text = SETUP.read_text(encoding="utf-8")
    reverse_check = text.index('apply --reverse --check "${RUNTIME_PATCH}"')
    apply_check = text.index('apply --check "${RUNTIME_PATCH}"')
    compile_check = text.index('-m py_compile "${MAIN_PY}" "${EVAL_QWEN_PY}"')
    assert reverse_check < apply_check < compile_check
