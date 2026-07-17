from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
PATCH = ROOT / "scripts" / "almguard" / "patches" / "a40_bf16_cached_mask.patch"
SETUP = ROOT / "scripts" / "almguard" / "setup_almguard_env.sh"


def test_a40_patch_is_parseable_and_preserves_fp32_optimizer_master():
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


def test_setup_reapplies_patch_idempotently_before_compile_check():
    text = SETUP.read_text(encoding="utf-8")
    reverse_check = text.index('apply --reverse --check "${A40_PATCH}"')
    apply_check = text.index('apply --check "${A40_PATCH}"')
    compile_check = text.index('-m py_compile "${MAIN_PY}" "${EVAL_QWEN_PY}"')
    assert reverse_check < apply_check < compile_check
