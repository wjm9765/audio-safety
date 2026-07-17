#!/usr/bin/env bash
# Reproducible isolated setup for ALMGuard (NeurIPS 2025, arXiv:2510.26096).
#
# ALMGuard pins torch==2.2.2 / transformers==4.46.3, which CANNOT coexist with our
# uv env (torch 2.9.1 / transformers>=4.48). Per AGENTS.md this is a sanctioned
# isolated-venv exception (same pattern as scripts/cosyvoice2_tts.py): ALMGuard runs
# in its OWN venv and we interface via files/subprocess (scripts/almguard/run_almguard.py).
#
# GPU box only. Idempotent-ish: re-running re-clones only if the dir is absent.
set -euo pipefail

# --- config (override via env) ------------------------------------------------
ALMGUARD_ROOT="${ALMGUARD_ROOT:-/workspace/almguard}"
ALMGUARD_REPO="${ALMGUARD_REPO:-https://github.com/WeifeiJin/ALMGuard.git}"
# Pin a commit for reproducibility. Set ALMGUARD_COMMIT before running; the setup
# refuses to proceed on an unpinned clone so the defense is a fixed artifact.
ALMGUARD_COMMIT="${ALMGUARD_COMMIT:-}"
PY="${ALMGUARD_PYTHON:-python3.11}"
# torch 2.2.2 cu121 wheels run fine on a CUDA 12.8 driver (backward compatible).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if [[ -z "${ALMGUARD_COMMIT}" ]]; then
  echo "ERROR: set ALMGUARD_COMMIT=<sha> to pin the defense artifact (reproducibility)." >&2
  echo "  find it with: git ls-remote ${ALMGUARD_REPO} HEAD" >&2
  exit 2
fi

mkdir -p "${ALMGUARD_ROOT}"
REPO_DIR="${ALMGUARD_ROOT}/ALMGuard"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[almguard] cloning ${ALMGUARD_REPO} -> ${REPO_DIR}"
  git clone "${ALMGUARD_REPO}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --all --tags
git -C "${REPO_DIR}" checkout "${ALMGUARD_COMMIT}"
echo "[almguard] checked out $(git -C "${REPO_DIR}" rev-parse --short HEAD)"

VENV="${ALMGUARD_ROOT}/venv"
if [[ ! -d "${VENV}" ]]; then
  echo "[almguard] creating venv ${VENV} (${PY})"
  "${PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip

# Install the torch 2.2.2 CUDA 12.1 stack FIRST (explicit index), then the rest.
echo "[almguard] installing torch 2.2.2 cu121 stack"
pip install --index-url "${TORCH_INDEX}" torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
echo "[almguard] installing ALMGuard requirements.txt"
pip install -r "${REPO_DIR}/requirements.txt"

# Whisper large-v3 is required by main.py (ASR term of M-GSM) at ./models/large-v3.pt.
echo "[almguard] NOTE: place Whisper large-v3 at ${REPO_DIR}/models/large-v3.pt (see ALMGuard README)."
echo "[almguard] NOTE: shipped M-GSM mask expected at ${REPO_DIR}/mask/global_saliency.npz — verify it is the Qwen2-Audio mask (k=48); else recompute."

# Smoke test: import-only is INSUFFICIENT (Codex 2026-07-17). Verify a real
# forward+backward and an 8-bit optimizer step actually run on this GPU/driver.
echo "[almguard] smoke test (forward+backward + bitsandbytes 8-bit step)"
python - <<'PY'
import torch, bitsandbytes as bnb
assert torch.cuda.is_available(), "CUDA not visible to the ALMGuard venv"
x = torch.randn(64, 64, device="cuda", requires_grad=True)
w = torch.randn(64, 64, device="cuda", requires_grad=True)
loss = (x @ w).pow(2).mean()
loss.backward()
opt = bnb.optim.Adam8bit([w], lr=1e-3)
opt.step()
print("[almguard] smoke OK | torch", torch.__version__, "| cuda", torch.version.cuda)
PY

echo "[almguard] setup complete. Env: ${VENV} | Repo: ${REPO_DIR} @ ${ALMGUARD_COMMIT}"
echo "[almguard] drive it with scripts/almguard/run_almguard.py (our env), which shells into this venv."
