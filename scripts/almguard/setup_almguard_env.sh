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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --- config (override via env) ------------------------------------------------
ALMGUARD_ROOT="${ALMGUARD_ROOT:-/workspace/almguard}"
ALMGUARD_REPO="${ALMGUARD_REPO:-https://github.com/WeifeiJin/ALMGuard.git}"
# Pin a commit for reproducibility. Set ALMGUARD_COMMIT before running; the setup
# refuses to proceed on an unpinned clone so the defense is a fixed artifact.
ALMGUARD_COMMIT="${ALMGUARD_COMMIT:-}"
PY="${ALMGUARD_PYTHON:-python3.11}"
# torch 2.2.2 cu121 wheels run fine on a CUDA 12.8 driver (backward compatible).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# Upstream pins mkl-service==2.4.0, which has no distribution for Python 3.11
# on PyPI. 2.4.1 is the nearest published patch release and changes packaging,
# not ALMGuard's model, optimization, or evaluation algorithm.
MKL_SERVICE_VERSION="${ALMGUARD_MKL_SERVICE_VERSION:-2.4.1}"

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

# All Run 9 fidelity-neutral adaptations to the pinned upstream artifact are captured
# in ONE reproducible patch (supersedes the earlier piecemeal argparse/--prefix/bf16
# edits, whose hunks it now contains). Scope:
#   (1) CLI defects: import argparse; declare the defaulted filename-only --prefix.
#   (2) A40 runtime: explicit BF16+SDPA; no unused Whisper resident when the shipped
#       saliency cache is present.
#   (3) Budget/schedule knobs: --max_iter and --num_epochs pass-through, universal-SAP
#       resume from per-step checkpoints, and a per-audio wall-clock watchdog. The SAP
#       OBJECTIVE is unchanged — mask k=48, tau=0.5, lr=3e-4, unified safe target, one
#       universal Mel perturbation. Only the training schedule/budget can be reduced.
#   (4) The upstream SorryBench SRoA scorer is made optional (--score_sroa) so a missing
#       judge model never blocks generation; the Run 9 gate scores responses itself.
# The older patches/a40_bf16_cached_mask.patch is retained for provenance only.
MAIN_PY="${REPO_DIR}/main.py"
EVAL_QWEN_PY="${REPO_DIR}/eval_qwen.py"
RUNTIME_PATCH="${SCRIPT_DIR}/patches/run9_almguard_runtime.patch"
if git -C "${REPO_DIR}" apply --reverse --check "${RUNTIME_PATCH}" >/dev/null 2>&1; then
  echo "[almguard] Run 9 runtime patch already applied"
elif git -C "${REPO_DIR}" apply --check "${RUNTIME_PATCH}"; then
  git -C "${REPO_DIR}" apply "${RUNTIME_PATCH}"
  echo "[almguard] applied Run 9 runtime patch (CLI + BF16/cached-mask + budget/resume/watchdog + optional SRoA)"
else
  echo "ERROR: Run 9 runtime patch does not apply cleanly to pinned ALMGuard" >&2
  exit 3
fi
"${PY}" -m py_compile "${MAIN_PY}" "${EVAL_QWEN_PY}"

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
# Keep the pinned upstream requirements immutable. Generate a recorded local
# compatibility copy with only the unavailable Python-3.11 mkl-service pin
# replaced by its nearest available patch release.
FILTERED_REQUIREMENTS="${ALMGUARD_ROOT}/requirements.py311.txt"
awk -v replacement="mkl-service==${MKL_SERVICE_VERSION}" \
  '{ if ($0 ~ /^mkl-service==/) print replacement; else print $0 }' \
  "${REPO_DIR}/requirements.txt" > "${FILTERED_REQUIREMENTS}"
echo "[almguard] installing filtered requirements: ${FILTERED_REQUIREMENTS}"
echo "[almguard] compatibility pin: $(grep '^mkl-service==' "${FILTERED_REQUIREMENTS}")"
pip install -r "${FILTERED_REQUIREMENTS}"

# Importing main.py and parsing --help catches upstream CLI defects without
# loading a model or touching any experiment artifact.
(cd "${REPO_DIR}" && python main.py --help >/dev/null)
echo "[almguard] upstream CLI smoke test passed"

# Whisper large-v3 is required by main.py (ASR term of M-GSM) at ./models/large-v3.pt.
echo "[almguard] NOTE: place Whisper large-v3 at ${REPO_DIR}/models/large-v3.pt (see ALMGuard README)."
echo "[almguard] NOTE: shipped M-GSM mask expected at ${REPO_DIR}/mask/global_saliency.npz — verify it is the Qwen2-Audio mask (k=48); else recompute."

echo "[almguard] setup complete. Env: ${VENV} | Repo: ${REPO_DIR} @ ${ALMGUARD_COMMIT}"
echo "[almguard] drive it with scripts/almguard/run_almguard.py (our env), which shells into this venv."
