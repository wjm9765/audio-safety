#!/usr/bin/env bash
# Reproducible isolated uv env for MiniCPM-o-2.6 (exp2 primary model).
#
# WHY a second env (measured, not assumed): MiniCPM-o-2.6 ships trust_remote_code
# modeling written against transformers 4.44.2. Our main uv env is transformers
# >=5.12, where loading it fails at import:
#
#     ImportError: cannot import name 'WHISPER_ATTENTION_CLASSES'
#                  from 'transformers.models.whisper.modeling_whisper'
#
# A shim for that one symbol was rejected: it would silently substitute eager
# attention for the recorded `attn_implementation: sdpa` — algebraically
# equivalent but NOT bitwise equivalent in bf16 — which is unacceptable in
# paper-producing runs.
#
# WHY 4.46.3 AND NOT THE 4.44.2 THE MODEL CARD DECLARES. On 4.44.2 the load gets
# one step further and then dies:
#
#     ImportError: This modeling file requires the following packages that were
#                  not found in your environment: flash_attn
#
# That is a FALSE POSITIVE. flash_attn is referenced only by the SigLIP *vision*
# tower (modeling_navit_siglip.py), which we never initialise (init_vision=False),
# and even there the import is already guarded by `if is_flash_attn_2_available():`.
# transformers 4.44.2's `get_imports()` filters only try/except blocks, so it
# scans that guarded import statically and demands the package. The filter for
# `is_flash_attn_*_available()` blocks was added in a later 4.4x release.
#
# So the working version must satisfy BOTH:
#   * has the is_flash_attn filter in get_imports()  -> added in 4.46
#   * still exports WHISPER_ATTENTION_CLASSES        -> removed in 4.48
# 4.46.x is the intersection, and 4.46.3 is already pinned elsewhere in this repo
# (scripts/almguard/setup_almguard_env.sh). Building flash-attn from source to
# satisfy the false positive was attempted and is NOT recommended: it OOM-killed
# at MAX_JOBS=32 and would spend an hour compiling kernels for a tower we disable.
#
# Verified on this stack: loads in 89s, 14.83 GiB VRAM (audio tower only), audio
# demonstrably consumed (real/silence/mismatched wavs give different, content-
# specific outputs), greedy decoding deterministic, 1.80 s/generation @96 tokens.
#
# Per AGENTS.md this is the sanctioned isolated-venv exception, same pattern as
# scripts/cosyvoice2_tts.py and scripts/almguard/setup_almguard_env.sh: the model
# runs in its OWN uv venv and we interface via files/subprocess. The main project
# env is never mutated.
#
# GPU box only. Idempotent: re-running reuses an existing venv.
set -euo pipefail

# --- config (override via env) ------------------------------------------------
MINICPM_ENV="${MINICPM_ENV:-/workspace/audio_safety_data/envs/minicpm}"
MINICPM_CACHE="${MINICPM_CACHE:-/workspace/audio_safety_data/cache}"
TRANSFORMERS_PIN="${MINICPM_TRANSFORMERS:-4.46.3}"   # see header: 4.44 false-positives on flash_attn
TORCH_PIN="${MINICPM_TORCH:-2.9.1}"                  # matches the main env's CUDA 12.8 stack
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# The MooseFS /workspace volume returns "Disk quota exceeded (errno 122)" while uv
# extracts many small .so files. Keep the package cache on the container overlay;
# only the venv itself lives on /workspace. Reproducibility comes from this
# script, not from preserving the extracted cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv-isolated}"
mkdir -p "${UV_CACHE_DIR}"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 2; }

echo "[minicpm] env          = ${MINICPM_ENV}"
echo "[minicpm] transformers = ${TRANSFORMERS_PIN}  (NOT the card's 4.44.2 - see header)"
echo "[minicpm] torch        = ${TORCH_PIN} (${TORCH_INDEX})"
echo "[minicpm] uv cache     = ${UV_CACHE_DIR} (off the quota'd volume)"

uv venv --python 3.11 "${MINICPM_ENV}"

# torchvision is required: MiniCPM's remote code imports AutoImageProcessor even
# when init_vision=False, so omitting it fails at load rather than at use.
uv pip install --python "${MINICPM_ENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url "${TORCH_INDEX}" \
  "torch==${TORCH_PIN}" "torchaudio==${TORCH_PIN}" "torchvision" \
  "transformers==${TRANSFORMERS_PIN}" \
  "accelerate>=0.33" "librosa>=0.10" "soundfile>=0.12" "numpy<2" \
  "sentencepiece" "protobuf"

echo "[minicpm] verifying..."
"${MINICPM_ENV}/bin/python" - <<'PY'
import torch, transformers, torchvision
assert transformers.__version__.startswith("4.46"), transformers.__version__
print(f"  transformers {transformers.__version__} | torch {torch.__version__} "
      f"| torchvision {torchvision.__version__} | cuda {torch.cuda.is_available()}")
PY

echo "[minicpm] setup complete."
echo "[minicpm] weights: prefetch with huggingface_hub.snapshot_download into ${MINICPM_CACHE}"
echo "[minicpm]   (NOT scripts/download_audio_lm.py — that calls AutoModel.from_pretrained,"
echo "[minicpm]    which needs the GPU and executes remote code before the bytes are on disk)"
echo "[minicpm] drive it with scripts/exp2/run_minicpm.py, which shells into this venv."
