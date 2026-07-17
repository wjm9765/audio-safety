#!/usr/bin/env -S uv run python
"""Render versioned Run 9 ASR-remediation candidates with CosyVoice2.

This adapter reuses ``cosyvoice2_tts.py`` for repository setup, model loading,
the neutral instruction, prompt voice, and synthesis.  Its only acoustic variant
is a deterministic per-item RNG seed supplied by the prepared retry job.  It
never overwrites a candidate: an existing candidate must have matching text-hash
and render-provenance sidecars or the run fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

RETRY_CHILD_ENV = "AUDIO_SAFETY_RUN9_TTS_RETRY_CHILD"
RETRY_CONTRACT_VERSION = "run9_clean_asr_retry_v1"


def _load_base_renderer() -> ModuleType:
    path = Path(__file__).with_name("cosyvoice2_tts.py")
    spec = importlib.util.spec_from_file_location("audio_safety_cosyvoice2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import CosyVoice wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_renderer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-jsonl", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=BASE.DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-audio", type=Path, default=None)
    parser.add_argument("--python", default=BASE.DEFAULT_TTS_PYTHON)
    # Attributes used by the shared wrapper's renderer preparation.
    parser.set_defaults(style="neutral", style_instruction=None)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _reexec_in_tts_venv(args: argparse.Namespace, venv_dir: Path) -> None:
    env = os.environ.copy()
    env[RETRY_CHILD_ENV] = "1"
    cache_root = BASE.project_cache_dir(args)
    shared_cache_root = cache_root.parent
    env.setdefault("HF_HOME", str(shared_cache_root / "huggingface"))
    env.setdefault("HF_HUB_CACHE", str(shared_cache_root / "huggingface" / "hub"))
    env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    python = BASE.venv_python(venv_dir)
    os.execvpe(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"missing/duplicate retry item_id: {item_id!r}")
        seen.add(item_id)
    return rows


def _validated_job(job: dict[str, Any]) -> dict[str, Any]:
    item_id = str(job.get("item_id") or "")
    if job.get("retry_contract_version") != RETRY_CONTRACT_VERSION:
        raise ValueError(f"retry contract mismatch for {item_id}")
    if str(job.get("style") or "") != "neutral":
        raise ValueError(f"Run 9 ASR retry must preserve neutral style for {item_id}")
    if job.get("style_instruction") is not None:
        raise ValueError(f"Run 9 ASR retry may not change style instruction for {item_id}")
    speed = float(job.get("speed", 1.0))
    if speed != 1.0:
        raise ValueError(f"Run 9 seed-only retry must preserve speed=1.0 for {item_id}")
    seed = job.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Run 9 retry seed must be a non-negative integer for {item_id}")
    text = BASE.text_from_job(job)
    expected_hash = str(job.get("reference_sha256") or "")
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"retry text hash mismatch for {item_id}")
    output = Path(str(job.get("output_path") or ""))
    provenance = Path(str(job.get("provenance_path") or ""))
    if not output.is_absolute() or not provenance.is_absolute():
        raise ValueError(f"retry output/provenance paths must be absolute for {item_id}")
    return {
        "item_id": item_id,
        "text": text,
        "reference_sha256": expected_hash,
        "seed": seed,
        "speed": speed,
        "output": output,
        "hash_sidecar": output.with_suffix(f"{output.suffix}.sha256"),
        "provenance": provenance,
        "retry_id": str(job.get("retry_id") or ""),
    }


def _provenance(
    values: dict[str, Any], *, output: Path, prompt_audio: Path, model_id: str
) -> dict[str, Any]:
    instruction = BASE.STYLE_INSTRUCTIONS["neutral"]
    return {
        "renderer": "CosyVoice2",
        "model_id": model_id,
        "item_id": values["item_id"],
        "style": "neutral",
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "prompt_audio_sha256": _file_sha256(prompt_audio),
        "reference_sha256": values["reference_sha256"],
        "output_sha256": _file_sha256(output),
        "seed": values["seed"],
        "speed": values["speed"],
        "retry_contract_version": RETRY_CONTRACT_VERSION,
        "retry_id": values["retry_id"],
    }


def _validate_existing(values: dict[str, Any], *, prompt_audio: Path, model_id: str) -> None:
    output = values["output"]
    sidecar = values["hash_sidecar"]
    provenance_path = values["provenance"]
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != values["reference_sha256"]
    ):
        raise ValueError(f"existing retry text-hash sidecar mismatch: {output}")
    if not provenance_path.is_file():
        raise ValueError(f"existing retry render provenance missing: {provenance_path}")
    actual = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = _provenance(
        values,
        output=output,
        prompt_audio=prompt_audio,
        model_id=model_id,
    )
    if actual != expected:
        raise ValueError(f"existing retry render provenance mismatch: {provenance_path}")


def render(args: argparse.Namespace) -> None:
    jobs = _load_jobs(args.batch_jsonl)
    if not jobs:
        print(json.dumps({"retry_jobs": 0, "rendered": 0, "resumed": 0}))
        return
    torch, _torchaudio, model, prompt_audio = BASE.prepare_renderer(args)
    import numpy as np
    from tqdm.auto import tqdm

    rendered = 0
    resumed = 0
    for job in tqdm(jobs, desc="Run9 TTS ASR retry", unit="wav", dynamic_ncols=True):
        values = _validated_job(job)
        output = values["output"]
        if output.exists():
            _validate_existing(values, prompt_audio=prompt_audio, model_id=args.model_id)
            resumed += 1
            continue
        # The original run did not record a per-item RNG state.  The retry does,
        # and resets it before every item so worker count/order cannot change audio.
        seed = values["seed"]
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        BASE.synthesize_to_file(
            torch=torch,
            torchaudio=_torchaudio,
            model=model,
            text=values["text"],
            style="neutral",
            style_instruction=None,
            prompt_audio=prompt_audio,
            output=output,
            announce=False,
        )
        provenance = _provenance(
            values,
            output=output,
            prompt_audio=prompt_audio,
            model_id=args.model_id,
        )
        _atomic_write(values["hash_sidecar"], values["reference_sha256"] + "\n")
        _atomic_write(
            values["provenance"],
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        )
        rendered += 1
    print(
        json.dumps(
            {"retry_jobs": len(jobs), "rendered": rendered, "resumed": resumed},
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _root, _repo, venv_dir, _model = BASE.bootstrap(args)
    if os.environ.get(RETRY_CHILD_ENV) != "1":
        _reexec_in_tts_venv(args, venv_dir)
    render(args)


if __name__ == "__main__":
    main()
