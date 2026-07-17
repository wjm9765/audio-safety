#!/usr/bin/env -S uv run python
"""Score Run 9 audio faithfulness with Whisper large-v3 in ALMGuard's venv.

The public process runs in this repository's environment and owns manifest
validation, resume merging, scoring, and atomic JSONL checkpoints.  It starts a
single persistent worker in ALMGuard's isolated venv, where Whisper is imported
and the requested large-v3 checkpoint is loaded exactly once.

Examples:
  scripts/almguard/score_run9_asr.py \
    --manifest data/manifests/run9_fresh_clean.jsonl \
    --data-dir data \
    --checkpoint /workspace/audio_safety_data/cache/whisper/large-v3.pt \
    --out outputs/run9/asr_clean.jsonl --device cuda

Rows require ``reference_text`` and either ``record_id`` or the stable tuple
``item_id`` + ``condition``/``style`` + ``path``.  Successful rows in ``--out``
are skipped on rerun; rows with ``asr_status=error`` are retried.  The output is
atomically replaced after every attempted row.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="fresh clean/attack manifest JSONL")
    parser.add_argument("--data-dir", type=Path, help="root used to resolve each row's path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="official OpenAI Whisper large-v3 .pt checkpoint",
    )
    parser.add_argument("--out", type=Path, help="resumable scored JSONL output")
    parser.add_argument("--device", default="cuda", help="Whisper device (default: cuda)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many pending rows (useful for a resumable smoke test)",
    )
    parser.add_argument(
        "--wer-max",
        type=float,
        default=0.20,
        help="maximum WER for transcript_control_passed (default: 0.20)",
    )
    parser.add_argument(
        "--token-overlap-min",
        type=float,
        default=0.60,
        help="minimum content-token recall (default: 0.60)",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _worker_main(args: argparse.Namespace) -> None:
    """JSON-lines Whisper worker; intentionally imports no project modules."""
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required in worker mode")

    # Whisper and torch exist only in ALMGuard's isolated environment.  Some
    # versions print model-loading progress to stdout, so keep stdout reserved
    # exclusively for the JSON-lines protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import librosa  # type: ignore[import-not-found]
        import numpy as np
        import soundfile as sf  # type: ignore[import-not-found]
        import whisper  # type: ignore[import-not-found]

        model = whisper.load_model(str(args.checkpoint), device=args.device)

    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            audio_path = str(request["path"])
            with contextlib.redirect_stdout(sys.stderr):
                audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1, dtype=np.float32)
                elif audio.ndim != 1:
                    raise ValueError(f"expected mono/stereo audio, got shape {audio.shape}")
                if sample_rate != 16_000:
                    audio = librosa.resample(
                        audio,
                        orig_sr=sample_rate,
                        target_sr=16_000,
                    )
                audio = np.ascontiguousarray(audio, dtype=np.float32)
                result = model.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                    verbose=False,
                    fp16=str(args.device).startswith("cuda"),
                )
            response = {
                "key": request.get("key"),
                "transcript": str(result.get("text") or "").strip(),
            }
        except Exception as exc:  # keep the worker alive across per-file failures
            response = {
                "key": request.get("key"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


class _WhisperWorker:
    def __init__(
        self,
        python: Path,
        script: Path,
        checkpoint: Path,
        device: str,
        repo: Path,
    ) -> None:
        command = [
            str(python),
            str(script),
            "--_worker",
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
        ]
        child_env = os.environ.copy()
        # ALMGuard vendors OpenAI Whisper under ALMGuard/whisper instead of
        # installing it into the isolated venv. Put that checkout ahead of
        # site-packages explicitly; cwd alone is not on sys.path when Python
        # executes this wrapper by absolute path.
        inherited_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(repo), *([inherited_pythonpath] if inherited_pythonpath else [])]
        )
        self.process = subprocess.Popen(
            command,
            cwd=repo,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def transcribe(self, key: str, audio_path: Path) -> tuple[str, str | None]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Whisper worker pipes are unavailable")
        self.process.stdin.write(
            json.dumps({"key": key, "path": str(audio_path)}, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return_code = self.process.poll()
            raise RuntimeError(f"Whisper worker exited without a response (code={return_code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Whisper worker returned invalid JSON: {line[:200]!r}") from exc
        if response.get("key") != key:
            raise RuntimeError(
                f"Whisper worker response key mismatch: {response.get('key')!r} != {key!r}"
            )
        return str(response.get("transcript") or ""), response.get("error")

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            return_code = self.process.wait(timeout=10)
        if return_code != 0:
            raise RuntimeError(f"Whisper worker exited with code {return_code}")


def _require_parent_args(args: argparse.Namespace) -> None:
    for name in ("manifest", "data_dir", "checkpoint", "out"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.wer_max < 0:
        raise SystemExit("--wer-max must be non-negative")
    if not 0 <= args.token_overlap_min <= 1:
        raise SystemExit("--token-overlap-min must be in [0, 1]")
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.data_dir.is_dir():
        raise SystemExit(f"data directory not found: {args.data_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Whisper checkpoint not found: {args.checkpoint}")
    if "large-v3" not in args.checkpoint.name.lower():
        raise SystemExit(
            "--checkpoint must identify the official Whisper large-v3 checkpoint "
            f"(expected a filename containing 'large-v3', got {args.checkpoint.name!r})"
        )


def _validate_resume_configuration(
    rows: list[dict[str, Any]],
    *,
    checkpoint: Path,
    wer_max: float,
    token_overlap_min: float,
) -> None:
    expected_checkpoint = str(checkpoint.resolve())
    expected_thresholds = {
        "wer_max": wer_max,
        "token_overlap_min": token_overlap_min,
    }
    for row in rows:
        if row.get("asr_status") != "ok":
            continue
        stored_checkpoint = row.get("asr_checkpoint")
        if (
            stored_checkpoint is None
            or str(Path(stored_checkpoint).resolve()) != expected_checkpoint
        ):
            raise SystemExit(
                "completed --out rows use a different/unknown ASR checkpoint; "
                "move the old output or rerun with its original --checkpoint"
            )
        if row.get("asr_thresholds") != expected_thresholds:
            raise SystemExit(
                "completed --out rows use different ASR thresholds; move the old output "
                "or reuse its --wer-max/--token-overlap-min values"
            )


def _resolve_audio_path(row: dict[str, Any], data_dir: Path) -> Path:
    raw = str(row.get("path") or "").strip()
    if not raw:
        raise ValueError("manifest row has no non-empty path")
    path = Path(raw)
    return path if path.is_absolute() else data_dir / path


def _parent_main(args: argparse.Namespace) -> None:
    _require_parent_args(args)

    from audio_safety.evaluation.asr_faithfulness import (
        asr_record_key,
        atomic_save_jsonl,
        merge_asr_resume_state,
        ordered_checkpoint_rows,
        score_asr_record,
    )
    from audio_safety.utils.io import load_jsonl

    manifest_rows = load_jsonl(args.manifest)
    existing_rows = load_jsonl(args.out) if args.out.exists() else []
    _validate_resume_configuration(
        existing_rows,
        checkpoint=args.checkpoint,
        wer_max=args.wer_max,
        token_overlap_min=args.token_overlap_min,
    )
    state, pending = merge_asr_resume_state(manifest_rows, existing_rows)
    if args.limit is not None:
        pending = pending[: args.limit]

    almguard_root = Path(os.environ.get("ALMGUARD_ROOT", "/workspace/almguard"))
    isolated_python = almguard_root / "venv" / "bin" / "python"
    almguard_repo = almguard_root / "ALMGuard"
    vendored_whisper = almguard_repo / "whisper" / "__init__.py"
    if pending and not isolated_python.is_file():
        raise SystemExit(
            f"ALMGuard venv python not found: {isolated_python}; "
            "run scripts/almguard/setup_almguard_env.sh first or set ALMGUARD_ROOT"
        )
    if pending and not vendored_whisper.is_file():
        raise SystemExit(
            f"ALMGuard vendored Whisper not found: {vendored_whisper}; "
            "run scripts/almguard/setup_almguard_env.sh first or set ALMGUARD_ROOT"
        )

    worker: _WhisperWorker | None = None
    attempted = 0
    passed = 0
    errors = 0
    try:
        if pending:
            worker = _WhisperWorker(
                isolated_python,
                Path(__file__).resolve(),
                args.checkpoint.resolve(),
                args.device,
                almguard_repo.resolve(),
            )
        for row in pending:
            key = asr_record_key(row)
            attempted += 1
            transcript = ""
            error: str | None = None
            try:
                audio_path = _resolve_audio_path(row, args.data_dir)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"audio file not found: {audio_path}")
                if worker is None:
                    raise RuntimeError("Whisper worker was not started")
                transcript, error = worker.transcribe(key, audio_path.resolve())
            except (OSError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            scored = score_asr_record(
                row,
                transcript,
                wer_max=args.wer_max,
                token_overlap_min=args.token_overlap_min,
                checkpoint=str(args.checkpoint.resolve()),
                error=error,
            )
            state[key] = scored
            passed += int(scored["transcript_control_passed"])
            errors += int(scored["asr_status"] == "error")
            atomic_save_jsonl(ordered_checkpoint_rows(manifest_rows, state), args.out)
    finally:
        if worker is not None:
            worker.close()

    # Also normalize an already-complete/out-of-order sidecar and materialize an
    # empty output when --limit 0 is used.
    atomic_save_jsonl(ordered_checkpoint_rows(manifest_rows, state), args.out)
    completed = sum(row.get("asr_status") == "ok" for row in state.values())
    total_passed = sum(bool(row.get("transcript_control_passed")) for row in state.values())
    summary = {
        "manifest_rows": len(manifest_rows),
        "already_checkpointed": len(existing_rows),
        "attempted": attempted,
        "attempt_passed": passed,
        "attempt_errors": errors,
        "completed_ok": completed,
        "total_passed": total_passed,
        "remaining": len(manifest_rows) - completed,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args._worker:
        _worker_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
