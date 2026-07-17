#!/usr/bin/env -S uv run python
"""Drive the isolated ALMGuard defense from our env via subprocess + files.

ALMGuard lives in its own venv (scripts/almguard/setup_almguard_env.sh) with a
torch/transformers stack incompatible with ours, so we NEVER import it — we shell
into its venv and call its own CLI (eval_qwen.py / main.py), which is the stable
interface, and exchange data as files. This wrapper is pure stdlib (no torch in our
process).

Modes:
  undefended  their eval_qwen.py with a ZERO perturbation -> the no-defense baseline
              measured IN the child env (Codex 2026-07-17: baseline must run in-child
              so the transformers-version and inference path are held fixed vs defended).
  defended    their eval_qwen.py with the trained SAP perturbation.
  train       their main.py to optimize the SAP on adversarial audios.

NON-NEGOTIABLE (encoded as a guard): in `train` mode the adversarial dirs must NOT
contain the channel/phase attack under test — an attack-aware SAP makes the gate
meaningless. Pass --assert-excludes <substr> to fail fast if any training file
name matches the attack tag.
"""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from audio_safety.pipelines.almguard_io import (
    align_responses,
    excluded_training_files,
    staged_mapping_row,
    staged_wav_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["undefended", "defended", "train"], required=True)
    p.add_argument("--almguard-root", type=Path, default=Path("/workspace/almguard"))
    p.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="MATCH the checkpoint our attack uses (Instruct), not ALMGuard's base default",
    )
    # eval (undefended/defended)
    p.add_argument("--manifest", type=Path, help="JSONL eval rows (item_id, path, ...)")
    p.add_argument("--data-dir", type=Path, help="root to resolve manifest 'path' fields")
    p.add_argument("--perturb-path", type=Path, help="trained SAP .pth (defended mode)")
    p.add_argument(
        "--zero-like",
        type=Path,
        default=None,
        help="undefended mode: shape the zero SAP like this SAP's PTB (if eval_qwen "
        "applies the perturbation shape-sensitively rather than as a scalar broadcast)",
    )
    p.add_argument("--out", type=Path, help="output JSONL of responses")
    # train
    p.add_argument("--adv-dirs", type=Path, nargs="*", default=[], help="adversarial wav dirs")
    p.add_argument("--sap-out", type=Path, help="SAP save dir (train mode)")
    p.add_argument("--asr-path", type=str, default="./models/large-v3.pt")
    p.add_argument(
        "--assert-excludes",
        type=str,
        default=None,
        help="fail if any training filename contains this tag (e.g. the channel-attack tag)",
    )
    p.add_argument("--work-dir", type=Path, default=None)
    return p.parse_args()


def _venv_python(root: Path) -> Path:
    py = root / "venv" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"ALMGuard venv python not found at {py}; run setup_almguard_env.sh first")
    return py


def _repo(root: Path) -> Path:
    repo = root / "ALMGuard"
    if not (repo / "eval_qwen.py").exists():
        raise SystemExit(f"ALMGuard repo not found at {repo}; run setup_almguard_env.sh first")
    return repo


def _stage_wavs(manifest: Path, data_dir: Path, work: Path) -> list[dict]:
    """Stage manifest audio into a ZERO-PADDED wav dir; return index->row map.

    Zero-padded names (staged_wav_name) make ALMGuard's filename sort — whether
    numeric or lexicographic — coincide with index order, so the positional
    responses[i] <-> row i alignment in align_responses is safe.
    """
    wav_dir = work / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    mapping = []
    for i, row in enumerate(rows):
        src = data_dir / str(row["path"])
        dst = wav_dir / staged_wav_name(i)
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copyfile(src, dst)
        mapping.append(staged_mapping_row(i, row))
    return mapping


def _load_responses(save_path: Path) -> list:
    pkls = sorted(save_path.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no .pkl response file produced under {save_path}")
    target = next((p for p in pkls if "responses" in p.name), pkls[-1])
    with target.open("rb") as fh:
        return pickle.load(fh)


def _run_eval(args: argparse.Namespace, perturb: Path, defense: str) -> None:
    py, repo = _venv_python(args.almguard_root), _repo(args.almguard_root)
    work = args.work_dir or (args.out.parent / f"_almguard_{defense}")
    work.mkdir(parents=True, exist_ok=True)
    # A fresh invocation directory prevents stale staged WAVs or response PKLs
    # from a prior/resumed call from corrupting positional alignment.
    invocation = work / "invocations" / uuid.uuid4().hex
    mapping = _stage_wavs(args.manifest, args.data_dir, invocation)
    save_path = invocation / "responses"
    save_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        "eval_qwen.py",
        "--model_path",
        args.model_path,
        "--wav_dirs",
        str((invocation / "wavs").resolve()),
        "--perturb_path",
        str(perturb.resolve()),
        "--save_path",
        str(save_path.resolve()),
    ]
    print(f"[almguard] {defense}: {' '.join(cmd)} (cwd={repo})", flush=True)
    subprocess.run(cmd, cwd=repo, check=True)
    responses = _load_responses(save_path)
    if len(responses) != len(mapping):
        print(
            f"[almguard] WARN: {len(responses)} responses vs {len(mapping)} inputs; "
            "alignment may be off",
            file=sys.stderr,
        )
    records = [{**rec, "defense": defense} for rec in align_responses(responses, mapping)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[almguard] wrote {len(records)} '{defense}' responses -> {args.out}", flush=True)


def _make_zero_ptb(args: argparse.Namespace, work: Path) -> Path:
    """Create a no-op perturbation IN the child venv (avoids cross-torch pickle).

    A 0-dim scalar zero is a clean no-op ONLY if eval_qwen.py applies the SAP as a
    plain additive broadcast (input_features + PTB). If --zero-like <sap.pth> is
    given, the zero tensor is shaped like that SAP's PTB instead — use this if the
    on-box eval_qwen applies the perturbation shape-sensitively (masked/indexed),
    so the undefended baseline is a true no-op rather than an accidental broadcast.
    """
    py = _venv_python(args.almguard_root)
    ptb = work / "zero_ptb.pth"
    if args.zero_like is not None:
        code = (
            "import torch; "
            f"src=torch.load(r'{Path(args.zero_like).resolve()}',map_location='cpu')['PTB']; "
            f"torch.save({{'PTB': torch.zeros_like(src)}}, r'{ptb.resolve()}')"
        )
    else:
        code = f"import torch; torch.save({{'PTB': torch.zeros(())}}, r'{ptb.resolve()}')"
    subprocess.run([str(py), "-c", code], check=True)
    return ptb


def main() -> None:
    args = parse_args()
    if args.mode in {"undefended", "defended"}:
        for req in ("manifest", "data_dir", "out"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} is required for {args.mode} mode")
        if args.mode == "defended":
            if args.perturb_path is None:
                raise SystemExit("--perturb-path (trained SAP) is required for defended mode")
            _run_eval(args, args.perturb_path, "almguard")
        else:
            work = args.work_dir or (args.out.parent / "_almguard_undefended")
            work.mkdir(parents=True, exist_ok=True)
            _run_eval(args, _make_zero_ptb(args, work), "none")
        return

    # train
    if not args.adv_dirs or args.sap_out is None:
        raise SystemExit("--adv-dirs (3) and --sap-out are required for train mode")
    if args.assert_excludes:
        bad = excluded_training_files(args.adv_dirs, args.assert_excludes)
        if bad:
            raise SystemExit(
                f"attack-aware SAP guard: training paths match '{args.assert_excludes}': "
                f"{bad[:3]}... — the attack under test must be EXCLUDED from SAP training"
            )
    py, repo = _venv_python(args.almguard_root), _repo(args.almguard_root)
    args.sap_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        "main.py",
        "--model_path",
        args.model_path,
        "--asr_path",
        args.asr_path,
        "--save_path",
        str(args.sap_out.resolve()),
        "--wav_dirs",
        *[str(Path(d).resolve()) for d in args.adv_dirs],
    ]
    print(f"[almguard] train SAP: {' '.join(cmd)} (cwd={repo})", flush=True)
    subprocess.run(cmd, cwd=repo, check=True)
    print(f"[almguard] SAP checkpoints under {args.sap_out}", flush=True)


if __name__ == "__main__":
    main()
