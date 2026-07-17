#!/usr/bin/env -S uv run python
"""Validate the completed Run 9 ALMGuard SAP without loading it on the GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from audio_safety.pipelines.almguard_training_data import (
    DataContractError,
    validate_prepared,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--almguard-root", type=Path, default=Path("/workspace/almguard"))
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sap-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--expected-upstream-commit",
        default="244c657f35eeca3c08b6970efbf6fb92b9361712",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_ptb(python: Path, checkpoint: Path, mask_path: Path, tau: float) -> dict:
    code = r"""
import json, sys
import numpy as np
import torch

checkpoint, mask_path, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
artifact = torch.load(checkpoint, map_location="cpu")
if not isinstance(artifact, dict) or "PTB" not in artifact:
    raise SystemExit("checkpoint must be a dict containing PTB")
ptb = artifact["PTB"]
if not isinstance(ptb, torch.Tensor):
    raise SystemExit("PTB must be a torch.Tensor")
if ptb.ndim != 3 or ptb.shape[0] != 1 or ptb.shape[1] != 128:
    raise SystemExit(f"unexpected PTB shape: {tuple(ptb.shape)}")
if not bool(torch.isfinite(ptb).all()):
    raise SystemExit("PTB contains non-finite values")
mask = torch.from_numpy(np.load(mask_path)["mask"]).to(torch.bool)
if tuple(mask.shape) != (128,):
    raise SystemExit(f"unexpected mask shape: {tuple(mask.shape)}")
outside = ptb[:, ~mask, :]
outside_max = float(outside.abs().max()) if outside.numel() else 0.0
max_abs = float(ptb.abs().max())
nonzero = int(torch.count_nonzero(ptb))
if max_abs > tau + 1e-5:
    raise SystemExit(f"PTB exceeds tau: {max_abs} > {tau}")
if outside_max > 1e-7:
    raise SystemExit(f"PTB is nonzero outside the shipped mask: {outside_max}")
if nonzero == 0:
    raise SystemExit("PTB is identically zero")
print(json.dumps({
    "shape": list(ptb.shape),
    "dtype": str(ptb.dtype),
    "finite": True,
    "max_abs": max_abs,
    "outside_mask_max_abs": outside_max,
    "nonzero": nonzero,
}))
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [str(python), "-c", code, str(checkpoint), str(mask_path), str(tau)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.tau <= 0:
        raise SystemExit("--tau must be positive")
    # Prefer the schedule the training run recorded (reduced budget) so a 3-epoch SAP
    # validates without a manual --num-epochs; fall back to the CLI value otherwise.
    run_config_path = args.sap_dir.resolve() / "sap_run_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        num_epochs = int(run_config.get("num_epochs", args.num_epochs))
        num_epochs_source = "sap_run_config.json"
    else:
        run_config = None
        num_epochs = args.num_epochs
        num_epochs_source = "--num-epochs"
    if num_epochs < 1:
        raise SystemExit("resolved num_epochs must be >= 1")
    root = args.almguard_root.resolve()
    repo = root / "ALMGuard"
    python = root / "venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"isolated ALMGuard python not found: {python}")
    try:
        contract = validate_prepared(
            args.prepared_root.resolve(), data_dir=args.data_dir.resolve()
        )
    except DataContractError as exc:
        raise SystemExit(f"prepared SAP data failed validation: {exc}") from exc
    if contract.get("upstream_commit") != args.expected_upstream_commit:
        raise SystemExit("prepared-data upstream commit differs from the pinned Run 9 commit")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual_commit != args.expected_upstream_commit:
        raise SystemExit("checked-out ALMGuard commit differs from the pinned Run 9 commit")
    forbidden = {str(value).casefold() for value in contract.get("forbidden_tokens", ())}
    if not {"phase", "pv_standard"}.issubset(forbidden):
        raise SystemExit("prepared contract does not exclude both phase and pv_standard")

    mask_path = repo / "mask" / "global_saliency.npz"
    mask_data = np.load(mask_path)
    if set(mask_data.files) != {"avg_a", "avg_j", "mask"}:
        raise SystemExit("shipped saliency artifact has unexpected keys")
    mask = np.asarray(mask_data["mask"])
    if mask.shape != (128,) or int(np.count_nonzero(mask)) != 48:
        raise SystemExit("shipped Qwen mask must have shape (128,) and k=48")

    train_total = int(contract["train_total"])
    expected_name = f"perturb_mel_epoch_{num_epochs - 1}_iter_{train_total - 1}.pth"
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else (args.sap_dir.resolve() / expected_name)
    )
    if checkpoint.name != expected_name:
        raise SystemExit(
            f"checkpoint is not the expected final artifact {expected_name}: {checkpoint.name}"
        )
    if not checkpoint.is_file():
        raise SystemExit(f"final SAP checkpoint not found: {checkpoint}")
    expected_count = num_epochs * train_total
    checkpoints = list(args.sap_dir.resolve().glob("perturb_mel_epoch_*_iter_*.pth"))
    if len(checkpoints) != expected_count:
        raise SystemExit(
            f"expected {expected_count} SAP checkpoints, found {len(checkpoints)}"
        )
    metrics = _inspect_ptb(python, checkpoint, mask_path, args.tau)
    report = {
        "status": "passed",
        "schema_version": 1,
        "upstream_commit": actual_commit,
        "train_total": train_total,
        "num_epochs": num_epochs,
        "num_epochs_source": num_epochs_source,
        "sap_run_config": run_config,
        "checkpoint_count": len(checkpoints),
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": _sha256(checkpoint),
        "mask_path": str(mask_path),
        "mask_sha256": _sha256(mask_path),
        "mask_shape": list(mask.shape),
        "mask_nonzero": int(np.count_nonzero(mask)),
        "forbidden_tokens": sorted(forbidden),
        "ptb": metrics,
        "prompt_or_response_bodies_logged": False,
    }
    report_path = args.sap_dir.resolve() / "run9_sap_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
