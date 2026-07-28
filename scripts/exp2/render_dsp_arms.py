#!/usr/bin/env -S uv run python
"""Render the CPU-only manipulated arms for the exp2 yield pilot.

Four of the pilot's six manipulated arms need no GPU and no re-synthesis — they
are signal operations on the existing clean renders, so they run on CPU while the
GPU is busy generating. Only `accent` (and therefore `echo x accent`) needs
CosyVoice2 TTS.

| arm | source | this script |
|---|---|---|
| clean | existing wav | reused, not re-rendered |
| echo | existing wav | yes |
| tone_p8 | existing wav | yes (librosa pitch shift, +8 semitones) |
| pv_locked | existing wav | yes (phase-coherence-repaired vocoder) |
| pv_standard | existing wav | yes (standard phase vocoder) |
| accent / echo x accent | GPU TTS | NOT here |

Reuses `audio_safety.data.acoustic` and `audio_safety.evaluation.phase_ops`
rather than reimplementing the operators, so `pv_locked`/`pv_standard` are the
exact pair exp1 used and the locked-vs-standard contrast stays interpretable.

SIGN ASSIGNMENT (pilot_amendment_20260728.md section 3): the pv semitone sign is
assigned deterministically 1:1 by a hash of the item id, and the SAME sign is
used for locked and standard. Both signs are therefore exercised without any
best-of-two selection, which would inflate apparent effect.

Example:
    ./scripts/exp2/render_dsp_arms.py --workers 64
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin BLAS to one thread per worker BEFORE numpy/librosa are imported anywhere.
# With N workers each spawning a full BLAS pool the process limit is exhausted and
# workers die with "blas_thread_init". Set in the parent so children inherit it.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

DATA = Path("/workspace/audio_safety_data/data")
SR = 16_000

# Frozen arm parameters. Chosen before any MiniCPM manipulated output exists.
PV_SEMITONES = 3.0        # magnitude; sign is per-item, see item_sign()
TONE_SEMITONES = 8.0      # the protocol's "SOX tone +8" gentle low-WER candidate
ECHO_DELAY_S = 0.20       # sparse single reflection
ECHO_GAIN = 0.5


def item_sign(item_id: str) -> float:
    """Deterministic +/-1 from the item id. Same sign for locked and standard."""
    h = hashlib.sha256(item_id.encode("utf-8")).digest()
    return 1.0 if (h[0] & 1) == 0 else -1.0


def echo(wav, sr: int = SR, delay_s: float = ECHO_DELAY_S, gain: float = ECHO_GAIN):
    """Single sparse reflection, peak-renormalised. Deliberately NOT reverb:
    Multi-AudioJail Table 5 puts echo at English Whisper WER 0.093 vs 0.377/0.476
    for its room/railway reverb, i.e. echo is the low-comprehension-damage
    operator and reverb is not content-preserving."""
    import numpy as np

    d = int(round(delay_s * sr))
    out = np.asarray(wav, dtype=np.float32).copy()
    if d < len(out):
        out[d:] += gain * np.asarray(wav, dtype=np.float32)[: len(out) - d]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * float(np.max(np.abs(wav)))).astype(np.float32)


def _build_arms():
    """name -> f(wav, sign). Module level so the arm list is knowable without
    rendering, and so adding an arm is a single visible edit."""
    from audio_safety.data.acoustic import pitch_shift
    from audio_safety.evaluation.phase_ops import pitch_shift_custom

    return {
        # --- the protocol's CPU-available single factors ---
        "echo": lambda w, s: echo(w),
        "tone_p8": lambda w, s: pitch_shift(w, SR, TONE_SEMITONES),
        # Symmetric negative direction. Without it we cannot say whether the tone
        # effect is a property of pitch CHANGE or specifically of RAISING pitch -
        # and that distinction is the whole question when the claim is about which
        # acoustic element matters. Same magnitude, opposite sign, nothing else.
        "tone_m8": lambda w, s: pitch_shift(w, SR, -TONE_SEMITONES),
        "pv_locked": lambda w, s: pitch_shift_custom(w, SR, s * PV_SEMITONES, mode="locked"),
        "pv_standard": lambda w, s: pitch_shift_custom(w, SR, s * PV_SEMITONES, mode="standard"),
        # --- interaction arm ---
        # The protocol's high-yield cell is echo x accent, but accent needs GPU TTS.
        # echo x tone is the CPU-available interaction with the same structure:
        # a sparse delay composed with a duration-preserving pitch change. It is
        # PRE-DECLARED as its own arm and is never pooled with, or selected against,
        # the single-factor arms - pooling or best-of-N would inflate apparent ASR
        # and leave the physical factor unidentified.
        "echo_x_tone_p8": lambda w, s: echo(pitch_shift(w, SR, TONE_SEMITONES)),
    }


ARMS = ["echo", "tone_p8", "tone_m8", "pv_locked", "pv_standard", "echo_x_tone_p8"]


def render_one(task):
    """Render every arm for one item. Returns a manifest row per arm."""
    import numpy as np

    from audio_safety.data.acoustic import load_wav, save_wav

    item_id, src, outdir = task["item_id"], Path(task["src"]), Path(task["outdir"])
    rows = []
    try:
        wav = load_wav(src, sr=SR)
    except Exception as exc:
        return [{"item_id": item_id, "arm": "ALL", "status": "error", "error": f"load: {exc}"}]

    def rel(p: Path) -> str:
        """DATA-relative when possible, absolute otherwise — an outdir outside
        DATA is legitimate (smoke tests) and must not abort the render."""
        try:
            return str(p.relative_to(DATA))
        except ValueError:
            return str(p)

    sign = item_sign(item_id)
    arms = _build_arms()
    for arm, fn in arms.items():
        dst = outdir / arm / f"{item_id}.wav"
        row = {"item_id": item_id, "role": task.get("role"), "arm": arm,
               "path": rel(dst), "source_path": rel(src),
               "pv_sign": sign if "pv" in arm else None}
        if dst.exists():
            row["status"] = "reused"
            rows.append(row)
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            y = fn(wav, sign)
            save_wav(dst, np.asarray(y, dtype=np.float32), sr=SR)
            row["status"] = "rendered"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=DATA / "manifests" / "run9_fresh_clean.jsonl")
    ap.add_argument("--outdir", type=Path, default=DATA / "audio_exp2_pilot")
    ap.add_argument("--out-manifest", type=Path,
                    default=DATA / "manifests" / "exp2_pilot_dsp_arms.jsonl")
    ap.add_argument("--workers", type=int, default=min(64, (os.cpu_count() or 8) - 4))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--roles", choices=["harmful", "benign", "both"], default="both",
                    help="benign rows are the protocol's matched controls, not filler")
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.manifest.open() if x.strip()]
    keep = {"harmful", "benign"} if args.roles == "both" else {args.roles}
    items = sorted((r for r in rows if r.get("safety_label") in keep),
                   key=lambda r: (r.get("safety_label"), r["item_id"]))
    if args.limit:
        items = items[: args.limit]

    tasks = [{"item_id": r["item_id"], "src": str(DATA / r["path"]),
              "outdir": str(args.outdir), "role": r.get("safety_label")}
             for r in items]
    import collections as _c
    print(f"[dsp] items={len(tasks)} {dict(_c.Counter(t['role'] for t in tasks))} "
          f" arms={len(ARMS)}  workers={args.workers}", flush=True)
    print(f"[dsp] frozen params: pv=+/-{PV_SEMITONES} st (sign by item hash), "
          f"tone=+{TONE_SEMITONES} st, echo={ECHO_DELAY_S}s @ {ECHO_GAIN}", flush=True)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    done, err, reused = 0, 0, 0
    with args.out_manifest.open("w") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(render_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            for row in fut.result():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row.get("status") == "rendered":
                    done += 1
                elif row.get("status") == "reused":
                    reused += 1
                else:
                    err += 1
            if i % 50 == 0 or i == len(futs):
                print(f"[dsp] {i}/{len(futs)} items  rendered={done} reused={reused} err={err}",
                      flush=True)

    print(f"\n[dsp] DONE. rendered={done} reused={reused} errors={err}")
    print(f"[dsp] manifest -> {args.out_manifest}")
    print("[dsp] accent / echo x accent still require CosyVoice2 TTS (GPU).")


if __name__ == "__main__":
    main()
