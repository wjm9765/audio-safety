#!/usr/bin/env python3
"""Measure whether both audio LMs should share the A40, instead of guessing.

The two models live in different venvs (MiniCPM needs transformers 4.46.3, Qwen
runs on the main 5.12.x env), so they are separate OS processes either way. The
only open question is whether running them CONCURRENTLY beats running them
back-to-back. Measured residency is 14.83 + 15.64 = 30.5 GiB of 45, so it fits;
fitting is not the same as being faster.

This script is one worker. It runs N generations for ONE model and writes timing,
peak VRAM and the exact generated texts. The driver (--mode) runs it three ways:

    solo-qwen | solo-minicpm | concurrent

Co-residency is adopted only if ALL of these hold (Codex R7 Q4):
  * combined peak residency <= 38.25 GiB (85% of 46068 MiB)
  * aggregate throughput improves by >= 20%
  * greedy text is IDENTICAL to the solo run, per item
  * no module is offloaded to CPU
Anything less and we serialize; the 89 s load time is immaterial next to the risk
of an OOM 60% into a capture run.

Invoke per model with its own interpreter:
    $MINICPM_ENV/bin/python scripts/exp2/bench_coresidency.py --model minicpm
    .venv/bin/python scripts/exp2/bench_coresidency.py --model qwen
where MINICPM_ENV=/workspace/audio_safety_data/envs/minicpm
"""

import argparse
import json
import os
import time
from pathlib import Path

CACHE = "/workspace/audio_safety_data/cache"
os.environ.setdefault("HF_HUB_CACHE", CACHE)
os.environ.setdefault("HF_HOME", CACHE)
DATA = Path("/workspace/audio_safety_data/data")
INSTRUCTION = "Please respond to the request in the audio."


def pick_items(manifest: Path, n: int) -> list[dict]:
    """Deterministic: the n/2 longest wavs plus a hash-spread sample.

    The longest files matter because peak VRAM and per-item latency are driven by
    audio length, and a benchmark of short clips would understate both.
    """
    rows = [json.loads(x) for x in manifest.open() if x.strip()]
    rows = [r for r in rows if r.get("safety_label") == "harmful"]
    for r in rows:
        p = DATA / r["path"]
        r["_size"] = p.stat().st_size if p.exists() else 0
    longest = sorted(rows, key=lambda r: -r["_size"])[: n // 2]
    rest = sorted((r for r in rows if r not in longest), key=lambda r: r["item_id"])
    stride = max(1, len(rest) // max(1, n - len(longest)))
    spread = rest[:: stride][: n - len(longest)]
    return longest + spread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["qwen", "minicpm"], required=True)
    ap.add_argument("--tag", default="solo", help="solo | concurrent")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--manifest", type=Path,
                    default=DATA / "manifests" / "run9_fresh_clean.jsonl")
    ap.add_argument("--outdir", type=Path,
                    default=Path("/workspace/audio_safety_data/outputs/exp2_bench"))
    args = ap.parse_args()

    items = pick_items(args.manifest, args.n)
    args.outdir.mkdir(parents=True, exist_ok=True)

    import torch

    t_load = time.time()
    if args.model == "minicpm":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-2_6", trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16, cache_dir=CACHE,
            init_vision=False, init_audio=True, init_tts=False,
        ).eval().cuda()
        tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6",
                                            trust_remote_code=True, cache_dir=CACHE)

        import librosa

        def gen(path):
            audio, _ = librosa.load(str(path), sr=16000, mono=True)
            with torch.inference_mode():
                return model.chat(msgs=[{"role": "user", "content": [audio, INSTRUCTION]}],
                                  tokenizer=tok, sampling=False,
                                  max_new_tokens=args.max_new_tokens,
                                  use_tts_template=False, generate_audio=False)
    else:
        from audio_safety.config.schema import ModelConfig
        from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio
        cfg = ModelConfig(model_id="Qwen/Qwen2-Audio-7B-Instruct", dtype="bfloat16",
                          device_map="auto", attn_implementation="sdpa")
        model, processor = load_qwen2_audio(cfg, cache_dir=Path(CACHE))

        def gen(path):
            return generate_audio_response(model, processor, path, INSTRUCTION,
                                           max_new_tokens=args.max_new_tokens, do_sample=False)

    load_s = time.time() - t_load
    devices = {str(p.device) for p in model.parameters()}
    offloaded = any(not d.startswith("cuda") for d in devices)

    # warm up so the first CUDA graph / kernel autotune is not charged to item 1
    gen(DATA / items[0]["path"])
    torch.cuda.reset_peak_memory_stats()

    recs, t0 = [], time.time()
    for it in items:
        t = time.time()
        text = gen(DATA / it["path"])
        recs.append({"item_id": it["item_id"], "seconds": time.time() - t,
                     "text": text if isinstance(text, str) else str(text)})
    wall = time.time() - t0

    out = {
        "model": args.model, "tag": args.tag, "n": len(recs),
        "load_seconds": round(load_s, 1),
        "wall_seconds": round(wall, 2),
        "sec_per_gen": round(wall / len(recs), 3),
        "gens_per_hour": round(3600 * len(recs) / wall, 1),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "param_devices": sorted(devices),
        "cpu_offloaded": offloaded,
        "records": recs,
    }
    p = args.outdir / f"{args.model}_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[bench] {args.model}/{args.tag}: {out['sec_per_gen']}s/gen  "
          f"{out['gens_per_hour']}/h  peak {out['peak_reserved_gib']} GiB reserved  "
          f"offload={offloaded}  -> {p}", flush=True)


if __name__ == "__main__":
    main()
