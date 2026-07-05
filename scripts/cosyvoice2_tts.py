#!/usr/bin/env -S uv run python
"""CosyVoice2 TTS wrapper used by the Audio-RDO render stage.

This script keeps CosyVoice2 isolated from the project environment. On first use it
clones the official CosyVoice repo under AUDIO_SAFETY_CACHE_DIR, creates a local
TTS virtualenv with uv, installs the official requirements there, downloads the
CosyVoice2-0.5B checkpoint from Hugging Face, then re-execs itself inside that
TTS virtualenv. Single-file rendering is supported for debugging; production
renders should use --batch-jsonl so the CosyVoice model is loaded once per batch.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = Path("/workspace/audio_safety_data")
DEFAULT_REPO = "https://github.com/FunAudioLLM/CosyVoice.git"
DEFAULT_MODEL_ID = "FunAudioLLM/CosyVoice2-0.5B"
DEFAULT_TTS_PYTHON = "3.11"
CHILD_ENV = "AUDIO_SAFETY_COSYVOICE2_CHILD"
TTS_BUILD_DEPS = ["setuptools<81", "wheel", "wheel_stub", "numpy==1.26.4", "cython"]

STYLE_INSTRUCTIONS = {
    "neutral": "Speak in a neutral, clear voice.<|endofprompt|>",
    "sad": "Speak in a sad, subdued voice.<|endofprompt|>",
    "fearful": "Speak in a fearful, anxious voice.<|endofprompt|>",
    "angry": "Speak in an angry, tense voice.<|endofprompt|>",
    "elderly_male": "Speak like an elderly male narrator, slow and clear.<|endofprompt|>",
    "child_female": "Speak like a young female child, clear and natural.<|endofprompt|>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=None, help="raw text to render")
    parser.add_argument("--text-json", default=None, help="JSON string containing the text")
    parser.add_argument("--batch-jsonl", type=Path, default=None, help="JSONL batch render jobs")
    parser.add_argument("--style", default="neutral", help="style key from the experiment config")
    parser.add_argument("--output", type=Path, default=None, help="wav output path")
    parser.add_argument("--setup-only", action="store_true", help="bootstrap CosyVoice2 and exit")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override TTS cache root")
    parser.add_argument("--repo-dir", type=Path, default=None, help="override CosyVoice repo path")
    parser.add_argument(
        "--model-dir", type=Path, default=None, help="override model checkpoint path"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--prompt-audio", type=Path, default=None, help="zero-shot prompt wav")
    parser.add_argument("--prompt-text", default="", help="optional zero-shot prompt text")
    parser.add_argument(
        "--python",
        default=DEFAULT_TTS_PYTHON,
        help="Python version for the isolated TTS virtualenv",
    )
    return parser.parse_args()


def project_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir
    env_cache = os.environ.get("AUDIO_SAFETY_CACHE_DIR")
    if env_cache:
        return Path(env_cache) / "cosyvoice2"
    env_workspace = os.environ.get("AUDIO_SAFETY_WORKSPACE")
    workspace = Path(env_workspace) if env_workspace else DEFAULT_WORKSPACE
    return workspace / "cache" / "cosyvoice2"


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root = project_cache_dir(args)
    repo_dir = args.repo_dir or root / "CosyVoice"
    venv_dir = root / ".venv"
    model_dir = args.model_dir or root / "pretrained_models" / "CosyVoice2-0.5B"
    return root, repo_dir, venv_dir, model_dir


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[cosyvoice2] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def ensure_repo(repo_dir: Path) -> None:
    if (repo_dir / "cosyvoice").is_dir() and (repo_dir / "third_party").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(f"CosyVoice repo path exists but is incomplete: {repo_dir}")
    run(["git", "clone", "--recursive", DEFAULT_REPO, str(repo_dir)])


def ensure_venv(venv_dir: Path, repo_dir: Path, python_request: str) -> None:
    py = venv_python(venv_dir)
    if not py.exists():
        try:
            run(["uv", "venv", str(venv_dir), "--python", python_request])
        except subprocess.CalledProcessError:
            run(["uv", "venv", str(venv_dir), "--python", sys.executable])
    requirements = repo_dir / "requirements.txt"
    stamp = venv_dir / ".audio_safety_cosyvoice2_requirements_installed"
    if stamp.exists():
        return
    run(
        [
            "uv",
            "--no-cache",
            "pip",
            "install",
            "--python",
            str(py),
            "--index-strategy",
            "unsafe-best-match",
            *TTS_BUILD_DEPS,
        ]
    )
    run(
        [
            "uv",
            "--no-cache",
            "pip",
            "install",
            "--python",
            str(py),
            "--index-strategy",
            "unsafe-best-match",
            "--no-build-isolation",
            "-r",
            str(requirements),
        ]
    )
    stamp.write_text("ok\n")


def ensure_model(model_dir: Path, model_id: str) -> None:
    if any(
        (model_dir / name).exists() for name in ("cosyvoice2.yaml", "cosyvoice.yaml", "config.yaml")
    ):
        return
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=str(model_dir))


def bootstrap(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root, repo_dir, venv_dir, model_dir = paths(args)
    root.mkdir(parents=True, exist_ok=True)
    ensure_repo(repo_dir)
    ensure_venv(venv_dir, repo_dir, args.python)
    return root, repo_dir, venv_dir, model_dir


def reexec_in_tts_venv(args: argparse.Namespace, venv_dir: Path) -> None:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    cache_root = project_cache_dir(args)
    env.setdefault("HF_HOME", str(DEFAULT_WORKSPACE / "cache" / "huggingface"))
    env.setdefault("HF_HUB_CACHE", str(DEFAULT_WORKSPACE / "cache" / "huggingface" / "hub"))
    env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    py = venv_python(venv_dir)
    os.execvpe(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def text_from_args(args: argparse.Namespace) -> str:
    if args.text_json is not None:
        try:
            return str(json.loads(args.text_json))
        except json.JSONDecodeError:
            return args.text_json
    if args.text is not None:
        return args.text
    raise ValueError("one of --text, --text-json, or --batch-jsonl is required")


def text_from_job(job: dict[str, Any]) -> str:
    if job.get("text") is not None:
        return str(job["text"])
    if job.get("text_json") is not None:
        value = str(job["text_json"])
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            return value
    raise ValueError(f"batch job is missing text/text_json: {job}")


def load_batch_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_renderer(args: argparse.Namespace):
    _root, repo_dir, _venv_dir, model_dir = paths(args)
    ensure_model(model_dir, args.model_id)

    sys.path.insert(0, str(repo_dir))
    sys.path.append(str(repo_dir / "third_party" / "Matcha-TTS"))

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    prompt_audio = args.prompt_audio or repo_dir / "asset" / "zero_shot_prompt.wav"
    model = AutoModel(model_dir=str(model_dir))
    return torch, torchaudio, model, prompt_audio


def synthesize_to_file(
    *,
    torch,
    torchaudio,
    model,
    text: str,
    style: str,
    prompt_audio: Path,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    instruction = STYLE_INSTRUCTIONS.get(
        style, f"Speak in a {style} style.<|endofprompt|>"
    )
    chunks = []
    for chunk in model.inference_instruct2(text, instruction, str(prompt_audio), stream=False):
        chunks.append(chunk["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice2 produced no audio chunks")
    speech = torch.cat(chunks, dim=-1).detach().cpu()
    torchaudio.save(str(output), speech, model.sample_rate)
    print(f"[cosyvoice2] wrote {output}", flush=True)


def render(args: argparse.Namespace) -> None:
    text = text_from_args(args)
    output = args.output
    if output is None:
        raise ValueError("--output is required unless --setup-only or --batch-jsonl is used")
    torch, torchaudio, model, prompt_audio = prepare_renderer(args)
    synthesize_to_file(
        torch=torch,
        torchaudio=torchaudio,
        model=model,
        text=text,
        style=args.style,
        prompt_audio=prompt_audio,
        output=output,
    )


def render_batch(args: argparse.Namespace) -> None:
    if args.batch_jsonl is None:
        raise ValueError("--batch-jsonl is required for batch rendering")
    jobs = load_batch_jobs(args.batch_jsonl)
    if not jobs:
        print(f"[cosyvoice2] no batch jobs in {args.batch_jsonl}", flush=True)
        return
    torch, torchaudio, model, prompt_audio = prepare_renderer(args)
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        output_value = job.get("output_path") or job.get("output")
        if output_value is None:
            raise ValueError(f"batch job is missing output_path/output: {job}")
        output = Path(str(output_value))
        style = str(job.get("style") or args.style)
        item_id = str(job.get("item_id") or job.get("query_id") or output.stem)
        safety_label = str(job.get("safety_label") or job.get("query_type") or "unknown")
        if output.exists():
            print(f"[cosyvoice2] batch {index}/{total} exists {output}", flush=True)
            continue
        print(
            f"[cosyvoice2] batch {index}/{total} {item_id} {safety_label}/{style}",
            flush=True,
        )
        synthesize_to_file(
            torch=torch,
            torchaudio=torchaudio,
            model=model,
            text=text_from_job(job),
            style=style,
            prompt_audio=prompt_audio,
            output=output,
        )
    print(f"[cosyvoice2] batch complete {total} jobs from {args.batch_jsonl}", flush=True)


def main() -> None:
    args = parse_args()
    _root, _repo_dir, venv_dir, model_dir = bootstrap(args)
    if os.environ.get(CHILD_ENV) != "1":
        reexec_in_tts_venv(args, venv_dir)
    if args.setup_only:
        ensure_model(model_dir, args.model_id)
        print(f"[cosyvoice2] setup complete under {project_cache_dir(args)}")
        return
    if args.batch_jsonl is not None:
        render_batch(args)
        return
    render(args)


if __name__ == "__main__":
    main()
