# AudioSafety Agent Guide

이 문서는 Claude, Codex 및 기타 코드 에이전트가 이 프로젝트를 수정하거나 확장하기 전에 먼저 읽어야 하는 공통 기준 문서이다. Claude 전용 설정(스킬, 서브에이전트)은 `CLAUDE.md`에 있다. 실험 코드는 GPU 서버 실행, 재현성, 배포, 오픈소스 공개 가능성을 전제로 작성한다.

## 프로젝트 개요

Audio LLM(Qwen2-Audio)의 LLM residual stream 안에 조작 가능한 **audio-conditioned refusal axis**가 존재하는지 RDO-style gradient optimization으로 먼저 검증하는 mechanistic interpretability 실험 프로젝트다. 현재 실험의 사전 등록 설계는 `docs/experiments/exp1_refusal_cone_drift/design.md`에 있다.

## 기본 원칙

- 이 프로젝트는 Python 기반 실험 프로젝트이다.
- 패키지와 실행 환경 관리는 `uv`를 기준으로 한다.
- 실행 환경은 GPU 서버를 기준으로 한다.
- GPU 서버에서 `uv sync`만 실행해도 필요한 의존성이 설치될 수 있도록 `pyproject.toml`과 `uv.lock`을 관리한다.
- 코드, 설정, 스크립트, 실험 산출물의 역할을 명확히 분리한다.
- 하드코딩된 경로, 모델명, 데이터셋명, 하이퍼파라미터, 캐시 경로를 피하고 설정 파일 또는 환경 변수로 주입한다.

## 의존성 관리

`pyproject.toml`은 의존성을 역할별로 분리해서 관리한다.

- 기본 의존성: numpy/scipy/sklearn 등 분석·통계에 필요한 최소 런타임 패키지. **torch를 import하지 않고도 동작해야 한다** (통계·테스트는 CPU-only 환경에서 실행 가능).
- `dev` group: formatter(ruff), test runner(pytest), type checker 등
- `gpu` group: torch, transformers, accelerate, audio inference 관련 패키지

사용 흐름:

```bash
uv sync               # 기본 + dev
uv sync --group gpu   # GPU 서버에서 모델 실행까지
```

규칙:

- 작업을 시작하기 전에 이미 `uv sync`된 프로젝트 `.venv`가 있는지 확인하고, 있으면 `uv run`/`./scripts/<name>.py` 경로로 그 환경을 우선 사용한다.
- 프로젝트 환경 문제를 임시 `pip install` 또는 임의 `uv pip install`로 덮지 않는다. 필요한 의존성은 `pyproject.toml`/`uv.lock`에 반영한다.
- 예외는 `scripts/cosyvoice2_tts.py`처럼 외부 도구 충돌을 피하려고 명시적으로 관리하는 격리 venv뿐이다. 이 경우에도 설치 절차는 스크립트에 재현 가능하게 남긴다.
- 새 패키지를 추가할 때는 왜 필요한지, 어느 group에 들어가야 하는지 먼저 판단한다.
- `uv.lock`은 재현성을 위해 커밋 대상이다.
- 임시로 `pip install`한 상태를 코드의 전제로 삼지 않는다.
- `src/` 코드에서 torch/transformers는 함수 내부에서 lazy import한다 (기본 환경에서 import 실패 방지).

## 프로젝트 구조

```text
audio_safety/
  AGENTS.md                  # 공통 에이전트 지침 (이 파일)
  CLAUDE.md                  # Claude 진입점: AGENTS.md import + 스킬/서브에이전트 안내
  README.md                  # GitHub 공개용 프로젝트 설명
  pyproject.toml
  uv.lock
  .claude/
    skills/                  # /codex-cross-check, /update-experiment-log, /analyze-experiment
    agents/                  # adversarial-reviewer, research-code-reviewer
  configs/
    models/                  # 모델 로딩 설정 (모델 ID, dtype, device_map)
    experiments/             # 실험 단위 설정 (base config 참조 + override)
    paths/                   # workspace/cache/output 경로 설정
  docs/
    experiments/
      exp1_refusal_cone_drift/
        design.md            # 사전 등록 실험 설계 — 판정 기준 사후 수정 금지
        results.md           # 누적 run 로그 — append-only
  src/audio_safety/          # 재사용 가능한 라이브러리 코드
    config/                  # config schema(pydantic) + loader
    data/                    # 데이터셋 로딩, family 렌더링
    models/                  # 모델 로딩, activation hook
    pipelines/               # cone 구성, drift probe, activation 추출
    evaluation/              # 통계 검정, 판정 규칙
    utils/                   # 경로 해석, seed, io
  scripts/                   # CLI 진입점. 복잡한 로직은 src/로 위임
  tests/                     # config 로딩, 경로 처리, 통계 함수 단위 테스트
  data/                      # 원본 데이터는 커밋하지 않음. README만
  outputs/                   # run 산출물. git 추적 제외
```

역할 기준:

- `src/audio_safety/`: 재사용 가능한 라이브러리 코드
- `scripts/`: argument parsing과 pipeline 호출만. 복잡한 로직은 `src/`로 위임
- `configs/`: 모델, 데이터셋, 실험, 경로, 런타임 설정
- `tests/`: 핵심 설정 로딩, 경로 처리, 통계 함수 단위 테스트 (GPU 없이 통과해야 함)
- `data/`: 원본 대용량 데이터는 직접 커밋하지 않고 설명 파일만 둔다
- `outputs/`: 체크포인트, 로그, 평가 결과 등 생성물. 기본적으로 git 추적 제외

## 설정 관리

모델명, 데이터셋명, 경로, batch size, seed, decoding 설정, 평가 옵션 등은 코드에 직접 박지 않는다.

- 설정은 `configs/` 아래 YAML로 관리하고, `audio_safety.config` 로더로 pydantic schema에 검증해서 읽는다.
- 실험 config는 `model:`, `paths:` 키에 다른 config 파일 경로를 적어 참조(include)한다.
- CLI 인자는 config 파일 경로와 dotted override(`--override stats.n_permutations=100`)만 받는다.
- 랜덤 seed, output directory, run name은 모든 실험에서 명시적으로 기록한다.

실행 예시:

```bash
./scripts/run_experiment.py --config configs/experiments/exp1_refusal_cone_drift.yaml
```

## 스크립트 실행 규약

`scripts/*.py`는 GPU 서버와 로컬 개발 환경에서 바로 실행할 수 있어야 한다.

- 모든 Python CLI 스크립트는 첫 줄에 `#!/usr/bin/env -S uv run python` shebang을 둔다.
- 새 스크립트를 추가하거나 기존 스크립트를 CLI로 쓰게 만들면 반드시 executable bit를 설정한다 (`chmod +x scripts/<name>.py`).
- 문서와 예시 명령은 `uv run python scripts/...` 대신 `./scripts/<name>.py ...` 형식을 우선 사용한다.
- 복잡한 로직은 계속 `src/audio_safety/`에 두고, `scripts/`는 argument parsing과 pipeline 호출만 담당한다.

## 경로와 캐시 정책

Hugging Face, PyTorch, datasets, model checkpoints, 기타 캐시 파일은 기본 home/root 경로에 의존하지 않는다. GPU 서버/컨테이너에서 실행 사용자가 root일 수 있고, 기본 캐시가 `/root/.cache`로 들어가면 재현성·권한·용량 관리가 어려워지기 때문이다.

원칙:

- 캐시 루트는 환경 변수로 주입 가능해야 한다.
- GPU 서버에서는 가능하면 `/workspace` 하위에 캐시를 둔다.
- 코드 내부에서 `~/.cache`, `/root`, 개인 경로, 임의 절대 경로를 직접 참조하지 않는다.
- 모든 경로는 `pathlib.Path`로 다루며, 경로 해석은 `audio_safety.utils.paths`를 거친다.

권장 환경 변수:

```bash
export AUDIO_SAFETY_WORKSPACE=/workspace/audio_safety_data
export HF_HOME=/workspace/audio_safety_data/cache/huggingface
export HF_HUB_CACHE=/workspace/audio_safety_data/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/audio_safety_data/cache/huggingface/datasets
export TORCH_HOME=/workspace/audio_safety_data/cache/torch
export XDG_CACHE_HOME=/workspace/audio_safety_data/cache
```

경로 기본값 우선순위:

1. 명시적 CLI 인자
2. 프로젝트 전용 환경 변수 (`AUDIO_SAFETY_*`)
3. config 파일 값
4. `/workspace/audio_safety_data` 하위의 cache, output, data directory

## 코드 작성 규칙

- 실험 로직과 실행 스크립트를 분리한다.
- 모델 로딩, 데이터 로딩, 평가, 로깅은 각각 모듈화한다.
- 전역 상태와 import 시점의 무거운 초기화를 피한다.
- 파일 경로, 모델 ID, prompt template, metric 이름은 config에서 읽는다.
- 실험 결과는 JSONL, JSON, YAML, Markdown 등 후처리 가능한 형식으로 저장한다.
- 실패한 샘플, raw output, config snapshot, git commit hash를 함께 저장해 재현 가능하게 한다.
- 대용량 파일, 캐시, 체크포인트, 원본 데이터셋은 git에 커밋하지 않는다.

## 실험 문서 규약 (research discipline)

이 프로젝트는 pre-registration 방식으로 운영한다. 결과를 보고 기준을 옮기는 것을 구조적으로 막는 것이 목적이다.

- **`design.md` = 사전 등록 문서.** 판정 기준(§0 표), 가설, 통계 방법은 실험 시작 후 수정하지 않는다. 부득이한 수정은 문서 하단에 변경 이력(날짜, 사유)을 남긴다.
- **`results.md` = append-only run 로그.** run 단위로 항목을 추가만 하고, 과거 항목은 수정하지 않는다(정정은 새 항목으로).
- **run naming:** `exp1_{YYYYMMDD}_{HHMM}_{tag}` (예: `exp1_20260704_1430_layer_sweep`).
- **run 산출물 (`outputs/<run_name>/`):**
  - `config_snapshot.yaml` — 실행 시점 config 전체 + git commit hash
  - `metrics.json` — 판정 metric 포함 모든 수치
  - `figures/` — 생성된 그림
  - `analysis.md` — 분석 결과 (analyze-experiment 산출물)
  - raw activation/projection은 `.npz`/`.npy`로 같은 디렉터리에
- **판정(GO/NO-GO/AMBIGUOUS)은 반드시 `design.md` §0 표의 사전 등록 임계값으로만 내린다.**

## 교차 검증 (cross-validation between agents)

중요한 결정은 단일 모델의 판단에 의존하지 않는다.

- 다음은 Claude↔Codex 교차 검증 대상이다: (1) 통계·판정 코드 구현, (2) GO/NO-GO 판정 직전의 수치 해석, (3) 설계 변경.
- 교차 검증은 **blind**로 한다: 상대 에이전트에게 자신의 결론을 먼저 알려주지 않고, 원자료(코드, 수치, 사전 등록 기준)만 제공한다.
- 불일치 항목은 어느 쪽 모델도 정답으로 간주하지 않고, 재계산·소스 확인으로 판정한다.
- 교차 검증 리포트는 `outputs/cross_checks/`에 저장한다.

## 에이전트 작업 지침

새 코드를 작성할 때는 먼저 다음을 확인한다.

- 현재 `pyproject.toml`의 dependency group 구조
- 기존 `configs/`의 설정 패턴
- 기존 `src/audio_safety/` 모듈 경계
- 기존 `scripts/` 실행 방식
- 캐시와 output directory 처리 방식

수정 시 지켜야 할 점:

- 단기 실험을 위해 하드코딩하지 않는다.
- 새 설정이 필요하면 config schema(`src/audio_safety/config/schema.py`)와 config 파일에 함께 추가한다.
- 새 의존성이 필요하면 적절한 dependency group에 추가한다.
- GPU 서버에서 `uv sync` 기반으로 재현 가능해야 한다.
- 실행 예시는 executable script 형식(`./scripts/<name>.py ...`)으로 작성한다.
- TTS 렌더링은 `scripts/cosyvoice2_tts.py --batch-jsonl` 경로를 우선 사용한다. CosyVoice2 모델을 한 번만 GPU에 로드해 pending wav를 연속 생성하는 것이 A40 단일 GPU에서의 기본 fast path이다.
- `/workspace` 외부의 개인 경로나 임의 절대 경로를 문서/코드에 남기지 않는다.
- 통계·판정 로직을 수정하면 반드시 `tests/`의 해당 테스트도 갱신하고 `uv run pytest`로 확인한다.

## 우선순위

애매한 선택이 있을 때는 다음 순서를 따른다.

1. GPU 서버에서 재현 가능한 실행
2. 사전 등록 기준을 지키는 실험 관리 (설정 기반, append-only 로그)
3. 오픈소스 프로젝트로 읽기 쉬운 구조
4. 빠른 임시 구현

빠른 임시 구현이 필요하더라도, 나중에 config와 모듈 구조로 옮기기 쉬운 형태로 작성한다.
