# AudioSafety Shared Project Guide

이 문서는 Claude, Codex 및 기타 코드 에이전트가 이 프로젝트를 수정하거나 확장하기 전에 먼저 읽어야 하는 공통 기준 문서이다. `AGENTS.md`와 `CLAUDE.md`는 이 파일을 참조한다. 실험 코드는 GPU 서버 실행, 재현성, 배포, 오픈소스 공개 가능성을 전제로 작성한다.

## 기본 원칙

- 이 프로젝트는 Python 기반 실험 프로젝트이다.
- 패키지와 실행 환경 관리는 `uv`를 기준으로 한다.
- 실행 환경은 GPU 서버를 기준으로 한다.
- GPU 서버에서 `uv sync`만 실행해도 필요한 의존성이 설치될 수 있도록 `pyproject.toml`과 `uv.lock`을 관리한다.
- 코드, 설정, 스크립트, 실험 산출물의 역할을 명확히 분리한다.
- 하드코딩된 경로, 모델명, 데이터셋명, 하이퍼파라미터, 캐시 경로를 피하고 설정 파일 또는 환경 변수로 주입한다.

## 의존성 관리

`pyproject.toml`은 의존성을 역할별로 분리해서 관리한다.

- 기본 의존성: 프로젝트 실행에 필요한 최소 런타임 패키지
- 개발 의존성: formatter, linter, test runner, type checker 등
- 실험 의존성: 특정 실험에서만 필요한 패키지
- GPU 의존성: CUDA, PyTorch, audio/model inference 관련 패키지

권장 방향:

- `uv sync`로 기본 환경을 재현할 수 있게 한다.
- 개발 도구는 별도 dependency group으로 둔다.
- GPU 관련 패키지는 별도 group 또는 optional extra로 분리해 설치 단위를 명확히 한다.
- 새 패키지를 추가할 때는 왜 필요한지, 어느 group에 들어가야 하는지 먼저 판단한다.
- `uv.lock`은 재현성을 위해 커밋 대상이다.
- 임시로 `pip install`한 상태를 코드의 전제로 삼지 않는다.

예상 사용 흐름:

```bash
uv sync
uv sync --group dev
uv sync --group gpu
```

실제 group 이름은 `pyproject.toml`을 만들 때 프로젝트 상황에 맞게 확정한다.

## 프로젝트 구조

오픈소스 Python 프로젝트에 가까운 구조를 유지한다.

권장 구조:

```text
audio_safety/
  AGENTS.md
  README.md
  pyproject.toml
  uv.lock
  configs/
    models/
    experiments/
    paths/
  src/
    audio_safety/
      __init__.py
      config/
      data/
      models/
      pipelines/
      evaluation/
      utils/
  scripts/
    train.py
    evaluate.py
    run_experiment.py
  tests/
  data/
    README.md
  outputs/
    README.md
```

역할 기준:

- `src/audio_safety/`: 재사용 가능한 라이브러리 코드
- `scripts/`: CLI 실행 진입점. 복잡한 로직은 `src/`로 위임
- `configs/`: 모델, 데이터셋, 실험, 경로, 런타임 설정
- `tests/`: 핵심 설정 로딩, 경로 처리, 실험 파이프라인 단위 테스트
- `data/`: 원본 대용량 데이터는 직접 커밋하지 않고 설명 파일만 둔다
- `outputs/`: 체크포인트, 로그, 평가 결과 등 생성물. 기본적으로 git 추적 제외

## 설정 관리

모델명, 데이터셋명, 경로, batch size, seed, decoding 설정, 평가 옵션 등은 코드에 직접 박지 않는다.

권장 방식:

- 설정은 `configs/` 아래 YAML 또는 TOML로 관리한다.
- Python 코드에서는 설정 로더를 통해 구조화된 config 객체로 읽는다.
- 기본값은 한 곳에서만 정의한다.
- 실험별 설정은 base config를 확장하거나 override하는 방식으로 둔다.
- CLI 인자는 config 파일 경로와 필요한 override만 받도록 단순하게 유지한다.
- 랜덤 seed, output directory, run name은 모든 실험에서 명시적으로 기록한다.

예시 방향:

```bash
uv run python scripts/run_experiment.py --config configs/experiments/refusal_probe.yaml
```

## 경로와 캐시 정책

Hugging Face, PyTorch, datasets, model checkpoints, 기타 캐시 파일은 기본 home/root 경로에 의존하지 않는다.

이유:

- GPU 서버나 컨테이너 배포 환경에서는 실행 사용자가 root가 될 수 있다.
- 기본 캐시가 `/root/.cache`로 들어가면 재현성, 권한, 용량 관리가 어려워진다.
- 배포 시 `/workspace`가 persistent volume 또는 작업 디렉터리로 쓰일 가능성이 높다.

원칙:

- 캐시 루트는 환경 변수로 주입 가능해야 한다.
- GPU 서버에서는 가능하면 `/workspace` 하위에 캐시를 둔다.
- 코드 내부에서 `~/.cache`, `/root`, 개인 경로, 임의 절대 경로를 직접 참조하지 않는다.
- 모든 경로는 `pathlib.Path`로 다룬다.

권장 환경 변수:

```bash
export AUDIO_SAFETY_WORKSPACE=/workspace/audio_safety
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TORCH_HOME=/workspace/cache/torch
export XDG_CACHE_HOME=/workspace/cache
```

경로 기본값은 다음 우선순위로 결정한다.

1. 명시적 CLI 인자
2. 프로젝트 전용 환경 변수
3. config 파일 값
4. `/workspace` 하위의 cache, output, data directory

## 코드 작성 규칙

- 실험 로직과 실행 스크립트를 분리한다.
- `scripts/`에는 argument parsing과 pipeline 호출만 둔다.
- 모델 로딩, 데이터 로딩, 평가, 로깅은 각각 모듈화한다.
- 전역 상태와 import 시점의 무거운 초기화를 피한다.
- 파일 경로, 모델 ID, prompt template, metric 이름은 config에서 읽는다.
- 실험 결과는 JSONL, CSV, YAML, Markdown 등 후처리 가능한 형식으로 저장한다.
- 실패한 샘플, raw output, config snapshot을 함께 저장해 재현 가능하게 한다.
- 대용량 파일, 캐시, 체크포인트, 원본 데이터셋은 git에 커밋하지 않는다.

## 에이전트 작업 지침

새 코드를 작성할 때는 먼저 다음을 확인한다.

- 현재 `pyproject.toml`의 dependency group 구조
- 기존 `configs/`의 설정 패턴
- 기존 `src/audio_safety/` 모듈 경계
- 기존 `scripts/` 실행 방식
- 캐시와 output directory 처리 방식

수정 시 지켜야 할 점:

- 단기 실험을 위해 하드코딩하지 않는다.
- 새 설정이 필요하면 config schema 또는 config 파일에 추가한다.
- 새 의존성이 필요하면 적절한 dependency group에 추가한다.
- GPU 서버에서 `uv sync` 기반으로 재현 가능해야 한다.
- 실행 예시는 `uv run ...` 형식으로 작성한다.
- `/workspace` 외부의 개인 경로나 임의 절대 경로를 문서/코드에 남기지 않는다.

## 우선순위

애매한 선택이 있을 때는 다음 순서를 따른다.

1. GPU 서버에서 재현 가능한 실행
2. 설정 기반 실험 관리
3. 오픈소스 프로젝트로 읽기 쉬운 구조
4. 빠른 임시 구현

빠른 임시 구현이 필요하더라도, 나중에 config와 모듈 구조로 옮기기 쉬운 형태로 작성한다.
