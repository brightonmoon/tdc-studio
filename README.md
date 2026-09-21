# TDC-Studio

> **Modular MLOps Studio for Therapeutics Data Commons (TDC)**  
> 데이터 로딩, 신경망 아키텍처 선택, Optuna 하이퍼파라미터 최적화, W&B 실험 추적, Google Colab 클라우드 GPU 오케스트레이션 및 FastAPI/Docker 서빙을 느슨하게 결합(Loose Coupling)한 바이오 MLOps 플랫폼입니다.

---

## 1. 전체 아키텍처 (System Architecture)

```text
+-----------------------------------------------------------------------------------+
|                            [Local Developer Workstation]                          |
|  - Fast Iteration via UV (Python 3.11 Standard)                                   |
|  - Static Analysis (Ruff) & Zero-Training Unit Tests (Pytest + Mock/Toy Batches)  |
|  - CLI Trigger: `tdc-studio remote run --gpu a100 --config configs/config.yaml`   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          | (Google Colab CLI: google-colab-cli)
                                          v
+-----------------------------------------------------------------------------------+
|                        [Cloud GPU Worker: Google Colab CLI]                       |
|  - Ephemeral VM Provisioning: T4 / L4 / A100 / H100                               |
|  - Auto-bootstrap: `uv sync --extra tdc` on Linux                                 |
|  - Heavy Workloads: TDC Cache -> Optuna Sweep / Distributed Training              |
|  - Monitoring: Weights & Biases (Realtime Runs, Sweeps, Model Artifacts)          |
|  - Auto Tear-down: Job completion immediately stops/releases runtime              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          | (Model Checkpoint / Artifact Sync)
                                          v
+-----------------------------------------------------------------------------------+
|                            [Production Serving Container]                         |
|  - Lightweight Docker (Python 3.11-slim + libgomp1 + UV multistage)               |
|  - End-to-End InferencePipeline: SMILES -> RDKit Featurizer -> Tensor Engine      |
|  - FastAPI Microservice (Lifespan + /healthz + Async ThreadPool Offloading)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. 4대 프로젝트 관리 체크리스트

TDC-Studio는 실험의 재현성과 개발 속도를 보장하기 위해 다음 4대 체크리스트를 준수합니다:

1. **설치 및 환경:** 런타임은 Python 3.11 기반의 `uv sync --extra dev`로 관리. ([가이드](docs/01_environment_setup.md))
2. **품질 검증 (커밋 전 필수):** `uv run ruff check` 및 `uv run pytest -v` 통과 의무화. ([기여 규칙](CONTRIBUTING.md))
3. **학습 실행:** 로컬은 `--dry-run` 1초 점검, 실제 학습은 `tdc-studio remote run --gpu a100`으로 클라우드 위임. ([가이드](docs/03_training_and_hpo.md))
4. **실험 기록:** `configs/tracking/`에 W&B 프로젝트명을 분리하여 Run 및 Artifact 추적. ([설정 가이드](configs/README.md))

---

## 3. 문서 허브 (Documentation Index)

| 문서 | 설명 |
| :--- | :--- |
| **[01. 환경 설정 및 설치](docs/01_environment_setup.md)** | Python 3.11 고정 이유, UV 가상환경, Colab CLI 인증 가이드 |
| **[02. 알고리즘 선택 가이드](docs/02_algorithm_selection.md)** | ADMET/DTA 태스크별 의사결정 트리 및 추천 모델 백본 매트릭스 |
| **[03. 학습 및 HPO 운영](docs/03_training_and_hpo.md)** | 로컬 1-Step 드라이런, Colab GPU 위임 및 W&B 실시간 추적 |
| **[04. 컴포넌트 확장 가이드](docs/04_extending_components.md)** | `@MODELS`, `@DATASETS` 데코레이터를 이용한 신규 모델/데이터 추가법 |
| **[05. 서빙 및 컨테이너 배포](docs/05_serving_deployment.md)** | FastAPI API 명세서 및 Docker Compose 프로덕션 배포 |
| **[설정(Configs) 명세](configs/README.md)** | YAML 선언적 설정 파일 구조 및 파라미터 규격 |
| **[기여 및 품질 관리](CONTRIBUTING.md)** | 커밋 전 필수 검증 및 Zero-training 테스트 작성 원칙 |

---

## 4. 퀵스타트 (Quickstart)

### (1) 환경 설치
```bash
git clone https://github.com/brightonmoon/tdc-studio.git
cd tdc-studio
uv sync --extra dev
```

### (2) 무결성 검증 (린트 & 단위 테스트)
```bash
# 린트 검사
uv run ruff check tdc_studio tests

# 단위 테스트 (Zero-training 원칙, 15초 이내 완료)
uv run pytest -v
```

### (3) 주요 실행 명령어
```bash
# 1. 로컬 1-Step 무결성 사전 점검 (Dry-run)
uv run tdc-studio train --config configs/config.yaml --dry-run
uv run tdc-studio tune --config configs/config.yaml --n-trials 2 --dry-run

# 2. 클라우드 GPU(Google Colab)로 학습 위임
uv run tdc-studio remote run --gpu a100 --command "tdc-studio tune --config configs/config.yaml --n-trials 50"

# 3. 일회성 Colab Jupyter Notebook (.ipynb) 추출
uv run tdc-studio remote export-notebook --output tdc_colab_runner.ipynb

# 4. FastAPI 서빙 서버 구동 (Swagger: http://localhost:8000/docs)
uv run tdc-studio serve --port 8000
```
