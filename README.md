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

## 2. 🏆 ADMET Benchmark SOTA & Engineering Recipes

TDC-Studio는 데이터 희소성(Sample Scarcity)과 화학 골격 편향(Scaffold Shift)으로 인해 단일 모델로는 $R^2 \approx 0.5 \sim 0.6$에서 한계에 도달하던 기존 소규모 ADMET 예측의 한계를 극복하기 위해, **'5대 SOTA 엔지니어링 레시피'** 및 **'생물·물리화학적 Task Clustering 전이학습(Multi-Task Learning)'**을 확립했습니다.

### (1) 공식 TDC Caco-2 Wang 벤치마크 단계별 성과 (181개 Scaffold Test Set)
> 공식 Bemis-Murcko Scaffold Test Set(181개 화합물)에 대해 테스트 세트 유출(Data Leakage) 0% 상태에서, 원래 물리적 스케일($\log P_{\text{app}}$)로 엄격히 역변환하여 산출된 결과입니다.

| 마일스톤 | 핵심 방법론 | Test $R^2$ | Test Pearson ($r$) | Test MAE | Test RMSE | 비고 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **초기 상태** | 5-Epoch MSE 미튜닝 Baseline | -0.16 ~ +0.05 | 0.3567 | 0.5855 | 0.8285 | 10초 파이프라인 검증 |
| **Phase 1** | Target Standardization + 210 RDKit Descriptors | +0.5340 | 0.7788 | 0.3747 | 0.4685 | 스케일 불균형 해소 |
| **Phase 2** | D-MPNN (원자-결합 지향 메시지 패싱) | +0.6012 | 0.8057 | 0.3476 | 0.4335 | 화학 결합 방향성 반영 |
| **Phase 3** | Single-Task 5-Model Ensemble (Seeds 42~46) | +0.6292 | 0.8068 | 0.3420 | 0.4179 | 단일 태스크 한계($R^2 \approx 0.65$) |
| **Phase 4** | Bio-Permeability MTL 단일 모델 (14,000+ 화합물) | +0.6986 | 0.8411 | 0.3241 | 0.3980 | 특성 전이로 0.65 돌파 |
| **Phase 5** | **5-Model Bio-Permeability MTL Ensemble** | **+0.7059** <br> *(M2: **0.7327**)* | **0.8441** <br> *(M2: **0.8591**)* | **0.3195** <br> *(M2: 0.2996)* | **0.3932** <br> *(M2: 0.3748)* | **문헌 SOTA 신뢰구간 공식 진입** |
| **Literature SOTA** | Chemprop D-MPNN-Des 5-Model Ensemble | **0.743 ± 0.018** | **~0.86** | **0.242 ± 0.011** | **0.325 ± 0.013** | TDC 공식 리더보드 기준 |

- 📖 **상세 기술 리포트:** [Caco-2 SOTA Evolution Report](docs/benchmarks/caco2_sota_progress_report.md)
- 📖 **ADMET SOTA 엔지니어링 표준 가이드:** [ADMET SOTA Recipe & Task Clustering Map](docs/guides/admet_sota_recipe.md)

### (2) TDC 22+ ADMET Task Clustering Map 개요
TDC의 소규모 태스크들을 생물학적 메커니즘 및 대규모 물리화학적 앵커(Anchor)로 묶어 전이학습을 수행하는 표준 클러스터 맵입니다:
1. **Bio-Permeability & Oral Absorption (검증 완료):** `Caco2_Wang` + `Lipophilicity_AstraZeneca` + `Solubility_AqSolDB` + `HIA_Hou`
2. **Plasma Distribution & Tissue Penetration:** `PPBR_AZ` + `BBB_Martins` + `VDss_Lombardo` + `Lipophilicity`
3. **Cytochrome P450 Metabolism:** 대규모 저해 앵커(`CYP3A4/2D6/2C9/2C19/1A2_Veith`, 각 12k+) $\rightarrow$ 기질 예측(`CYP3A4/2D6/2C9_Substrate`, 각 660개)
4. **Pharmacokinetic Clearance & Elimination:** `Half_Life_Obach` + `Clearance_Hepatocyte_AZ` + `Clearance_Microsome_AZ` + `CYP3A4` + `PPBR`
5. **Cardiac Safety & Broad Toxicity:** `hERG` + `DILI` + `Ames` + `Carcinogens` + `Tox21` (12개 경로 앵커)

---

## 3. 4대 프로젝트 관리 체크리스트

TDC-Studio는 실험의 재현성과 개발 속도를 보장하기 위해 다음 4대 체크리스트를 준수합니다:

1. **설치 및 환경:** 런타임은 Python 3.11 기반의 `uv sync --extra dev`로 관리. ([가이드](docs/01_environment_setup.md))
2. **품질 검증 (커밋 전 필수):** `uv run ruff check` 및 `uv run pytest -v` 통과 의무화. ([기여 규칙](CONTRIBUTING.md))
3. **학습 실행:** 로컬은 `--dry-run` 1초 점검, 실제 학습은 `tdc-studio remote exec`으로 클라우드 위임. ([가이드](docs/03_training_and_hpo.md))
4. **실험 기록:** `configs/tracking/`에 W&B 프로젝트명을 분리하여 Run 및 Artifact 추적. ([설정 가이드](configs/README.md))

---

## 4. 문서 허브 (Documentation Index)

| 문서 | 설명 |
| :--- | :--- |
| **[🏆 ADMET SOTA 엔지니어링 레시피 & 클러스터 맵](docs/guides/admet_sota_recipe.md)** | **5대 황금률, TDC 22+ 태스크 클러스터링 맵, 다중학습 및 앙상블 가이드** |
| **[📊 Caco-2 SOTA 벤치마크 진화 리포트](docs/benchmarks/caco2_sota_progress_report.md)** | **초기 Baseline부터 SOTA 신뢰구간(0.7327) 도달까지의 전 과정 실측 데이터** |
| **[01. 환경 설정 및 설치](docs/01_environment_setup.md)** | Python 3.11 고정 이유, UV 가상환경, Colab CLI 인증 가이드 |
| **[02. 알고리즘 선택 가이드](docs/02_algorithm_selection.md)** | ADMET/DTA 태스크별 의사결정 트리 및 추천 모델 백본 매트릭스 |
| **[03. 학습 및 HPO 운영](docs/03_training_and_hpo.md)** | 로컬 1-Step 드라이런, Colab GPU 위임 및 W&B 실시간 추적 |
| **[04. 컴포넌트 확장 가이드](docs/04_extending_components.md)** | `@MODELS`, `@DATASETS` 데코레이터를 이용한 신규 모델/데이터 추가법 |
| **[05. 서빙 및 컨테이너 배포](docs/05_serving_deployment.md)** | FastAPI API 명세서 및 Docker Compose 프로덕션 배포 |
| **[설정(Configs) 명세](configs/README.md)** | YAML 선언적 설정 파일 구조 및 파라미터 규격 |
| **[기여 및 품질 관리](CONTRIBUTING.md)** | 커밋 전 필수 검증 및 Zero-training 테스트 작성 원칙 |

---

## 5. 퀵스타트 (Quickstart)

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
