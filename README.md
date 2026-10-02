# TDC-Studio

> **Modular MLOps Studio for Therapeutics Data Commons (TDC)**  
> 신약 개발 후보물질의 **ADMET 25종 다중작업 예측**, **14-구획 PBPK 생체역학 시뮬레이션**, **DTI 파운데이션 모델 결합 친화도 예측**, 그리고 **역합성(Retrosynthesis) AI 탐색**을 엔드투엔드로 통합한 바이오 MLOps 플랫폼입니다.

---

## 1. 핵심 기능 및 역량 (Core Capabilities)

- 🧪 **ADMET 25종 풀 패널 예측**: 흡수(6), 체내분포(3), CYP450 대사 8-Head, 배설(3), 핵심 및 확장 독성(5)을 포괄하는 SOTA 멀티태스크 신경망.
- 🧮 **14-구획 ODE PBPK 생체역학 시뮬레이터**: $V_{dss}$, $t_{1/2}$, $f_u$, $CL_{\text{total}}$, $CL_H$ 및 간 추출비($E_H$)를 통합하여 혈중 농도-시간 곡선($C_{\text{plasma}} - t$)을 실시간 모사.
- 🧬 **DTI/DTA Foundation Pipeline**: ChemBERTa(화합물) + ESM-2(단백질)와 Bilinear Cross-Attention을 결합하여 결합 친화도($pK_d$, $K_d \text{ nM}$) 예측 (BindingDB Cold-Drug CI = 0.7464 달성).
- 🔬 **역합성(Retrosynthesis) AI**: 타깃 분자의 템플릿 기반 및 딥러닝 단일/다단계 합성 경로 역추적 및 RDKit 화학 정규화 검증.
- ⚡ **하이브리드 MLOps 파이프라인**: 로컬 워크스테이션(UV)의 1-Step 무결성 점검 ↔ Google Colab CLI를 통한 원격 GPU 오케스트레이션 ↔ Docker/FastAPI 프로덕션 서빙.

---

## 2. 전체 아키텍처 (System Architecture)

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

## 3. 코드베이스 구조 (Repository Layout)

```text
tdc-studio/
├── tdc_studio/                  # 핵심 라이브러리 패키지
│   ├── cli/                     # CLI 커맨드 엔트리포인트 (predict, ti, ui, train, remote 등)
│   ├── core/                    # 공통 레지스트리(@MODELS, @DATASETS), 기본 추상 클래스
│   ├── data/                    # TDC DataModule, RDKit Featurizer, Graph Batcher
│   ├── models/                  # D-MPNN, GIN, ChemBERTa, ESM-2, Hybrid Stacker 모델 정의
│   ├── pbpk/                    # 14-구획 ODE 생체약동학 시뮬레이션 수식 및 파이프라인
│   ├── remote/                  # Google Colab CLI 래퍼 및 원격 GPU 실행기
│   ├── retrosynthesis/          # 템플릿 기반 및 딥러닝 역합성 경로 탐색 엔진
│   ├── serving/                 # FastAPI 애플리케이션 및 추론 파이프라인
│   └── ui/                      # Streamlit/FastAPI 웹 대시보드 UI 인터페이스
├── configs/                     # 선언적 YAML 설정 (ADMET 클러스터, DTI, 서빙, 트래킹)
├── deploy/                      # Colab 실행 스크립트, Dockerfile, docker-compose.yml
├── docs/                        # 시스템 상세 가이드, 아키텍처, 벤치마크 아카이브
├── models/                      # 학습된 체크포인트 (.pt), 스케일러 (.pkl), 모델 아티팩트
├── scripts/                     # W&B 모델 동기화, 배치 벤치마크 유틸리티 스크립트
└── tests/                       # 180+ 회귀 및 단위 테스트 (Zero-training 원칙)
```

---

## 4. 환경 설정 및 무결성 검증 (Setup & Verification)

### (1) 환경 요구사항
- **Python**: `3.11` (TDC 데이터셋 라이브러리 및 PyTorch Geometric C++ 익스텐션 바이너리 호환 필수)
- **패키지 매니저**: [uv](https://github.com/astral-sh/uv) (초고속 가상환경 격리 및 의존성 동기화)

### (2) 로컬 개발 환경 동기화
```bash
git clone https://github.com/brightonmoon/tdc-studio.git
cd tdc-studio

# 개발 의존성 포함 가상환경 자동 동기화
uv sync --extra dev
```

### (3) 코드 무결성 검증 (린트 & 단위 테스트)
```bash
# 1. Ruff 정적 분석 검사
uv run ruff check tdc_studio tests

# 2. 단위 테스트 실행 (Zero-training Mock/Toy 배치 기반으로 15초 이내 완료)
uv run pytest -v
```

---

## 5. 빠른 시작 및 핵심 사용법 (Quickstart & Usage)

### (1) ADMET & PBPK 일괄 예측 CLI
SMILES 입력 시 25종 ADMET 지표와 14-구획 PBPK 파라미터를 일괄 산출합니다:
```bash
uv run tdc-studio predict "CC(=O)Oc1ccccc1C(=O)O"
```

### (2) 치료지수(TI) 및 안전역 평가 CLI
타깃 결합력($K_d$)과 hERG 심장 독성 예측값을 연계하여 치료 안전역을 판정합니다:
```bash
uv run tdc-studio ti "CC(=O)Oc1ccccc1C(=O)O" --kd 10.0
```

### (3) 대화형 웹 대시보드 UI 구동
분자 구조 렌더링, ADMET 레이더 차트, PBPK 혈중 농도 시뮬레이션 곡선을 웹에서 확인합니다:
```bash
uv run tdc-studio ui --port 8000
# 접속: http://localhost:8000/
```

### (4) 로컬 무결성 1-Step 점검 (Dry-run)
실제 대용량 학습 전 1-Batch 포워드/백워드 무결성을 즉시 검증합니다:
```bash
uv run tdc-studio train --config configs/config.yaml --dry-run
uv run tdc-studio tune --config configs/config.yaml --n-trials 2 --dry-run
```

### (5) Google Colab 클라우드 GPU 원격 학습 위임
로컬 리소스 소모 없이 Google Colab GPU 인스턴스(T4, A100 등)로 학습을 위임합니다:
```bash
# A100 GPU 할당 및 원격 실행
uv run tdc-studio remote run --gpu a100 --command "tdc-studio train --config configs/admet/caco2.yaml"

# 일회성 Colab Jupyter Notebook (.ipynb) 파일로 내보내기
uv run tdc-studio remote export-notebook --output tdc_colab_runner.ipynb
```

### (6) 프로덕션 REST API 서빙
```bash
# 1. FastAPI 서버 시작
uv run tdc-studio serve --port 8000

# 2. PBPK 생체역학 파라미터 예측 쿼리
curl -X POST "http://localhost:8000/predict/pbpk" \
     -H "Content-Type: application/json" \
     -d '{"smiles": ["CC(=O)NC1=CC=C(O)C=C1"]}'

# 3. DTI 약물-표적 단백질 결합 친화도(pKd, Kd) 예측 쿼리
curl -X POST "http://localhost:8000/predict/dti" \
     -H "Content-Type: application/json" \
     -d '{
       "smiles": ["CC1=C(C(=O)N2CCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"],
       "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD"],
       "return_kd_nm": true
     }'

# 4. 헬스체크
curl "http://localhost:8000/healthz"
```

### (7) W&B Model Registry 자동 동기화 & Docker 실행
```bash
# 학습된 클라우드 체크포인트를 로컬 models/export/로 자동 인덱싱 및 동기화
uv run python scripts/sync_wandb_models.py

# 프로덕션 서빙 컨테이너 구동
docker compose -f deploy/docker-compose.yml up -d
```

---

## 6. 에이전트 & 엔지니어링 개발 지침 (Agent Guidelines)

TDC-Studio 기여 및 코드 수정 시 반드시 준수해야 하는 5대 원칙입니다:

1. **Zero-Training 단위 테스트 원칙**:
   - 단위 테스트에서 실제 대용량 데이터셋 다운로드나 긴 에포크 학습을 수행하지 않습니다.
   - Mocking 및 `toy_batch`를 활용하여 15초 이내에 모든 회귀 테스트(`pytest`)가 통과되도록 유지합니다.
2. **안전한 모델 직렬화 (`weights_only=True`)**:
   - 모델 가중치 역직렬화 시 임의 코드 실행 취약점을 방지하기 위해 `torch.load(..., weights_only=True)`를 강제합니다.
3. **화학적 동등성 및 정규화 (Canonicalization)**:
   - 분자 구조 비교 및 역합성 검증 시 RDKit `Chem.MolToSmiles(canonical=True)` 정규화를 필히 적용합니다.
4. **NaN 손실 방어 및 마스킹**:
   - 다중 작업(Multi-task) 학습 시 결측 레이블로 인한 기울기 왜곡을 방지하기 위해 `MaskedLoss`와 `zero_division=0` 가드를 유지합니다.
5. **Colab 세션 분리 거버넌스**:
   - GPU 리소스 경합 및 세션 충돌을 방지하기 위해 세션명을 분리합니다 (`tdc-studio-admet`, `tdc-studio-dti`, `tdc-studio-retro`).

---

## 7. 종합 문서 허브 (Documentation Index)

자세한 벤치마크 결과, 이론적 설계 배경 및 로드맵은 다음 문서들을 참조하십시오:

### 📊 벤치마크 및 실측 성과 아카이브
| 문서 | 설명 |
| :--- | :--- |
| **[🏆 ADMET 5대 클러스터 SOTA 벤치마크 아카이브](docs/benchmarks/admet_cluster_sota_archive.md)** | **Caco-2, PPBR/VDss/BBB, CYP450, Clearance, Safety 전 클러스터 실측 수치, W&B 런 및 체크포인트** |
| **[📊 Caco-2 SOTA 벤치마크 진화 리포트](docs/benchmarks/caco2_sota_progress_report.md)** | Baseline부터 SOTA 신뢰구간($R^2=0.7327$) 도달까지의 전 과정 실측 데이터 |
| **[📊 PPBR 체내분포 Tri-Hybrid 리포트](docs/benchmarks/ppbr_distribution_sota_progress_report.md)** | Tri-Hybrid Stacker($R^2=0.5412$) 및 ChEMBL HSA 전이학습 실측 데이터 |
| **[🎯 ADMETlab 3.0 대비 목표 성능 규격서](docs/benchmarks/admetlab3_tdc_target_performance.md)** | NAR 2024 부록 기준 22+ TDC 전 태스크별 목표 성능 수치 및 갭 분석 |
| **[🧬 DTI Phase B 벤치마크 결과 리포트](docs/dti/dti_phase_b_benchmark_report.md)** | BindingDB Kd Cold-Drug CI = 0.7464 달성 실측 성과 및 Phase B 구조 명세서 |

### 📐 설계 원리 및 도메인 가이드
| 문서 | 설명 |
| :--- | :--- |
| **[🧪 22+ ADMET 지표 선정 근거 및 PBPK 설계 철학](docs/guides/admet_selection_and_pbpk_rationale.md)** | **TDC 38종 중 22+(25)종 선별 이유, 비선정/통합 4대 사유 표, 14-구획 PBPK 메커니즘** |
| **[🏆 ADMET SOTA 엔지니어링 레시피 & 클러스터 맵](docs/guides/admet_sota_recipe.md)** | 5대 황금률, TDC 22+ 태스크 클러스터링 맵, 다중학습 및 앙상블 기법 가이드 |
| **[📐 ADMET 5대 클러스터 모델 설계 청사진](docs/guides/admet_cluster_architectures.md)** | 클러스터별 최적 백본 신경망 구조 및 피처 파이프라인 상세 설계안 |
| **[02. 알고리즘 선택 가이드](docs/02_algorithm_selection.md)** | ADMET/DTA 태스크별 의사결정 트리 및 추천 모델 백본 매트릭스 |

### 🛠️ 엔지니어링 및 운영 가이드
| 문서 | 설명 |
| :--- | :--- |
| **[01. 환경 설정 및 설치](docs/01_environment_setup.md)** | Python 3.11 고정 이유, UV 가상환경, Colab CLI 인증 가이드 |
| **[03. 학습 및 HPO 운영](docs/03_training_and_hpo.md)** | 로컬 1-Step 드라이런, Colab GPU 위임 및 W&B 실시간 추적 |
| **[04. 컴포넌트 확장 가이드](docs/04_extending_components.md)** | `@MODELS`, `@DATASETS` 데코레이터를 이용한 신규 모델/데이터 추가법 |
| **[05. 서빙 및 컨테이너 배포](docs/05_serving_deployment.md)** | FastAPI API 명세서 및 Docker Compose 프로덕션 배포 |
| **[설정(Configs) 명세](configs/README.md)** | YAML 선언적 설정 파일 구조 및 파라미터 규격 |
| **[기여 및 품질 관리 규칙](CONTRIBUTING.md)** | 커밋 전 필수 검증 및 Zero-training 테스트 작성 원칙 |

### 📋 프로젝트 로드맵 및 연구 계획
| 문서 | 설명 |
| :--- | :--- |
| **[📋 학습 파이프라인 실행 체크리스트 & TODOLIST](docs/roadmaps/training_pipeline_todolist.md)** | 클러스터별 1-Epoch Smoke Test 및 실측 학습 실행 로드맵 |
| **[🔬 역합성 마스터 작업 계획서](docs/retrosynthesis/retrosynthesis_master_work_plan.md)** | 템플릿 기반 및 딥러닝 역합성 모듈 구현 마스터 플랜 |
| **[🧬 DTI Phase C 고도화 계획서](docs/dti/dti_phase_c_advancement_plan.md)** | ESM-2 + ChemBERTa 결합 고도화 및 DTI 확장 계획 |
| **[🧭 차기 단계 종합 작업 계획서](docs/roadmaps/next_phase_work_plan.md)** | TDC-Studio 전 모듈 통합 및 프로덕션 릴리스 로드맵 |
