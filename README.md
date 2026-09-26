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
2. **Plasma Distribution & Tissue Penetration (검증 완료):** `PPBR_AZ` + `BBB_Martins` + `VDss_Lombardo` + `Lipophilicity`
3. **Cytochrome P450 Metabolism (검증 완료):** 대규모 저해 앵커(`CYP3A4/2D6/2C9/2C19/1A2_Veith`, 각 12k+) $\rightarrow$ 기질 전이학습(`CYP3A4/2D6/2C9_Substrate`, 각 660개)
4. **Pharmacokinetic Clearance & Elimination (준비 완료):** `Half_Life_Obach` + `Clearance_Hepatocyte_AZ` + `Clearance_Microsome_AZ` + `CYP3A4` + `PPBR`
5. **Cardiac Safety & Broad Toxicity:** `hERG` + `DILI` + `Ames` + `Carcinogens` + `Tox21` (12개 경로 앵커)

### (3) 공식 TDC Cluster 3: CYP450 8-Head 대사 매트릭스 벤치마크 성과 (Scaffold Test Set)
> Veith et al. 5대 저해 효소(60,000+건)로 학습된 DMPNN 백본 지식을 Carbon-Mangels 3대 기질 희소 과제(각 660건)로 전이시키는 **2단계 전이학습(Two-Stage Protocol: Staged Unfreezing + Cosine Annealing)**을 적용하여 도출된 공식 벤치마크 결과입니다.

| 태스크 분류 | 엔드포인트 명칭 | 데이터 규모 | Test AUROC | Accuracy (ACC) | Balanced ACC | Matthews Corr (MCC) | F1-Score | 상태 / 성과 판정 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CYP1A2** | `cyp1a2_veith` | 12,579 | **`0.9194`** | **84.3%** | 84.1% | **`+0.686`** | 0.827 | **SOTA 달성 (AUC > 0.90 돌파)** |
| **CYP2C9** | `cyp2c9_veith` | 12,092 | **`0.8827`** | **81.3%** | 77.9% | **`+0.578`** | 0.713 | **매우 우수 (안정적 수렴)** |
| **CYP3A4** | `cyp3a4_veith` | 12,328 | **`0.8784`** | **78.7%** | 77.6% | **`+0.566`** | 0.739 | **Primary 벤치마크 앵커 완결** |
| **CYP2C19**| `cyp2c19_veith`| 12,665 | **`0.8774`** | **79.1%** | 78.8% | **`+0.586`** | 0.766 | **우수 (안정적 수렴)** |
| **CYP2D6** | `cyp2d6_veith` | 13,130 | **`0.8323`** | **84.1%** | 69.1% | **`+0.470`** | 0.536 | **양호 (클래스 불균형 극복)** |
| **CYP2D6 기질** | `cyp2d6_substrate` | 667 | **`0.9928`** | **97.7%** | 98.3% | **`+0.952`** | **0.968** | **준-완전 예측 (전이학습 극대화)** |
| **CYP2C9 기질** | `cyp2c9_substrate` | 669 | **`0.9638`** | **97.0%** | 95.6% | **`+0.912`** | **0.931** | **압도적 SOTA (+31.9% 도약)** |
| **CYP3A4 기질** | `cyp3a4_substrate` | 670 | **`0.9130`** | **90.2%** | 89.4% | **`+0.800`** | **0.918** | **목표 초과 달성 (+23.4% 폭증)** |
| **기질 3종 평균**| **Substrates Transfer** | **2,000+** | **`0.9565`** | **95.0%** | **94.4%** | **`+0.888`** | **0.939** | **Cluster 3 태스크 공식 종결** |

- 🔗 **W&B 공식 런 (Stage 1 Joint Pretraining):** [wandb.ai/tdc-studio/tdc-learning/runs/f9v2gcsr](https://wandb.ai/tdc-studio/tdc-learning/runs/f9v2gcsr)
- 🔗 **W&B 공식 런 (Stage 2 Substrate Transfer):** [wandb.ai/tdc-studio/tdc-learning/runs/3qihar8z](https://wandb.ai/tdc-studio/tdc-learning/runs/3qihar8z)
- 💾 **공식 체크포인트:** `models/checkpoint_cyp450_stage2/best_model.pt` (21.5MB)

---

## 3. 🧬 DTI/DTA Phase B — Foundation Model 기반 Drug-Target 결합력 예측

TDC-Studio는 ADMET 예측을 넘어, **Drug-Target Affinity (DTA)** 예측을 위한 대형 사전학습(Foundation) 모델 파이프라인을 제공합니다. Phase B는 HuggingFace 기반의 ChemBERTa(화합물) + ESM-2(단백질) 파운데이션 인코더를 결합하여 **BindingDB Kd Cold-Drug 벤치마크에서 CI ≥ 0.70을 달성**합니다.

### (1) Phase B 아키텍처

```text
                    ┌─────────────────────────┐
    SMILES ─────▶   │  ChemBERTaEncoder        │  ─────┐
                    │  DeepChem/ChemBERTa-77M  │       │   ┌──────────────────────────┐
                    │  384d → 256d Projection  │       ├─▶ │  BilinearAttentionFusion  │ ─▶ pKd Score
                    └─────────────────────────┘       │   └──────────────────────────┘
                    ┌─────────────────────────┐       │
    AA Seq ──────▶  │  ESM2Encoder             │  ─────┘
                    │  facebook/esm2_t12_35M   │
                    │  480d → 256d Projection  │
                    │  + In-Memory Cache       │
                    └─────────────────────────┘
```

| 컴포넌트 | HuggingFace 모델 ID | 출력 차원 | 특이사항 |
|:---|:---|:---:|:---|
| **ChemBERTaEncoder** | `DeepChem/ChemBERTa-77M-MTR` | 256 | Masked Mean Pooling, `freeze()`/`unfreeze(n)` |
| **ESM2Encoder** | `facebook/esm2_t12_35M_UR50D` | 256 | max_length=1024, **고유 서열 인메모리 캐시** (15× 가속) |
| **BilinearAttentionFusion** | — | 1 | 약물·단백질 교차 어텐션 융합 후 pKd 회귀 |

> [!TIP] **YAML 2줄로 Phase A → B 업그레이드**: `drug_encoder` / `protein_encoder` 키만 교체하면 기존 Phase A GIN+CNN 파이프라인에서 Phase B 파운데이션 파이프라인으로 전환됩니다.

### (2) 🏆 BindingDB Kd Cold-Drug 벤치마크 결과

> 공식 TDC Cold-Drug Split (학습 약물과 완전히 다른 신약 후보 평가) 기준의 최종 테스트 결과입니다.

| Phase | 약물 인코더 | 단백질 인코더 | Cold-Drug CI ↑ | MSE ↓ | RMSE ↓ | Pearson r ↑ |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **Phase A** | GINE (GNN) | ProteinCNN | — | — | — | — |
| **Phase B** | ChemBERTa-77M | ESM-2 35M | **0.7464 ✅** | **0.6609 ✅** | 0.8130 | 0.6508 |
| **목표 기준** | — | — | ≥ 0.70 | ≤ 0.75 | — | — |

- 📄 **상세 결과**: [`models/dti/phase_b/benchmark_summary.yaml`](models/dti/phase_b/benchmark_summary.yaml)
- ⚙️ **학습 설정**: [`configs/config_dti_phase_b.yaml`](configs/config_dti_phase_b.yaml)
- 🚀 **Colab 학습 스크립트**: [`deploy/train_dti_phase_b.py`](deploy/train_dti_phase_b.py)

### (3) DTI/DTA Phase B 실행 가이드

#### 로컬 무결성 검증 (Dry-run)
```bash
# 모델 레지스트리, DataModule, forward/backward 통합 검증
uv run python deploy/dti_phase_b_dry_run.py
```

#### Colab GPU 학습 (세션 분리 권장)
```bash
# tdc-studio-dti 전용 세션으로 T4 GPU 학습 실행
# (ADMET 학습과 세션 분리: tdc-studio-admet / tdc-studio-dti)
uv run python -c "
from tdc_studio.remote.colab_runner import ColabRunner
runner = ColabRunner(session='tdc-studio-dti')
runner.run_script('deploy/train_dti_phase_b.py',
                  config='configs/config_dti_phase_b.yaml')
"
```

#### Phase A → Phase B YAML 전환 (핵심 2줄 변경)
```yaml
# configs/config_dti_phase_b.yaml (변경된 핵심 항목)
drug_encoder:
  name: chemberta_encoder      # ← Phase A: gine_model
  pretrained_model: DeepChem/ChemBERTa-77M-MTR

protein_encoder:
  name: esm2_encoder           # ← Phase A: protein_cnn
  pretrained_model: facebook/esm2_t12_35M_UR50D
  freeze_backbone: true
```

#### 단위 테스트 (Zero-training, 인터넷 없이 Mock 기반)
uv run --extra dev pytest tests/test_pretrained_encoders.py tests/test_serving_api.py tests/test_cross_attention_fusion.py -v
# 예상 출력: 17 passed
```

#### DTI 프로덕션 서빙 (/predict/dti)
```bash
# 1. 서빙 서버 구동 (DTI 모델 디렉토리 지정 시 자동 로딩)
MODEL_DIR=models/export/dti_phase_c uv run tdc-studio serve --port 8000

# 2. DTI 결합 친화도 추론 요청 (pKd 및 Kd nM 산출)
curl -X POST "http://localhost:8000/predict/dti" \
     -H "Content-Type: application/json" \
     -d '{
       "smiles": ["CC1=C(C(=O)N2CCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"],
       "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD"],
       "return_kd_nm": true
     }'
```
### (4) 브랜치 거버넌스

| 브랜치 | 목적 |
|:---|:---|
| `main` | Phase B 전체 코드 포함 (Fast-forward 병합 완료) |
| `dta/phase-b-foundation` | DTI/DTA Phase C 연구를 위한 전용 보존 브랜치 |
| `brightonmoon/ADMET` | ADMET 전용 개발 브랜치 |

---

## 4. 4대 프로젝트 관리 체크리스트

TDC-Studio는 실험의 재현성과 개발 속도를 보장하기 위해 다음 4대 체크리스트를 준수합니다:

1. **설치 및 환경:** 런타임은 Python 3.11 기반의 `uv sync --extra dev`로 관리. ([가이드](docs/01_environment_setup.md))
2. **품질 검증 (커밋 전 필수):** `uv run ruff check` 및 `uv run pytest -v` 통과 의무화. ([기여 규칙](CONTRIBUTING.md))
3. **학습 실행:** 로컬은 `--dry-run` 1초 점검, 실제 학습은 `tdc-studio remote exec`으로 클라우드 위임. ([가이드](docs/03_training_and_hpo.md))
4. **실험 기록:** `configs/tracking/`에 W&B 프로젝트명을 분리하여 Run 및 Artifact 추적. ([설정 가이드](configs/README.md))

---

## 4. 문서 허브 (Documentation Index)

| 문서 | 설명 |
| :--- | :--- |
| **[📋 학습 파이프라인 실행 체크리스트 & TODOLIST](docs/roadmaps/training_pipeline_todolist.md)** | **내일 즉시 시작 가능한 클러스터별 1-Epoch Smoke Test 및 실측 학습 로드맵** |
| **[🎯 ADMETlab 3.0 기반 TDC 목표 성능 규격서](docs/benchmarks/admetlab3_tdc_target_performance.md)** | **NAR 2024 부록 기준 22+ TDC 전 태스크별 목표 성능 수치(R², RMSE, MAE, AUC, ACC)** |
| **[📐 ADMET 5대 클러스터 모델 설계 청사진](docs/guides/admet_cluster_architectures.md)** | **Caco-2를 벤치마킹한 Lipo, PPBR/BBB/VDss, CYP450, Clearance, Tox 전용 모델 설계안** |
| **[🏆 ADMET SOTA 엔지니어링 레시피 & 클러스터 맵](docs/guides/admet_sota_recipe.md)** | **5대 황금률, TDC 22+ 태스크 클러스터링 맵, 다중학습 및 앙상블 가이드** |
| **[📊 PPBR 체내분포 SOTA & Tri-Hybrid 리포트](docs/benchmarks/ppbr_distribution_sota_progress_report.md)** | **Tri-Hybrid 스태커(R²=0.5412) 및 잔차 오차 진단, ChEMBL HSA 전이 실측 데이터** |
| **[📊 Caco-2 SOTA 벤치마크 진화 리포트](docs/benchmarks/caco2_sota_progress_report.md)** | **초기 Baseline부터 SOTA 신뢰구간(0.7327) 도달까지의 전 과정 실측 데이터** |
| **[🧬 DTI Phase B 실행 계획서](docs/dti/dti_phase_b_plan.md)** | **ChemBERTa + ESM-2 파운데이션 모델 설계 · 8단계 실행 로드맵 · 세션 거버넌스** |
| **[📊 DTI Phase B 벤치마크 결과](models/dti/phase_b/benchmark_summary.yaml)** | **BindingDB Kd Cold-Drug: CI=0.7464 ✅, MSE=0.6609 ✅ (목표 초과 달성)** |
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
