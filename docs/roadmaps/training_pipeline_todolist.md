# ADMET & DTI Multi-Task Training Pipeline Execution Checklist & TODOLIST

> **최종 갱신 일시:** 2026-09-26 18:30 (KST)  
> **W&B 추적 프로젝트:** `tdc-studio/tdc-learning`  
> **실행 환경 원칙:** **로컬 학습 지양, Google Colab Cloud GPU (`remote exec`) 기본 적용**

---

## 📅 오늘(2026-09-26) 작업 완료 요약 (Today's Executive Summary)

### 1. 체내 분포(Cluster 2: PPBR / BBB / VDss) SOTA 및 프로덕션 서빙 달성
- **PPBR SOTA**: Standalone Super-DMPNN ($R^2 = \mathbf{0.5754}$) 및 Tri-Hybrid Super-Stacker (Test $R^2 = \mathbf{0.5525}$, Spearman $\rho = \mathbf{0.7662}$) 달성.
- **VDss Lombardo SOTA**: $\log_{10}(\text{L/kg})$ 타겟 정식화 및 3D Conformer + Oie-Tozer $f_u$ 연계 모티프 Tri-Hybrid Stacker (Test $R^2 = \mathbf{0.5361}$, Spearman $\rho = \mathbf{0.7778}$) 완성.
- **FastAPI 서빙 연동**: `/predict` 및 `/predict/vdss` (임상 약동학 Low/Moderate/High 3단계 분류) 프로덕션 서빙 구현.

### 2. 대사 효소(Cluster 3: CYP450 8-Head Matrix) Two-Stage SOTA 달성
- **Stage 1 (Veith 5대 저해 효소 60,000+ 사전학습)**: 복합 Test ROC-AUC = **0.8784** (`cyp1a2` **0.9194**, `cyp2c9` **0.8827**).
- **Stage 2 (Carbon-Mangels 3대 기질 turnover 전이 미세조정)**: 기질 평균 ROC-AUC = **0.9565** (+22.6% 도약, `cyp2d6_sub` **0.9928**, `cyp2c9_sub` **0.9638**, `cyp3a4_sub` **0.9130**).

### 3. DTI / DTA Phase B & C 파운데이션 모델 결합 및 Cold-Drug SOTA 돌파
- **Phase B (ChemBERTa + ESM-2)**: 인메모리 임베딩 캐싱으로 30배 학습 가속 달성, Cold-Drug 벤치마크 **CI = 0.7464, MSE = 0.6609** 기록.
- **Phase C (CrossAttentionFusion + XAI)**: 양방향 교차 어텐션 헤드 도입으로 Cold-Drug **CI = 0.7659, MSE = 0.6214** 추가 경신.
- **DTI 프로덕션 서빙 연동**: FastAPI `/predict/dti` 엔드포인트 ($pK_d$ 및 $K_d$ nM 변환) 완성.

### 4. 코드베이스 병합 및 즉시 보완 작업(Quick Wins) 4종 완료
- **Q1**: DTI 추론 시 ChemBERTa 인코더 대상 불필요한 RDKit 분자 그래프 변환 스킵 (Latency 최적화).
- **Q2**: FastAPI 서버 ADMET 및 DTI 다중 모델 동시 서빙(`ADMET_MODEL_DIR`, `DTI_MODEL_DIR`, 자동 탐색).
- **Q3**: CrossAttention XAI Attention Map REST API(`/predict/dti`) 노출 (`return_attention` 옵션).
- **Q4**: 전체 91개 단위/통합 테스트 100% Pass 무결성 검증.


---

## 📌 1. 사전 점검 완료 항목 (Pre-flight Checklist - COMPLETE)

- [x] **1-1. 최신 커밋 상태 확인 및 브랜치 점검**
  - 브랜치: `brightonmoon/ADMET` 격리 워크트리 작업 진행 중 (DTA는 `feature/dti-phase-b-foundation` 분리 운용).
- [x] **1-2. Google Colab GPU 세션 연결 확인**
  - `uv run tdc-studio remote status` 명령 구현 및 세션 탐지 검증 완료.
  - 현재 활성 세션: `[tdc-studio-dti]` (DTA 전용 세션 동작 중 확인).
  - 다중 계정 체계(4개 계정: `munhyeongdo4@gmail.com` 등)를 통해 쿼터 충돌 없는 ADMET 독립 GPU 세션 분리 가능.
- [x] **1-3. W&B 프로젝트 연결 확인**
  - 프로젝트 URL: `https://wandb.ai/tdc-studio/tdc-learning` (사용자 `munhyoungdo` 연동 정상).
  - 최근 업로드된 ChEMBL HSA 5-태스크(`b0jd5xaz`) 수렴 확인 완료 (Train Loss -2.2459 안정 수렴, Val Loss 0.5434, Lipophilicity R²=0.7069).

---

## 🚀 2. 진행 결과 및 잔여 TODOLIST (Execution Results & Action Plan)

### [Track A] PPBR / Distribution 클러스터 SOTA R² 돌파 및 최종 패키징 (COMPLETE)
- [x] **Task A-1: 5-태스크 체크포인트 + Tri-Hybrid 통합 재평가 (Colab/Local GPU)**
  - ChEMBL HSA 5-태스크와 4-태스크를 융합한 **Standalone Super-DMPNN**: **Test R² = 0.5754**, **MAE = 5.58%**, **Pearson r = 0.7722**, **Spearman ρ = 0.7504** 달성!
- [x] **Task A-2: 저결합(< 70%) 집중 가중치(Step Asymmetric & Smooth KDE) 및 3D 입체 파라미터(Conformer PBF, Spherocity Index) 도입**
  - 18개 생물물리학적 모티프 + 6개 3D 형태 파라미터(PBF, Spherocity, Asphericity 등) + 저결합 역가중치 결합:
    - Tri-Hybrid Super-DMPNN + Step-Weighted GBDT: **Test R² = 0.5525**, **Spearman ρ = 0.7662**, **MAE = 6.12%**.
    - 저결합 화합물 오차 잔차 MAE: 23.37% -> 18.99%로 대폭 개선.
- [x] **Task A-3: Distribution 클러스터 SOTA 체크포인트 패키징 및 배포 (`models/export/ppbr_tri_hybrid_sota.pt`)**
  - 모델 아티팩트 배포 완료 (`ppbr_tri_hybrid_sota.pt`, `model.pt`, `config.json`, `export_manifest.json`, `training_meta.json`).
  - `api.py` 루트 모듈 및 `TriHybridInferencePipeline` 연동 검증 (FastAPI `/predict` 정상 200 OK, 5개 의약품 서빙 검증 완료).
- [x] **Task A-4: VDss Lombardo Log10 정식 정식화 및 Cluster 2 전체 파이프라인 완성**
  - 롱테일 이상치(최대 80.5~700.0 L/kg, 왜도 11.2)로 인한 선형 공간 손실 발산 해결: ADMETlab 3.0 문헌 표준 $\log_{10}(\text{L/kg})$ 타겟 변환 구현.
  - 실험 결과: 선형 GBDT($R^2 = 0.2837$, Pearson $0.5435$) $\to$ $\log_{10}$ GBDT($R^2 = 0.4973$, Pearson $0.7064$, Spearman $0.7229$, MAE $0.328$, RMSE $0.454$)로 $R^2$ +0.2136 대폭 상승.
  - `ADMETClusterDataModule`의 `prepare_data`, `_build_dataset` 및 `cli.py` 지표 리포팅/벤치마크 테이블에 `transform: log10` 완전 연동.
  - 70개 단위/통합 테스트 100% 통과 및 ruff 린트 무결성 검증 완료.
- [x] **Task A-5: VDss 앵커 5-태스크 DMPNN Graph 모델 원격 Colab GPU 학습**
  - 설정: `configs/config_vdss_graph.yaml` (Primary Task: `vdss_lombardo`, 225개 정식 벤치마크 테스트 세트)
  - 다중 계정 쿼터 격리: 대기 계정 `munhyeongdo4@gmail.com`으로 원격 GPU 실행 (DTA 세션 무결성 100% 유지)
  - W&B 실행: `xxw7u6ph` (Train Loss 1.3888 -> -2.2204, Best Val R² = **0.5184** 달성)
  - 최종 225개 테스트 세트 지표:
    - **VDss Lombardo**: Test $R^2 = \mathbf{0.4646}$, Pearson $r = \mathbf{0.6830}$, Spearman $\rho = \mathbf{0.7066}$, MAE = $0.363$, RMSE = $0.468$ ($\log_{10}\text{ L/kg}$)
    - **BBB Martins**: Test ROC-AUC = $\mathbf{0.9167}$ (ADMETlab 3.0 목표 $0.908$ 초과 달성)
    - **Lipophilicity**: Test $R^2 = \mathbf{0.7602}$, Pearson $r = \mathbf{0.9168}$, Spearman $\rho = \mathbf{0.9236}$ (문헌치 $0.650$ 대폭 초과)
    - **PPBR AZ**: Test $R^2 = \mathbf{0.6191}$, Pearson $r = \mathbf{0.7984}$
- [x] **Task A-6: VDss Lombardo SOTA Tri-Hybrid Stacker 구축 및 프로덕션 서빙 배포**
  - 3D Conformer + Extended PK(Oie-Tozer $f_u$ 연계 모티프) GBDT + 4-Layer D-MPNN Graph + ChemBERTa RidgeCV 앙상블 블렌딩.
  - 최적 볼록 조합 가중치($w_{\text{GBDT}}=54.3\%$, $w_{\text{DMPNN}}=37.4\%$, $w_{\text{ChemBERTa}}=8.4\%$) 도출.
  - 최종 225개 테스트 세트 SOTA 달성:
    - **Test $R^2 = \mathbf{0.5361}$** (단일 GBDT $0.4973$ 대비 $+0.0388$, 순수 D-MPNN $0.3499$ 대비 $+0.1862$).
    - **Spearman $\rho = \mathbf{0.7778}$**, **Pearson $r = \mathbf{0.7686}$**, **RMSE = $0.436$ $\log_{10}\text{ L/kg}$**, **MAE = $0.336$**.
  - 배포 아티팩트: `models/export/vdss_tri_hybrid_sota.pt` (17.1MB 자가수용형 단일 번들) 및 메타데이터 JSON 생성.
  - 서빙 연동: `api.py` 및 `tdc_studio/serving/`에 `/predict/vdss` 엔드포인트, 임상 약동학 3단계 분류(Low/Moderate/High) 로직 통합.
  - 테스트: 77개 테스트 100% 통과 (VDss 추론 및 API 서빙 테스트 6종 신규 추가).
  - 브랜치 머지: `brightonmoon/ADMET` $\to$ `main` 검수 및 머지 완료.

### [Track B] DTI / DTA Phase B (사전학습 파운데이션 모델 결합 - COMPLETE)
- [x] **Task B-1: Pretrained Protein & Compound Encoders 구현 (`tdc_studio/models/dti/pretrained_encoders.py`)**
  - Drug: ChemBERTa-77M-MTR (384차원)
  - Target: ESM-2 (`facebook/esm2_t6_8M_UR50D`, 320차원)
- [x] **Task B-2: BindingDB_Kd Cold-Drug 벤치마크 학습 실행 및 평가**
  - 인메모리 임베딩 캐싱을 통한 30배 가속화 달성.
  - Cold-Drug 벤치마크 평가: **CI = 0.7464**, **MSE = 0.6609** 달성 (목표 CI $\ge 0.70$, MSE $\le 0.75$ 초과 달성).

### [Track C] ADMET 타 클러스터 본격 학습 (Colab GPU)
- [ ] **Task C-1: Cluster 1 (Lipophilicity 앵커: Lipo + Sol + FreeSolv + Caco2) 정밀 학습**
  - 목표: Lipophilicity $R^2 \ge 0.74$, AqSolDB MAE $\le 0.70$.
- [x] **Task C-2: Cluster 3 (CYP450 8-Head Matrix) Two-Stage SOTA 학습 (COMPLETE)**
  - Stage 1 (Veith 5-Head 대규모 사전학습, Colab T4 GPU): 복합 Test ROC-AUC = **0.8784** (`cyp1a2` **0.9194**, `cyp2c9` **0.8827**, `cyp3a4` **0.8784**, `cyp2c19` **0.8774**, `cyp2d6` **0.8323**).
  - Stage 2 (Carbon-Mangels 3-Substrates Staged Unfreezing 전이 미세조정): 3대 기질 압도적 SOTA 달성 (`cyp2d6_substrate` **0.9928** / ACC 97.7% / MCC +0.952, `cyp2c9_substrate` **0.9638** / ACC 97.0% / MCC +0.912, `cyp3a4_substrate` **0.9130** / ACC 90.2% / MCC +0.800). 기질 평균 ROC-AUC = **0.9565** (+22.6% 도약).
  - 공식 W&B 런: `f9v2gcsr` (Stage 1), `3qihar8z` (Stage 2). 체크포인트: `models/checkpoint_cyp450_stage2/best_model.pt`.

### [Track D] DTI / DTA Phase C (Cross-Attention Fusion & 프로덕션 서빙 - COMPLETE)
- [x] **Task D-1: CrossAttentionFusion 모듈 및 XAI Attention Map 추출 구현 (`tdc_studio/models/dti/fusion.py`)**
  - Drug $\leftrightarrow$ Target 양방향 Multi-Head Cross-Attention 헤드 및 Per-pair Attention Weights 추출 로직 완성.
- [x] **Task D-2: Option A Head-Only 15 에포크 학습 수렴 및 벤치마크 평가**
  - Cold-Drug 벤치마크: **CI = 0.7659**, **MSE = 0.6214** (Phase B 대비 CI +0.0195, MSE -0.0395 추가 향상).
- [x] **Task D-3: DTI 프로덕션 서빙 파이프라인 연동 (`api.py`, `app.py`, `pipeline.py`)**
  - `/predict/dti` 엔드포인트 구현: SMILES + AA Sequence 입력 $\to$ $pK_d$ 및 $K_d$ (nM) 환산 및 잔기별 Attention weights 제공.

### [Track E] 즉시 보완 조치 (Quick Wins & Immediate Fixes - COMPLETE)
- [x] **Task E-1: DTI 추론 시 ChemBERTa 분자 그래프 변환 스킵 최적화**
  - `DTIInferencePipeline`에 `is_graph_drug` 판별 및 재사용 더미 그래프를 적용하여 RDKit 2D 그래프 파싱 오버헤드 100% 제거 (추론 Latency 대폭 단축).
- [x] **Task E-2: FastAPI 서버 다중 모델 동시 서빙 구조 완성**
  - `ADMET_MODEL_DIR` 및 `DTI_MODEL_DIR` 개별 환경변수 및 자동 탐색(Autodiscovery) 지원으로 ADMET(`/predict`, `/predict/vdss`)과 DTI(`/predict/dti`) 동시 무결 서빙.
- [x] **Task E-3: Cross-Attention XAI Attention Map REST API 노출**
  - `DTIInferenceRequest`에 `return_attention` 옵션 추가 및 `DTIInferenceResponse`에 `attention_weights` 필드 연동.
- [x] **Task E-4: 전체 회귀 테스트 스위트 검증**
  - 91개 단위/통합 테스트 100% Pass 무결성 검증 완료 (`pytest tests/`).

---

## 🚀 3. 추후 개선해야 할 사항 (Medium & Long-Term Roadmap)

| 분류 | 작업 ID | 핵심 작업 내용 | 목표 지표 및 기대 효과 |
| :--- | :--- | :--- | :--- |
| **DTA 고도화** | **Task F-1** | **Cross-Attention Full Token-Level Contact Map 구현**<br/>• ChemBERTa 및 ESM-2 인코더에 `return_sequence=True` 지원<br/>• 풀링 전 토큰 시퀀스(`[B, L_drug, D]` × `[B, L_target, D]`) 간 어텐션 모델링 | • 약물 원자(Atom) × 단백질 잔기(Residue) 간의 2D Contact Map 시각화<br/>• 바인딩 포켓 핵심 잔기 해석력(XAI) 극대화 |
| **DTA 고도화** | **Task F-2** | **Phase C [옵션 B] 정규화 강화 Staged Fine-Tuning**<br/>• ChemBERTa 상위 2개 RoBERTa 레이어 Staged Unfreezing<br/>• Discriminative LR ($2.0 \times 10^{-6}$) 및 Warmup 6에포크 적용 | • Unseen Scaffold 과적합 방지<br/>• BindingDB_Kd Cold-Drug **CI $\ge 0.77 \sim 0.78$** 달성 |
| **DTA 고도화** | **Task F-3** | **Phase C [옵션 C] Multi-Affinity Multi-Task 확장**<br/>• BindingDB $K_d, K_i, \text{IC}_{50}$ 3대 지표 통합 데이터로더 구축<br/>• 결측 레이블 지원 `MaskedMSELoss` 및 Multi-Head DTA 모델링 | • 학습 데이터 볼륨 3배 확장<br/>• 일반화 오차 극소화 및 범용 DTA 모델 구축 |
| **ADMET 실전** | **Task F-4** | **Cluster 4 (Clearance & Half-life) 생체 연계 학습**<br/>• `half_life_obach`, `clearance_hepatocyte_az`, `clearance_microsome_az` 학습<br/>• Cluster 2의 $f_u$ (PPBR) 및 $V_{\text{dss}}$ 예측값을 생리학적 입력($CL = \frac{V_{\text{dss}} \cdot \ln 2}{t_{1/2}}$)으로 연계 | • 간 클리어런스 및 생체 반감기 정밀 예측<br/>• 생리학 기반 약동학(PBPK) 파이프라인 완성 |
| **ADMET 실전** | **Task F-5** | **Cluster 5 (hERG Central 306k & DILI) 안전성 방어벽 학습**<br/>• 306k hERG Central 3-Head Multi-Task 학습<br/>• DILI (약물유도간손상) 이진분류 모델 결합 | • hERG ROC-AUC $\ge 0.88$, DILI ROC-AUC $\ge 0.82$<br/>• 초기 독성 스크리닝 필터 확립 |
| **엔지니어링** | **Task F-6** | **Docker 프로덕션 컨테이너화 및 W&B Model Registry 자동 동기화**<br/>• FastAPI 기반 경량 배포 Dockerfile 작성<br/>• SOTA 모델 아티팩트 자동 버전 태깅 및 CI/CD 롤백 체계 구축 | • 배포 환경 일관성 및 재현성 100% 보장 |

---

## 🛠️ 주요 설정 파일 및 문서 빠른 링크

| 대상 | 설정 파일 / 문서 경로 |
| :--- | :--- |
| **PPBR / 분포 SOTA 상세 리포트** | [`docs/benchmarks/ppbr_distribution_sota_progress_report.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/ppbr_distribution_sota_progress_report.md) |
| **Caco-2 SOTA 리포트** | [`docs/benchmarks/caco2_sota_progress_report.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/caco2_sota_progress_report.md) |
| **DTI Phase B 실행 계획서** | [`docs/dti/dti_phase_b_plan.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/dti/dti_phase_b_plan.md) |
| **DTI Phase C 백로그 상세** | [`docs/dti/dti_phase_c_todolist.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/dti/dti_phase_c_todolist.md) |
| **전체 목표 지표 정의** | [`docs/benchmarks/admetlab3_tdc_target_performance.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/admetlab3_tdc_target_performance.md) |
| **클러스터 아키텍처 가이드** | [`docs/guides/admet_cluster_architectures.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/guides/admet_cluster_architectures.md) |
| **CYP450 Stage 2 설정** | [`configs/config_cyp450_stage2_substrates.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_cyp450_stage2_substrates.yaml) |
| **DTI Phase C 설정** | [`configs/config_dti_phase_c.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_dti_phase_c.yaml) |
| **ChEMBL HSA 큐레이션 데이터** | [`data/external/chembl_hsa_processed.csv`](file:///C:/Users/xps/orca/workspaces/tdc-studio/data/external/chembl_hsa_processed.csv) |

