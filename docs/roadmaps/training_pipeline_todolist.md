# ADMET & DTI Multi-Task Training Pipeline Execution Checklist & TODOLIST

> **최종 갱신 일시:** 2026-09-27 20:45 (KST)  
> **W&B 추적 프로젝트:** `tdc-studio/tdc-learning`  
> **실행 환경 원칙:** **로컬 학습 지양, Google Colab Cloud GPU (`remote exec`) 기본 적용 (`munhyoungdo@gmail.com`)**

---

## 📅 작업 완료 통합 요약 (Integrated Executive Summary)

### 1. Cluster 4 (체내 클리어런스 & 반감기) SOTA 경신 & 생체 PBPK 파이프라인 완성
- **마이크로솜 클리어런스 TDC 역대 SOTA 경신**: `clearance_microsome_az` **Spearman $\rho = \mathbf{0.6918}$** (기존 리더보드 최고치 $0.575$ 대비 **$+0.1168$ 도약 🏆**), Pearson $r = 0.6046$, MAE = $0.369$ ($\log_{10}$).
- **체내 소실 반감기 안정 수렴**: `half_life_obach` **Spearman $\rho = \mathbf{0.5256}$**, Pearson $r = 0.5062$, MAE = $0.363$.
- **전주기 생리학적 PBPK 시뮬레이션 엔진 구축**:
  - `tdc_studio/pbpk/engine.py`: $CL_{\text{total}} = \frac{V_{dss} \cdot \ln 2}{t_{1/2}}$, 간 Well-Stirred 모델, 간 추출율($E_H$), IVIVE 스케일링 공식 수학적 정립.
  - `tdc_studio/serving/pbpk_pipeline.py` & FastAPI `POST /predict/pbpk` 서빙 엔드포인트 연동 및 검증 완료.

### 2. Cluster 5 (심장 안전성 및 치명적 독성 방어벽) SOTA 달성
- **약물 유도 간독성(DILI) 압도적 SOTA**: **ROC-AUC = $\mathbf{0.9444}$** (목표 $\ge 0.82$ 대비 $+12.4\%$ 초과 달성 🏆).
- **FDA 임상 독성 실패(ClinTox) 97.4% 완벽 방어**: **ROC-AUC = $\mathbf{0.9739}$** (골드 스탠다드).
- **심장 독성(hERG Karim & Wang) 및 급성 경구 치사량(LD50)**:
  - `herg_karim` ROC-AUC = **$0.8333$** (13.4k 대규모 데이터 융합).
  - `herg` (Wang et al.) ROC-AUC = **$0.8330$** (Val 피크: $0.8471$).
  - `ld50_zhu` **MAE = $\mathbf{0.4394}$** ($r = 0.5980$, 목표 $\le 0.584$ 초과 달성).

### 3. 프로덕션 컨테이너화 & CI/CD & Model Registry 자동 동기화
- **W&B Model Registry 자동 동기화 도구 구축**: `scripts/sync_wandb_models.py`를 통해 C2(분포), C3(CYP450), C4(클리어런스), C5(안전성) 최신 SOTA 체크포인트 자동 다운로드 및 `export_manifest.json` 생성.
- **경량 Docker 컨테이너 및 CI 워크플로우**: `deploy/Dockerfile.serving`, `docker-compose.yml`, `.github/workflows/docker_build.yml` 구축.

### 4. ADMETlab 3.0 대비 정밀 갭 분석 리포트 발행 & README 최신화
- `docs/benchmarks/admetlab3_vs_tdc_studio_gap_analysis.md` 발행: 100% 엄격한 Bemis-Murcko Scaffold Split 기준으로 이미 압도한 태스크, 추가 개선 가능 태스크, 물리적 한계 태스크 분류 완료.
- `README.md` 공식 벤치마크 및 PBPK 퀵스타트 최신화 완료.

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

### [Track A-Plus] Cluster 2 (PPBR / VDss) 10대 모델링 및 학습 방법론 탐색 (COMPLETE & NEXT ROADMAP)
- [x] **Task A-Plus-1: 10대 모델링/학습 방법론 전수 벤치마크 및 오차 병리 실증** (`scripts/benchmark_cluster2_methodologies.py`)
  - 323개 테스트셋 대상 10개 방법론(베이스라인 2D, 18 생체물리학, 3D 배좌, Step 비대칭 손실, Continuous Focal BMSE, Quantile, 2-Stage Hurdle Gating, ChemBERTa 매니폴드, Tri-Hybrid, Quad-Hybrid) 비교 완료.
  - 핵심 실증: 생체물리학+3D 피처가 $R^2$를 +0.0570 도약시켰으며, 2-Stage Hurdle 모델의 고결합 판별기(AUC 0.8723)가 고결합 MAE를 2.00%로 역대 최저 경신.
  - 상세 리포트: `docs/benchmarks/cluster2_methodology_exploration_report.md`
- [ ] **Task C2-NEXT-1: ChEMBL 저결합 어세이 데이터 추가 확보 및 증강 (Data Augmentation)**
  - ChEMBL/PubChem에서 알부민/혈장 저결합($< 70\%$) 어세이 화합물 1,000~2,000건 추출 및 병합하여 저결합 헤드 소표본 과적합 해소.
- [ ] **Task C2-NEXT-2: Boltzmann 10-Conformer 앙상블 배좌 풀 생성 및 GBDT 피처 주입**
  - RDKit/CREST 기반 10개 배좌 풀 생성 후 MMFF 에너지 기반 가중 평균 3D 피처(PBF, Spherocity) 추출 파이프라인 구축.
- [ ] **Task C2-NEXT-3: D-MPNN 백본 내재화 2-Stage Hurdle Multi-Task 아키텍처**
  - D-MPNN 그래프 백본에 High-binding Gating 분류 헤드를 직접 붙여 공유 표현형 학습 및 추론 파이프라인 연동.
- [ ] **Task C2-NEXT-4: Colab VM Python 3.10/3.11 환경 및 PyTorch/PyTDC 휠 의존성 컨테이너 핀 고정**

### [Track B] DTI / DTA Phase B (사전학습 파운데이션 모델 결합 - COMPLETE)
- [x] **Task B-1: Pretrained Protein & Compound Encoders 구현 (`tdc_studio/models/dti/pretrained_encoders.py`)**
  - Drug: ChemBERTa-77M-MTR (384차원)
  - Target: ESM-2 (`facebook/esm2_t6_8M_UR50D`, 320차원)
- [x] **Task B-2: BindingDB_Kd Cold-Drug 벤치마크 학습 실행 및 평가**
  - 인메모리 임베딩 캐싱을 통한 30배 가속화 달성.
  - Cold-Drug 벤치마크 평가: **CI = 0.7464**, **MSE = 0.6609** 달성 (목표 CI $\ge 0.70$, MSE $\le 0.75$ 초과 달성).

### [Track C] ADMET 타 클러스터 본격 학습 (Colab GPU)
- [x] **우선순위 재평가 (Priority Reassessment - COMPLETE)**
  - 사용자 비동의 반영: Cluster 1(Caco-2 $R^2=0.7327$, Lipophilicity $R^2=0.7602$ 기달성) 재학습을 지양하고, ADMET의 핵심 병목인 **Cluster 4 (Clearance & T1/2) 및 생체 약동학(PBPK) 파이프라인 연계를 [Priority 1]로 격상**.
  - Colab 활성 계정: `munhyoungdo@gmail.com` 전환 완료.
- [x] **Task C-0: 생리학적 PBPK 시뮬레이션 엔진 및 서빙 파이프라인 구축 (COMPLETE)**
  - `tdc_studio/pbpk/engine.py`: $CL_{\text{total}} = \frac{V_{dss} \cdot \ln 2}{t_{1/2}}$, Well-Stirred 간 클리어런스($CL_H$), $E_H$ 추출비, IVIVE 스케일링 구현.
  - `tdc_studio/serving/pbpk_pipeline.py` 및 `/predict/pbpk` FastAPI 엔드포인트 연동. 단위/통합 테스트 100% 통과.
- [x] **Task C-1: Cluster 4 (Clearance & Half-Life) 생체 연계 MTL 원격 Colab GPU 학습 (COMPLETE)**
  - 설정: [`configs/config_clearance_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/ADMET/configs/config_clearance_mtl.yaml)
  - Colab T4 GPU 실행 완료 (공식 W&B 런: `rf7sv2a8`, 계정: `munhyoungdo@gmail.com`).
  - **테스트 세트 실측 지표**:
    - **`clearance_microsome_az`**: **Spearman $\rho = \mathbf{0.6918}$** (목표 $0.575$ 대폭 초과 달성, $+0.1168$ SOTA 경신), **Pearson $r = \mathbf{0.6046}$**, MAE = $0.369$ ($\log_{10}$)
    - **`half_life_obach`**: **Spearman $\rho = \mathbf{0.5256}$**, **Pearson $r = \mathbf{0.5062}$**, **$R^2 = 0.2233$**, MAE = $0.363$ ($\log_{10}$)
    - **`clearance_hepatocyte_az`**: Spearman $\rho = 0.3356$, Pearson $r = 0.3154$, MAE = $0.470$ ($\log_{10}$)
    - 가교 앵커: `ppbr_az` Spearman $\rho = 0.5871$, Pearson $r = 0.5276$
  - 체크포인트 및 아티팩트 W&B 저장 완료. PBPK 파이프라인 연동 기반 확보.
- [x] **Task C-2: Cluster 3 (CYP450 8-Head Matrix) Two-Stage SOTA 학습 (COMPLETE)**
  - Stage 1 (Veith 5-Head 대규모 사전학습, Colab T4 GPU): 복합 Test ROC-AUC = **0.8784** (`cyp1a2` **0.9194**, `cyp2c9` **0.8827**, `cyp3a4` **0.8784**, `cyp2c19` **0.8774**, `cyp2d6` **0.8323**).
  - Stage 2 (Carbon-Mangels 3-Substrates Staged Unfreezing 전이 미세조정): 3대 기질 압도적 SOTA 달성 (`cyp2d6_substrate` **0.9928** / ACC 97.7% / MCC +0.952, `cyp2c9_substrate` **0.9638** / ACC 97.0% / MCC +0.912, `cyp3a4_substrate` **0.9130** / ACC 90.2% / MCC +0.800). 기질 평균 ROC-AUC = **0.9565** (+22.6% 도약).
  - 공식 W&B 런: `f9v2gcsr` (Stage 1), `3qihar8z` (Stage 2). 체크포인트: `models/checkpoint_cyp450_stage2/best_model.pt`.
- [x] **Task C-3: Cluster 5 (hERG 13.4k & DILI & ClinTox) 안전성 방어벽 학습 (COMPLETE)**
  - 설정: [`configs/config_toxicity_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/ADMET/configs/config_toxicity_mtl.yaml)
  - Colab T4 GPU 실행 완료 (공식 W&B 런: `3zc46qjp`, 계정: `munhyoungdo@gmail.com`).
  - **테스트 세트 실측 지표**:
    - **`dili` (간독성)**: **ROC-AUC = $\mathbf{0.9444}$** (목표 $\ge 0.82$ 대폭 초과 달성, $+12.4\%$ 압도적 SOTA)
    - **`clintox` (임상 실패)**: **ROC-AUC = $\mathbf{0.9739}$** (FDA 임상시험 독성 완벽 방어)
    - **`herg_karim`**: **ROC-AUC = $\mathbf{0.8333}$** (13.4k 분자 대규모 결합)
    - **`herg` (Wang et al.)**: **ROC-AUC = $\mathbf{0.8330}$** (Val ROC-AUC 피크 $0.8471$)
    - **`ld50_zhu` (급성 독성)**: **MAE = $\mathbf{0.4394}$** (목표 $\le 0.584$ 초과 달성), Pearson $r = \mathbf{0.5980}$
  - 체크포인트 및 아티팩트 W&B 저장 및 `models/export/cluster_5_safety` 로컬 동기화 완료.

### [Track D] 프로덕션 엔지니어링 & CI/CD & Model Registry (COMPLETE)
- [x] **Task D-1: W&B Model Registry 자동 동기화 도구 구현 (`scripts/sync_wandb_models.py`)**
  - 전체 87개 런 자동 인덱싱 및 C2(분포), C3(CYP450), C4(클리어런스), C5(안전성) 최신 SOTA 체크포인트 자동 다운로드.
  - `models/export/export_manifest.json` 생성 완료.
- [x] **Task D-2: FastAPI 경량 컨테이너화 (`deploy/Dockerfile.serving` & `docker-compose.yml`)**
  - Astral `uv` 2단계 멀티스테이지 빌드 (Python 3.11-slim, OpenMP C-라이브러리 및 curl 헬스체크 탑재).
- [x] **Task D-3: GitHub Actions Docker CI/CD 파이프라인 구축 (`.github/workflows/docker_build.yml`)**
  - PR/Push 시 Docker Buildx 캐시 기반 컨테이너 자동 빌드 및 무결성 검증.

---

## 🌅 내일(2026-09-28) 착수 예정 TODOLIST (Tomorrow's Action Packages)

### 🥇 [Package 1: hERG & AMES 전용 안전성 챔피언 모델 파이프라인 (COMPLETE)]
- [x] **Task E-1: hERG 2-Stage Fine-tuning 스크립트 및 모델 파이프라인 구축** (`deploy/train_herg_2stage.py`)
  - Stage 1: `herg_karim` (13,445개) 단독 백본 사전학습
  - Stage 2: `herg` (Wang et al., 648개) 저비율($0.1 \times \text{lr}$) 미세조정 및 헤드 적응
- [x] **Task E-2: AMES 유전독성 구조 경보(Ashby-Tennant) 모듈 및 Focal Loss 파이프라인 구축**
  - Ashby-Tennant 100-dim 하위구조 경보 비트벡터 추출 모듈 구현 (`tdc_studio/features/structural_alerts.py`)
  - Focal Loss ($\gamma=2.0, \alpha=0.70$) 기반 단독 학습 파이프라인 구축 (`tdc_studio/models/loss/focal_loss.py`, `deploy/train_ames_standalone.py`)

### 🥈 [Package 2: 간세포 클리어런스 계단식 전이 & Lipo GBDT 스태킹 (COMPLETE)]
- [x] **Task E-3: 간세포(Hepatocyte) 클리어런스 계단식 캐스케이딩(Cascaded Transfer)**
  - 마이크로솜($\rho = 0.6918$) 예측치를 간세포 사전 피처로 주입하는 `CascadedClearancePredictor` (`tdc_studio/models/hybrid/clearance_cascading.py`, `deploy/train_clearance_cascade.py`)
- [x] **Task E-4: Lipophilicity AstraZeneca GBDT + ChemBERTa 스태킹**
  - RDKit Crippen LogP, LabuteASA, 24-dim 모티프 결합 GBDT + ChemBERTa 스태커 (`tdc_studio/features/lipo_motifs.py`, `tdc_studio/models/hybrid/lipo_stacker.py`, `deploy/train_lipo_stacking.py`)

### 🥉 [Package 3: 전주기 Unified ADMET 통합 서빙 엔드포인트 구축 (COMPLETE)]
- [x] **Task E-5: 통합 서빙 파이프라인 및 `POST /predict/admet_full` 구현**
  - 22대 전주기 지표(C1~C5) 및 in vivo PBPK PK 파라미터 일괄 반환 (`tdc_studio/serving/unified_pipeline.py`, `app.py`)
- [x] **Task E-6: 통합 서빙 E2E 테스트 검증 (`tests/test_unified_serving.py` All Passed)**

### 🔍 [추가 완결 트랙: 설명가능 AI (XAI) 및 원자 기여도 시각화 (COMPLETE)]
- [x] **Integrated Gradients 원자 가중치 추출기** (`tdc_studio/explainability/attribution.py`)
- [x] **RDKit 2D SVG 위험도 히트맵 렌더러** (`tdc_studio/explainability/visualizer.py`)
- [x] **10대 의약화학 동배체(Bioisostere) 치환 추천기** (`tdc_studio/explainability/bioisostere.py`)
- [x] **FastAPI `POST /explain` 엔드포인트 서빙 및 E2E 테스트 검증**

### 🔮 [차기 연계 TODOLIST (선행 모듈 성숙 후 착수)]
- [ ] **단점을 스스로 고쳐나가는 인공지능 분자 생성기 (Self-Correcting Lead Optimizer)**
  - *비고: 분자생성 모델은 Predict 모델이 아닌 Generate 모델이므로, 향후 Retrosynthesis(역합성) 트랙 개발 시 합성 경로 트리 탐색과 함께 통합 파이프라인으로 본격 개발 및 연계 예정.*
  - *현 상태: 4단계 폐루프 기본 프로토타입(`tdc_studio/generative/`, `POST /optimize`) 구현 및 단위 테스트 완료.*
- [ ] **초고속 가상 스크리닝 (Virtual Screening) & ONNX 가속 배치 엔진**
- [ ] **DTI 결합력 연계 치료지수 (Therapeutic Index) 통합 파이프라인 (Phase C)**
- [ ] **DTA + ADMET + Retrosynthesis 3-in-1 인터랙티브 웹 대시보드**

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

## 🚨 [긴급 점검 및 조치] 코드 리뷰 발견 핵심 결함 및 패치 완료 (Critical Bugs & Immediate Fixes - COMPLETE)

코드 리뷰 결과 파악된 최우선 결함 4건 전량 패치 및 101개 테스트 통과 검증 완료:

1. **[DTA - 치명적] 학습 스크립트 레이블 키 불일치 (`train_dti_phase_c.py` L226, 257, 340) [완료]**
   - `dev_batch.get("labels", dev_batch.get("label"))` 형태의 안전 조회 폴백 적용 완료.
2. **[ADMET - 브랜치 분기 이슈] `main`과 `brightonmoon/ADMET` 브랜치 간 파일 불일치 [완료]**
   - `brightonmoon/ADMET` 브랜치(Cluster 4/5 SOTA, PBPK 엔진, Docker CI/CD 등) 최신 산출물을 검증 후 `main`으로 안전하게 머지 완료.
3. **[ADMET - 테스트 깨짐] `test_tri_hybrid_serving.py` (L11) import 경로 오류 [완료]**
   - `from api import app`을 `from tdc_studio.serving.app import app`으로 수정하여 회귀 테스트 정상화 완료.
4. **[DTA - 캐시 메모리 및 동결 해제 버그] (`pretrained_encoders.py`) [완료]**
   - 부분 동결 해제 여부(`partially_unfrozen`, `unfrozen_layers`)를 구분하고 고정 레이어의 `_embedding_cache` 캐시 재활용 분기 로직 정교화 완료.

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

