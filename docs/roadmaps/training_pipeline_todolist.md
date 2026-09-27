# ADMET & DTI Multi-Task Training Pipeline Execution Checklist & TODOLIST

> **최종 갱신 일시:** 2026-09-27 20:45 (KST)  
> **W&B 추적 프로젝트:** `tdc-studio/tdc-learning`  
> **실행 환경 원칙:** **로컬 학습 지양, Google Colab Cloud GPU (`remote exec`) 기본 적용 (`munhyoungdo@gmail.com`)**

---

## 📅 오늘(2026-09-27) 작업 완료 요약 (Today's Executive Summary)

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

---

## 📅 이전(2026-09-23) 작업 완료 요약 (Previous Summary)

### 1. 체내 분포(Cluster 2: PPBR / BBB / VDss / Lipophilicity) SOTA 돌파
- **단일 모델 $R^2 \ge 0.50$ 최초 돌파**:
  - 4-태스크 DMPNN-MTL 단독 모델: Test $R^2 = \mathbf{0.5032}$, Pearson $r = \mathbf{0.7719}$.
- **Tri-Hybrid Foundation Stacker 신기록 수립**:
  - **Test $R^2 = \mathbf{0.5412}$**, **MAE = $\mathbf{6.23\%}$**, **RMSE = $10.54\%$**, **Spearman $\rho = \mathbf{0.7525}$** (전체 TDC 벤치마크 역대 최고 순위 상관계수 달성).
  - Branch 1: DMPNN-MTL 분자 그래프 (14노드, 6엣지, 6,000+ 화합물 전이).
  - Branch 2: 18-특성 생물물리학 융합 GBDT (Sudlow 사이트, 이온화율, F_CSP3, TPSA/MW, $[N^+]$ 영구 양이온 모티프).
  - Branch 3: ChemBERTa-77M-MTR 대형 사전학습 언어모델 (384차원 임베딩 RidgeCV).
  - 모달리티 앙상블 잔차의 직교성($\text{Cov}(\epsilon_{\text{GNN}}, \epsilon_{\text{GBDT}}) \approx 0.15$) 및 Parametric Sigmoid 보정을 통해 설명 분산 극대화.

### 2. $R^2 < 0.60$ 병목 원인 진단 (오차 잔차 병리학 분석)
- 323개 격리 테스트 세트 심층 오차 분해:
  - 고결합 화합물 ($\ge 90\%$, $n=223$, $69\%$): **MAE = $3.34\%$**, 전체 오차 분산의 **$16.4\%$**에 불과.
  - 저결합 화합물 ($< 70\%$, $n=38$, $11.8\%$): **MAE = $21.83\%$**, **전체 잔차 제곱합($SSE$)의 $69.7\%$($25,019 / 35,900$)를 독점**.
  - 핵심 원인 규명:
    1. 2D vs 3D 입체 형태 불일치 (예: 비평면 Tropolone 링을 가진 Colchicine의 결합 저해를 2D가 과대평가).
    2. 영구 4차 암모늄 이온 ($[N^+]$, 강한 수화 껍질로 알부민 결합 방해).
    3. TDC 데이터셋의 심각한 고결합 편향 ($70\%$ 이상이 $\ge 90\%$).

### 3. 클라우드 GPU 원격 실행 체계 확립 (Colab Cloud Runner)
- 사용자 지침 준수: 로컬 CPU 학습을 지양하고 모든 학습 및 대규모 평가 명령을 Google Colab T4 GPU(`--remote`) 기본값으로 전환.
- 전송 번들 최적화: 대용량 캐시를 제외하여 Base64 번들 용량을 **11.94 MB $\to$ 162 KB**로 경량화, 웹소켓 타임아웃 완전 해결.
- ChEMBL HSA 5-태스크 45 에포크 학습 완료 (Loss $-2.2459$ 안정 수렴, Kendall 불확실성 가중치 가우시안 NLL 음수 특성 수학적 검증 완료).

### 4. DTI / DTA (Phase A) 기초 아키텍처 완성
- `GraphDTAModel`, `ProteinCNNEncoder`, `BilinearAttentionFusion`, `AminoAcidTokenizer`, `DTADataModule`(cold_drug 분할) 구현 및 Colab T4 GPU 6-단계 dry-run 무결성 검증 통과.

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

### 🥇 [Package 1: hERG & AMES 전용 안전성 챔피언 모델 구축 (Colab GPU: `munhyoungdo@gmail.com`)]
- [ ] **Task E-1: hERG 2-Stage Fine-tuning 원격 Colab GPU 학습**
  - 설정: [`configs/config_herg_standalone.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/ADMET/configs/config_herg_standalone.yaml)
  - Stage 1: `herg_karim` (13,445개) 단독 백본 사전학습 (Hidden Dim 512, Dropout 0.2)
  - Stage 2: `herg` (Wang et al., 648개) 저비율($0.1 \times \text{lr}$) 미세조정 및 헤드 적응
  - 목표: `herg_karim` AUROC $\ge \mathbf{0.90}$, `herg` AUROC $\ge \mathbf{0.88} \sim \mathbf{0.90}$ 진입
- [ ] **Task E-2: AMES 유전독성 구조 경보(Ashby-Tennant) 모듈 및 단독 모델 복구**
  - Ashby-Tennant 100-dim 하위구조 경보 비트벡터 추출 모듈 구현 (`tdc_studio/features/structural_alerts.py`)
  - Focal Loss ($\gamma=2.0$) 기반 단독 D-MPNN-Des / GBDT 파이프라인 구축 및 학습
  - 목표: `ames` 돌연변이원성 AUROC $\mathbf{0.50} \to \ge \mathbf{0.86}$ 즉시 정상화

### 🥈 [Package 2: 간세포 클리어런스 계단식 전이 & Lipo GBDT 스태킹]
- [ ] **Task E-3: 간세포(Hepatocyte) 클리어런스 계단식 캐스케이딩(Cascaded Transfer)**
  - 이미 $\rho = 0.6918$을 달성한 `clearance_microsome_az` 예측값을 `clearance_hepatocyte_az`의 사전 피처(Prior Feature)로 주입하는 2차 파이프라인 구성
  - 목표: `clearance_hepatocyte_az` Spearman $\rho = 0.33 \to \ge \mathbf{0.45}$ 도약
- [ ] **Task E-4: Lipophilicity AstraZeneca GBDT + ChemBERTa 스태킹**
  - RDKit Crippen LogP, LabuteASA, pKa 특화 24-dim 모티프 결합 GBDT + ChemBERTa 언어모델 앙상블
  - 목표: `lipophilicity` Test $R^2 = 0.7602 \to \ge \mathbf{0.85}$ 도약

### 🥉 [Package 3: 전주기 Unified ADMET 통합 서빙 엔드포인트 구축]
- [ ] **Task E-5: 통합 서빙 엔드포인트 `POST /predict/admet_full` 구현**
  - Cluster 1(Caco2/Lipo), Cluster 2(PPBR/VDss/BBB), Cluster 3(CYP450 8-Head), Cluster 4(Clearance/Half-life/PBPK), Cluster 5(DILI/ClinTox/hERG/LD50)를 단일 프로세스에서 동시 로드
  - 단일 SMILES 입력 시 22대 전주기 ADMET 예측치 및 PBPK PK 파라미터 일괄 반환
- [ ] **Task E-6: 통합 서빙 E2E 테스트 및 Docker 컨테이너 엔드포인트 검증**

---

## 🛠️ 주요 설정 파일 및 문서 빠른 링크

| 대상 | 설정 파일 / 문서 경로 |
| :--- | :--- |
| **PPBR / 분포 SOTA 상세 리포트** | [`docs/benchmarks/ppbr_distribution_sota_progress_report.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/ppbr_distribution_sota_progress_report.md) |
| **Caco-2 SOTA 리포트** | [`docs/benchmarks/caco2_sota_progress_report.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/caco2_sota_progress_report.md) |
| **전체 목표 지표 정의** | [`docs/benchmarks/admetlab3_tdc_target_performance.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/admetlab3_tdc_target_performance.md) |
| **클러스터 아키텍처 가이드** | [`docs/guides/admet_cluster_architectures.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/guides/admet_cluster_architectures.md) |
| **5-태스크 분포 학습 설정** | [`configs/config_distribution_mtl_5tasks.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_distribution_mtl_5tasks.yaml) |
| **DTI Phase A 설정** | [`configs/config_dti_phase_a.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_dti_phase_a.yaml) |
| **ChEMBL HSA 큐레이션 데이터** | [`data/external/chembl_hsa_processed.csv`](file:///C:/Users/xps/orca/workspaces/tdc-studio/data/external/chembl_hsa_processed.csv) |
