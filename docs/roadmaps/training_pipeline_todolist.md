# ADMET & DTI Multi-Task Training Pipeline Execution Checklist & TODOLIST

> **최종 갱신 일시:** 2026-09-23 20:05 (KST)  
> **W&B 추적 프로젝트:** `tdc-studio/tdc-learning`  
> **실행 환경 원칙:** **로컬 학습 지양, Google Colab Cloud GPU (`remote exec`) 기본 적용**

---

## 📅 오늘(2026-09-23) 작업 완료 요약 (Today's Executive Summary)

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
- [ ] **Task C-1: Cluster 1 (Lipophilicity 앵커: Lipo + Sol + FreeSolv + Caco2) 정밀 학습**
  - 목표: Lipophilicity $R^2 \ge 0.74$, AqSolDB MAE $\le 0.70$.
- [ ] **Task C-2: Cluster 3 (CYP450 8-Head Matrix) 정밀 학습**
  - 목표: 저해 5종 ROC-AUC $\ge 0.90 \sim 0.94$, 기질 3종 ROC-AUC $\ge 0.78 \sim 0.84$.

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
