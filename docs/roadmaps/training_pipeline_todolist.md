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

## 📌 1. 내일(2026-09-24) 작업 시작 시 즉시 확인할 체크포인트 (Pre-flight Checklist)

- [ ] **1-1. 최신 커밋 상태 확인 및 브랜치 점검**
  ```powershell
  git status
  git log -n 3 --oneline
  ```
- [ ] **1-2. Google Colab GPU 세션 연결 확인**
  ```powershell
  uv run tdc-studio remote status
  ```
- [ ] **1-3. W&B 프로젝트 연결 확인**
  - `https://wandb.ai/tdc-studio/tdc-learning` 대시보드 상태 점검

---

## 🚀 2. 내일 이어서 진행할 작업 TODOLIST (Tomorrow's Action Plan)

### [Track A] PPBR / Distribution 클러스터 $R^2 \ge 0.60$ 돌파 및 최종 패키징
- [ ] **Task A-1: 5-태스크 체크포인트 + Tri-Hybrid 통합 재평가 (Colab GPU)**
  - 신규 학습된 5-태스크 DMPNN 체크포인트와 기존 4-태스크 모델의 표현형 앙상블 블렌딩 검증.
- [ ] **Task A-2: 저결합($< 70\%$) 집중 가중치(Reweighted / Focal / Sample-Weighted) 학습**
  - $69.7\%$의 오차를 유발하는 $11.8\%$의 저결합 화합물에 대해 역빈도 가중치($w_i \propto \frac{1}{\text{density}(y_i)}$) 또는 3D 입체 파라미터(Conformer PBF, Spherocity Index)를 추가하여 저결합 예측 잔차 집중 공략.
- [ ] **Task A-3: Distribution 클러스터 SOTA 체크포인트 내보내기 (`models/export/ppbr_tri_hybrid_sota.pt`)**
  - FastAPI 서빙 엔드포인트 연동 및 추론 파이프라인 검증.

### [Track B] DTI / DTA Phase B (사전학습 파운데이션 모델 결합)
- [ ] **Task B-1: Pretrained Protein & Compound Encoders 구현 (`tdc_studio/models/dti/pretrained_encoders.py`)**
  - Drug: ChemBERTa-77M-MTR (384차원)
  - Target: ESM-2 (`facebook/esm2_t6_8M_UR50D` 또는 `esm2_t12_35M_UR50D`)
- [ ] **Task B-2: BindingDB_Kd Cold-Drug 벤치마크 학습 실행 (Colab GPU)**
  - 설정: `configs/config_dti_phase_a.yaml` 기반 Colab GPU 원격 학습.
  - 목표 지표: Cold Drug Concordance Index (CI) $\ge 0.70$, MSE $\le 0.75$.

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
