# ADMET Multi-Task Training Pipeline Execution Checklist & TODOLIST

> **작성 일시:** 2026-09-22 23:28 (KST)  
> **기준 Git Commit:** `b7c346a` (hERG Standalone 및 독성 MTL 분리 완료)  
> **W&B 추적 프로젝트:** `tdc-studio/tdc-learning`  
> **핵심 원칙:** *Caco-2 SOTA 달성 기법(물리화학 앵커 MTL + Kendall-Gal 불확실성 가중치 + 도메인 불변 앙상블)을 전 클러스터로 확장*

---

## 📌 1. 내일 작업 시작 시 즉시 확인할 체크포인트 (Pre-flight Checklist)

- [ ] **1-1. 최신 커밋 상태 확인 및 브랜치 점검**
  ```powershell
  git status
  git log -n 3 --oneline
  ```
- [ ] **1-2. TDC 의존성 및 Python 환경 확인**
  - PyTDC 패키지 및 RDKit, PyTorch Geometric 로드 상태 확인:
  ```powershell
  uv run python -c "import torch, torch_geometric, rdkit; print('Environment OK')"
  ```
  *(필요 시 `uv add PyTDC` 또는 `uv pip install PyTDC` 확인)*
- [ ] **1-3. W&B 로깅 자격증명 및 엔터티 점검**
  - `tdc-studio` 엔터티의 `tdc-learning` 프로젝트 연결 상태 확인

---

## 🚀 2. 단계별 학습 파이프라인 테스트 TODOLIST

### [Phase 1] 1-Epoch 초고속 Smoke Test (데이터 로딩 및 입출력 Shape 무결성 검증)
*실제 GPU 장시간 학습 전, 1 에폭 및 미니 배치를 통해 파이프라인 충돌 유무를 검증합니다.*

- [x] **Task 1-A: Cluster 1 (Lipophilicity MTL: Lipo + Sol + FreeSolv + Caco2) Smoke Test**
  - 설정 파일: `configs/config_lipophilicity_mtl.yaml`
  - 검증 완료: 4개 타깃 동시 역전파 및 Z-Score 정규화 (W&B Run: [`0802sz7x`](https://wandb.ai/tdc-studio/tdc-learning/runs/0802sz7x))
- [x] **Task 1-B: Cluster 2 (Distribution MTL: PPBR + BBB + VDss) Smoke Test**
  - 설정 파일: `configs/config_distribution_mtl.yaml`
  - 검증 완료: 혼합형(회귀: PPBR, VDss / 분류: BBB) 및 Kendall-Gal $\sigma$ 가중치 학습 (W&B Run: [`1ps5hiat`](https://wandb.ai/tdc-studio/tdc-learning/runs/1ps5hiat))
- [x] **Task 1-C: Cluster 3 (CYP450 Matrix: 5-Inhib + 3-Substrate) Smoke Test**
  - 설정 파일: `configs/config_cyp450_mtl.yaml`
  - 검증 완료: 8-Head 다중 분류 출력 및 Masked Loss (데이터셋별 샘플 누락 처리) (W&B Run: [`pcyuwgfh`](https://wandb.ai/tdc-studio/tdc-learning/runs/pcyuwgfh))
- [x] **Task 1-D: Cluster 4 (Clearance & Half-Life) Smoke Test**
  - 설정 파일: `configs/config_clearance_mtl.yaml`
  - 검증 완료: 간세포(Hepatocyte) & 마이크로솜(Microsome) $CL_{\text{int}}$ Spearman $\rho$ 연산 (W&B Run: [`7icc0eye`](https://wandb.ai/tdc-studio/tdc-learning/runs/7icc0eye))
- [x] **Task 1-E: Cluster 5 (Toxicity Balanced MTL) Smoke Test**
  - 설정 파일: `configs/config_toxicity_mtl.yaml`
  - 검증 완료: hERG(648), LD50(7.4k), DILI(475), AMES(7.2k) 균형 학습 및 Focal Loss 수렴 (W&B Run: [`kqhdqf3o`](https://wandb.ai/tdc-studio/tdc-learning/runs/kqhdqf3o))
- [x] **Task 1-F: Standalone Cardiotoxicity (hERG Central 306k) Batch Shape Test**
  - 설정 파일: `configs/config_herg_standalone.yaml`
  - 검증 완료: `hERG_at_1uM`, `hERG_at_10uM`, `hERG_inhib` 3-Head 출력 및 대용량 배치 메모리 할당 (W&B Run: [`j108jp0n`](https://wandb.ai/tdc-studio/tdc-learning/runs/j108jp0n))

---

### [Phase 2] 본격 모델 수렴 및 메트릭 벤치마킹 (Full Training & HPO)
*Smoke test가 통과되면 클러스터별 본격 학습을 실행하고 SOTA 지표 도달 여부를 기록합니다.*

- [ ] **Task 2-A: Cluster 1 (Lipophilicity 앵커) 정밀 학습**
  - 목표: Lipophilicity $R^2 \ge 0.74$, AqSolDB MAE $\le 0.70$
- [ ] **Task 2-B: Cluster 2 (PPBR / BBB / VDss) 정밀 학습**
  - 목표: PPBR MAE $\le 9.2\%$, BBB ROC-AUC $\ge 0.912$, VDss $R^2 \ge 0.58$
- [ ] **Task 2-C: Cluster 3 (CYP450 8-Head Matrix) 정밀 학습**
  - 목표: 저해 5종 ROC-AUC $\ge 0.90 \sim 0.94$, 기질 3종 ROC-AUC $\ge 0.78 \sim 0.84$
- [ ] **Task 2-D: Cluster 4 (Clearance & Half-Life) 정밀 학습**
  - 목표: Microsome Spearman $\rho \ge 0.575$, Hepatocyte Spearman $\rho \ge 0.403$, $t_{1/2}$ $R^2 \ge 0.65$
- [ ] **Task 2-E: Cluster 5 (균형 독성 프로파일링) 정밀 학습**
  - 목표: hERG ROC-AUC $\ge 0.887$, LD50 MAE $\le 0.584$, DILI ROC-AUC $\ge 0.860$, AMES $\ge 0.882$
- [ ] **Task 2-F: hERG Standalone 2단계 학습 (Pretrain 306k $\rightarrow$ Fine-tune Karim/Wang)**
  - 목표: `hERG_Karim` ROC-AUC $\ge 0.940$, `hERG` (Wang) ROC-AUC $\ge 0.895$

---

### [Phase 3] 벤치마크 리더보드 및 W&B 대시보드 리포팅
- [ ] **Task 3-A: W&B Run 결과 요약 및 클러스터별 Best Run 태깅**
- [ ] **Task 3-B: `docs/benchmarks/admetlab3_tdc_target_performance.md`에 실측 성능 업데이트**
- [ ] **Task 3-C: SOTA 체크포인트 저장 (`checkpoints/{cluster}_best.pt`) 및 배포 가이드 동기화**

---

## 🛠️ 주요 설정 파일 및 문서 빠른 링크

| 대상 | 설정 파일 / 문서 경로 |
| :--- | :--- |
| **Caco-2 SOTA 리포트** | [`docs/benchmarks/caco2_sota_progress_report.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/caco2_sota_progress_report.md) |
| **전체 목표 지표 정의** | [`docs/benchmarks/admetlab3_tdc_target_performance.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/benchmarks/admetlab3_tdc_target_performance.md) |
| **클러스터 아키텍처 가이드** | [`docs/guides/admet_cluster_architectures.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/guides/admet_cluster_architectures.md) |
| **SOTA 엔지니어링 레시피** | [`docs/guides/admet_sota_recipe.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/docs/guides/admet_sota_recipe.md) |
| **Caco-2 앵커 설정** | [`configs/config_caco2_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_caco2_mtl.yaml) |
| **Lipophilicity 앵커 설정** | [`configs/config_lipophilicity_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_lipophilicity_mtl.yaml) |
| **체내 분포 설정** | [`configs/config_distribution_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_distribution_mtl.yaml) |
| **CYP450 매트릭스 설정** | [`configs/config_cyp450_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_cyp450_mtl.yaml) |
| **클리어런스/반감기 설정** | [`configs/config_clearance_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_clearance_mtl.yaml) |
| **일반 독성 설정** | [`configs/config_toxicity_mtl.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_toxicity_mtl.yaml) |
| **hERG 306k 단독 모델 설정** | [`configs/config_herg_standalone.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_herg_standalone.yaml) |
