# 🧬 TDC-Studio: 작업 인수인계 및 후속 TODOLIST

## 1. 📋 현재 상태 및 해결된 핵심 성과 요약

| 구분 | 주요 해결 및 개선 내역 | 상태 |
| :--- | :--- | :---: |
| **.ipynb 노트북 정리** | `colab_train_sota.ipynb`, `caco2_t4_runner.ipynb` 완전 삭제 및 Git 정리 | **완료** (`922c9ef`) |
| **대화형 커널 `__file__` 이슈** | `colab_exec.ps1` 헤더 절대경로 주입 및 `deploy/train_clearance_cascade.py` fallback 가드 추가 | **완료** (`12ac862`) |
| **RDKit 디스크립터 차원 불일치** | 최신 RDKit(217개) vs 사전학습 D-MPNN(210개) 불일치로 인한 `RuntimeError: running_mean` 완벽 분석 및 `[:210]` 슬라이싱 보정 | **완료** (`b1f1ca8`) |
| **실시간 진행도 로깅** | D-MPNN 표현 추출 및 CatBoost 9개 앙상블 학습 과정 실시간 콘솔 flush(`flush=True`) 적용 | **완료** (`8bed6f3`) |
| **Git 저장소 정리** | 모든 변경 사항 커밋 및 `origin/main` 푸시 완료 (Working tree clean) | **완료** (`3d4e098`) |

---

## 2. 🔍 Colab 원격 실행 타임아웃 원인 분석 (Root Cause)

1. **`colab install` CLI 타임아웃 (30초 제한)**
   - `colab-cli`의 `automation.py` 내부 `execute_interactive`는 기본 타임아웃이 30초로 하드코딩되어 있습니다.
   - 단일 패키지 설치는 몇 초 만에 완료되나, 여러 패키지를 한꺼번에 설치할 경우 30초 제한에 걸려 `TimeoutError: Timeout waiting for output`이 발생합니다.
2. **`pip` vs `uv` 속도 격차**
   - Colab VM 내부에서 일반 `pip install --upgrade`는 PyTorch 의존성 트리 전체를 재탐색하여 10분(600초) 이상 소요됩니다.
   - 반면 VM 내부 사전 설치된 `uv pip install --system` 또는 개별 패키지 `colab install`을 활용하면 수 초 내에 즉시 설치가 완료됩니다.
3. **Colab 세션 계정 상태**
   - `munhyeongdo4@gmail.com`: 무료 GPU 일일 할당량 소진 (`TooManyAssignmentsError`).
   - `20172417@gm.hannam.ac.kr`: 신규 세션 생성 완료 (`sota-training` / `gpu-t4-s-kkb-usw1b2-3ef9zw50xqj76`).

---

## 3. 🎯 차기 작업 TODOLIST

### [Phase 1] Colab 원격 환경 의존성 원클릭 고속 초기화 [완료]
- `scikit-learn`, `catboost`, `PyTDC`, `torch-geometric`, `rdkit`, `transformers`를 `colab install`로 고속 구성 완료.
- 커널 재기동(`colab restart-kernel -s sota-training`) 완료.

### [Phase 2] 간세포 클리어런스(Hepatocyte Clearance) 최종 원격 학습 [완료]
- D-MPNN 210차원 전이 피처 기반 Tri-Objective CatBoost 앙상블 학습 완료 (`deploy/train_clearance_cascade.py`).
- **최종 검증 성과 (Bemis-Murcko Scaffold Split)**:
  - **Spearman $\rho$**: **0.5847** (공식 가이드 목표치 $\ge 0.45$ 초과 달성!)
  - **Pearson $r$**: 0.5384
  - **MAE**: 27.5106
  - **RMSE**: 40.9603
  - **$R^2$**: 0.2720

### [Phase 3] 학습 아티팩트 다운로드 및 동기화 [완료]
- 모델 산출물 로컬 동기화 완료:
  - `models/export/clearance_cascade/clearance_cascade_summary.json`
  - `models/export/clearance_cascade/clearance_cascade_model.joblib` (3.7 MB)

### [Phase 4] 지질친화도(Lipophilicity) Tri-Hybrid 스태킹 고도화 [완료]
- `colab_exec.ps1` UTF-8 without BOM 인코딩 보정 및 `deploy/train_lipo_stacking.py`, `tdc_studio/models/hybrid/lipo_stacker.py` GPU ChemBERTa 가속/D-MPNN 210차원 슬라이싱 적용.
- 원격 GPU 학습 완료 (`deploy/train_lipo_stacking.py`).
- **최종 검증 성과 (Bemis-Murcko Scaffold Split)**:
  - **$R^2$**: 0.6004
  - **Pearson $r$**: 0.7879
  - **Spearman $\rho$**: 0.7738
  - **MAE**: 0.5887
  - **RMSE**: 0.7502
  - **Stack Weights**: [0.616 (Motifs GBDT), 0.355 (ChemBERTa), 0.029 (D-MPNN)]
- 산출물 동기화 완료:
  - `models/export/lipophilicity_stacker/lipophilicity_stacking_summary.json`
  - `models/export/lipophilicity_stacker/lipo_stacker_model.joblib` (2.4 MB)
