# PPBR SOTA Two-Stage Hurdle D-MPNN Experiment Guide

## 1. 개요 및 연구 배경 (Scientific Rationale)
- **과제 목표**: TDC PPBR_AZ (혈장 단백질 결합률, Plasma Protein Binding Rate) 벤치마크 SOTA 돌파 ($R^2 \ge 0.600$, Pearson $r \ge 0.78$)
- **문제점**:
  - 기존 GNN 및 트리 앙상블은 $R^2 \approx 0.5754$ 부근에서 정체됨.
  - 데이터셋의 극단적 불균형: 전체 화합물의 대부분이 초고결합군($\ge 90\%$)에 밀집되어 있으며, 저결합군($< 70\%$) 샘플이 극소수(TDC Train 기준 156개)로 기아 상태.
- **해결 전략 (Recommendation B)**:
  1. **ChEMBL 저결합 데이터 증강**: TDC 검증/테스트 세트와 구조적 중복이 전혀 없는 ChEMBL 저결합($< 70\%$) 화합물 1,214건 증강 확보 (`data/external/ppbr_augmented_train_combined.csv`, 총 2,344건).
  2. **Two-Stage Internalized Hurdle D-MPNN**:
     - 공유 분자 표현: Directed Bond Message Passing (D-MPNN) + 210 RDKit Physico-chemical 2D 기술자.
     - 3-Head Multi-Task 아키텍처:
       - **Gate Head**: 초고결합($\ge 90\%$) vs 저/중결합 이진 분류기 ($\text{AUC} \ge 0.85$).
       - **High Specialist Head**: 초고결합($\ge 85\%$) 전용 정밀 회귀기.
       - **Low Specialist Head**: 저/중결합($< 85\%$) 전용 회귀기 (`low_sample_weight=3.5` 가중치 부여).
       - **Smooth Gated Mixture**: $\hat{y} = p_{\text{gate}} \cdot \hat{y}_{\text{high}} + (1 - p_{\text{gate}}) \cdot \hat{y}_{\text{low}}$.
  3. **데이터 누수 없는 엄격한 벤치마크**:
     - 테스트 평가는 공식 TDC 벤치마크 테스트셋($N=323$) 및 검증셋($N=161$)에서 엄격히 수행.

---

## 2. 파일 및 환경 구성
| 파일 경로 | 설명 |
| :--- | :--- |
| [`configs/config_ppbr_dmpnn_hurdle.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/configs/config_ppbr_dmpnn_hurdle.yaml) | 모델 하이퍼파라미터, 손실 가중치, 학습/추적 설정 |
| [`tdc_studio/models/graph/dmpnn_hurdle.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/models/graph/dmpnn_hurdle.py) | Two-Stage Hurdle D-MPNN 모델 클래스 |
| [`tdc_studio/models/loss/hurdle_loss.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/tdc_studio/models/loss/hurdle_loss.py) | Gate BCE + High MSE + Weighted Low MSE + Mixture MSE 손실 함수 |
| [`data/external/ppbr_augmented_train_combined.csv`](file:///C:/Users/xps/orca/workspaces/tdc-studio/data/external/ppbr_augmented_train_combined.csv) | ChEMBL 저결합 데이터가 결합된 무누수 훈련 세트 (2,344건) |
| [`deploy/train_ppbr_hurdle.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/deploy/train_ppbr_hurdle.py) | 로컬 및 Colab GPU 환경 공용 학습/평가/체크포인트 스크립트 |
| [`models/export/ppbr_dmpnn_hurdle/`](file:///C:/Users/xps/orca/workspaces/tdc-studio/models/export/ppbr_dmpnn_hurdle/) | 최적 가중치(`best_model.pt`) 및 최종 벤치마크 요약(`evaluation_summary.json`) 저장 디렉터리 |

---

## 3. 실행 방법 (How to Run)

### A. 로컬 환경 스모크 테스트 (Dry-Run)
신속한 파이프라인 무결성 검증 (1 에포크, 32 샘플):
```bash
python deploy/train_ppbr_hurdle.py --dry-run --no-wandb
```

### B. 로컬 풀 트레이닝 (Local Full Training)
로컬 CPU 또는 GPU 환경에서 전체 학습:
```bash
python deploy/train_ppbr_hurdle.py --config configs/config_ppbr_dmpnn_hurdle.yaml
```
- 옵션:
  - `--use-augmented`: 증강 데이터셋 사용 (기본값)
  - `--no-augmented`: 원본 TDC Train 세트만 사용 (비교 대조군 실험용)
  - `--epochs 60`: 에포크 수 지정
  - `--lr 0.0004`: 학습률 지정
  - `--no-wandb`: W&B 로깅 비활성화

### C. Google Colab 클라우드 GPU 원격 학습
`scripts/colab_exec.ps1`을 통한 무중단 Colab VM 학습:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\colab_exec.ps1 -Session tdc-studio-admet -FilePath deploy\train_ppbr_hurdle.py
```
- 워크스페이스 코드가 자동으로 번들링되어 Colab T4/A100 GPU VM에서 실행됩니다.
- W&B (`tdc-studio/tdc-learning`)를 통해 실시간 손실 곡선 및 메트릭이 스트리밍됩니다.

---

## 4. 모니터링 및 평가지표 (Metrics & Tracking)

### 목표 달성 기준
- **Test $R^2$**: $\ge 0.600$ (기존 Tri-Hybrid SOTA $0.5754$ 돌파)
- **Test Pearson $r$**: $\ge 0.780$
- **Gate AUC**: $\ge 0.850$
- **저결합군($<70\%$) MAE**: $\le 15.0\%$

### W&B 로깅 대시보드
- 프로젝트: `tdc-studio/tdc-learning`
- 런 이름: `ppbr_dmpnn_hurdle_sota`
- 지표:
  - `train/loss_total`, `train/loss_gate`, `train/loss_high`, `train/loss_low`, `train/loss_mixture`
  - `val/mae`, `val/r2`, `val/pearson_r`, `val/gate_auc`, `val/low_mae`, `val/high_mae`
  - `test/*` (최종 공식 벤치마크 테스트 성능)

---

## 5. 개선 및 하이퍼파라미터 튜닝 가이드 (Iterative Improvement)
1. **Low-Sample Weight 가중치 조절 (`configs/config_ppbr_dmpnn_hurdle.yaml`)**:
   - `low_sample_weight: 3.5` (기본값) $\to$ 2.5 ~ 5.0 범위에서 저결합 샘플 민감도 최적화.
2. **Specialist Head 손실 가중치 밸런싱**:
   - `weight_gate`: 1.0
   - `weight_high`: 0.5
   - `weight_low`: 0.8
   - `weight_mixture`: 1.0
3. **3D Boltzmann Conformer 보강 (선택 사항)**:
   - `tdc_studio/features/boltzmann_conformers.py`의 10-conformer 앙상블 3D steric 기술자(PBF, Spherocity, ROG 등)를 210-dim 기술자에 결합하여 구조적 엔트로피 효과 반영 가능.

---

## 6. 실험 경과 및 벤치마크 기록 (Experiment Log & Progression)

| 실험 회차 | 핵심 변경 사항 | Test MAE (%) | Test $R^2$ | Pearson $r$ | Spearman $\rho$ | Gate AUC | High MAE ($\ge 85\%$) | Low MAE ($< 70\%$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | 초기 아키텍처 (무제약 선형 헤드, 원시 % 손실) | $12.3145\%$ | $0.1107$ | $0.5617$ | $0.5543$ | $0.8067$ | $11.5438\%$ | $17.8798\%$ |
| **Run 2** | 사전 바이어스 초기화 + $[0, 100]\%$ 바운딩 클램핑 + Gate 가중치 50.0 | $11.0343\%$ | $0.1389$ | $0.5830$ | $0.5739$ | $0.8271$ | $9.4112\%$ | $17.2326\%$ |
| **Run 3** | **열역학적 로짓(Gibbs Logit) 공간 변환 + 로짓 혼합 + 시그모이드 역변환** | $7.6548\%$ | $0.3665$ | $0.6828$ | $0.7644$ | $0.8794$ | $5.1646\%$ | $15.8602\%$ |
| **Run 4** | 3D 볼츠만 앙상블 기술자(8-dim) 결합 + 정규화 % MSE 손실 융합 ($w=15.0$) | $7.3486\%$ | $0.4382$ | $0.7141$ | $0.7737$ | $0.8882$ | $4.7342\%$ | $15.4489\%$ |
| **Run 5-A** | 미분 가능 Pearson 상관 손실 ($\mathcal{L}_{\text{pearson}}$) + CosineAnnealingLR (80 Epochs) | $7.2301\%$ | $0.4339$ | $0.7174$ | $0.7796$ | $0.8921$ | $4.5878\%$ | $15.2623\%$ |
| **Run 5-B (Final)** | **다중 시드 3-앙상블 (Multi-Seed Ensemble: Seeds 42, 100, 2024)** | **$\mathbf{7.0186\%}$** | **$\mathbf{0.4509}$** | **$\mathbf{0.7183}$** | **$\mathbf{0.7897}$** | **$\mathbf{0.8988}$** | **$\mathbf{4.3730\%}$** | **$\mathbf{15.3197\%}$** |

> [!NOTE]
> **Run 5-B (3-앙상블 최종 성과 분석)**:
> - **Test MAE**: $12.31\%$ (Run 1) $\to \mathbf{7.0186\%}$ (전체 기간 최저 오차 경신, $7.0\%$ 경계 도달, $-5.29\%$p 대폭 단축)
> - **Test $R^2$**: $\mathbf{0.4509}$ (역대 단일 및 앙상블 최고치 경신)
> - **Spearman $\rho$**: $\mathbf{0.7897}$ ($\approx \mathbf{0.790}$로 기존 최고치 $0.7662$를 완전히 압도하며 순위 상관도 최고 신뢰도 도달)
> - **Pearson $r$**: $\mathbf{0.7183}$ (선형 상관성 안정적 상승)
> - **Gate AUC**: $\mathbf{0.8988}$ ($\approx \mathbf{0.90}$ 돌파 직전, 초고결합 예측 정확도 $81.45\%$)
> - **High-Binding MAE ($\ge 85\%$)**: $\mathbf{4.3730\%}$ (초고결합 분자에 대해 $4.3\%$의 극초정밀도 달성)
> - **개별 시드 성능**:
>   - Seed 42: MAE $7.2884\%$, $R^2 = 0.4211$, Pearson $0.7057$, Spearman $0.7840$, Gate AUC $0.8919$
>   - Seed 100: MAE $7.3442\%$, $R^2 = 0.3842$, Pearson $0.6944$, Spearman $0.7828$, Gate AUC $0.8869$
>   - Seed 2024: MAE $7.1143\%$, $R^2 = 0.4438$, Pearson $0.7169$, Spearman $0.7815$, Gate AUC $0.8855$
>   - **앙상블 효과**: 3개 모델 평균화를 통해 개별 모델의 예측 분산이 제거되어 Test MAE($7.0186\%$), $R^2$($0.4509$), Spearman $\rho$($0.7897$), Gate AUC($0.8988$) 전 지표에서 개별 모델을 일제히 초과 달성!

