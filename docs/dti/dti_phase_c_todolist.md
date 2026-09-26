# TDC-Studio DTI Phase C: Future TODOLIST Backlog (옵션 B & C 이관)

> **이관 일자**: 2026-09-26  
> **상태**: 추후 진행 (오늘 작업에서는 제외)  
> **대상 모듈**: DTA/DTI 트랙 (`tdc_studio/models/dti`, `tdc_studio/data`, `deploy`)

---

## 📌 배경 및 현황
- **오늘 완료된 작업 (옵션 A)**:
  - Cross-Attention Fusion 아키텍처 및 XAI Attention Map 추출 구현.
  - ChemBERTa/ESM-2 백본을 Frozen 상태로 유지하고 Cross-Attention Fusion 헤드만 15 에포크 동안 충분히 수렴 (Scaffold Overfitting 방지).
  - FastAPI 프로덕션 서빙 파이프라인 연동 (`/predict/dti`).
- **추후 이관 사유**:
  - 오늘 작업 범위에서는 토큰 및 컴퓨팅 자원을 최소화하고 안정된 서빙 파이프라인을 조기 완성하기 위해, 백본 미세조정(옵션 B) 및 멀티태스크 확장(옵션 C)은 추후 고도화 단계로 분리하여 관리함.

---

## 📋 추후 작업 목록 (Future Backlog)

### 1. [옵션 B] 정규화 강화 Staged Fine-Tuning
- **우선순위**: P2 (Phase C 옵션 A 수렴 모델 성능 검토 후 필요 시 착수)
- **목표**: 미지의 신약 골격(Unseen Scaffold)에 대한 ChemBERTa 표현력 미세조정 및 Cold-Drug CI 0.77+ 돌파.
- **주요 작업 내용**:
  1. **Layer-wise Discriminative Learning Rate**:
     - CrossAttention Fusion 헤드: $5.0 \times 10^{-5}$
     - ChemBERTa 상위 2개 RoBERTa 레이어: $2.0 \times 10^{-6}$ (매우 완만한 미세조정으로 Scaffold 과적합 방지)
  2. **정규화 및 스케줄링 강화**:
     - Backbone 파라미터 대상 `weight_decay: 0.05` 적용.
     - Warmup 기간을 4~6 에포크로 확장하여 Cross-Attention 헤드가 완벽히 안정된 후 Unfreezing 진행.
  3. **대상 파일**:
     - [`configs/config_dti_phase_c.yaml`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/configs/config_dti_phase_c.yaml)
     - [`deploy/train_dti_phase_c.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/deploy/train_dti_phase_c.py)

---

### 2. [옵션 C] Multi-Affinity Multi-Task 확장 ($K_d + K_i + \text{IC}_{50}$)
- **우선순위**: P3 (데이터 확장 및 범용 친화도 예측 모델 구축)
- **목표**: BindingDB 내의 $K_d, K_i, \text{IC}_{50}$ 데이터를 통합하여 데이터 볼륨을 3배 이상 확장하고 일반화 오차 극소화.
- **주요 작업 내용**:
  1. **멀티태스크 데이터로더 구축**:
     - [`tdc_studio/data/multi_pred.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/tdc_studio/data/multi_pred.py) 내 `MultiAffinityDTADataModule` 구현.
     - 동일 약물-타깃 쌍의 다중 측정치 통합 및 스케일러 정규화.
  2. **Multi-Head 아키텍처 및 손실 함수**:
     - [`tdc_studio/models/dti/dta_model.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/tdc_studio/models/dti/dta_model.py)에 3개 태스크별 선형 헤드 분기.
     - 결측 레이블을 무시하는 `MaskedMSELoss` 구현.
  3. **서빙 스키마 및 API 확장**:
     - [`tdc_studio/serving/schema.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/tdc_studio/serving/schema.py)에 $K_i, \text{IC}_{50}$ 필드 추가.
     - [`tests/test_multi_task_dti.py`](file:///C:/Users/xps/orca/workspaces/tdc-studio/greenling/tests/test_multi_task_dti.py) 단위 테스트 작성.
