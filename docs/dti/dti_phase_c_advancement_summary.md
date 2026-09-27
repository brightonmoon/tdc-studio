# TDC-Studio DTI Phase C 고도화 완료 보고서 (Tasks F-1, F-2, F-3)

> **작성 일자**: 2026-09-27  
> **수행 환경**: Google Colab GPU (`munhyeongdo4@gmail.com`, 세션 `dti-gpu`, Tesla T4 14.6 GB VRAM)  
> **평가 데이터셋**: TDC `BindingDB_Kd` (Cold-Drug Split: Train 34,701 / Valid 7,217 / Test 10,356)  

---

## 1. 배경 및 과제 목표

Phase C의 목표는 사전학습 단백질-화학 파운데이션 모델(ChemBERTa-77M-MTR, ESM-2 35M) 간 양방향 교차 어텐션(CrossAttentionFusion)을 통해 미지의 신약 골격(Cold-Drug)에 대한 결합 친화도를 정확하게 예측하고, 원자-잔기 상호작용의 설명가능성(XAI)을 확보하는 것입니다.

본 작업에서는 `docs/dti/dti_phase_c_todolist.md`의 후속 고도화 백로그 3대 과제를 완수하였습니다:
- **Task F-1**: 풀링 전 시퀀스 토큰 기반의 2D Contact Map 추출 파이프라인 구현 (XAI)
- **Task F-2**: ChemBERTa 상위 2개 RoBERTa 레이어 Staged Unfreezing 및 차별적 학습률 미세조정 실험 (정규화 강화)
- **Task F-3**: BindingDB $K_d, K_i, \text{IC}_{50}$ 3대 지표 통합 멀티태스크 아키텍처 및 손실 함수(`MaskedMSELoss`) 구현

---

## 2. 아키텍처 구현 내용

### 2.1 Task F-1: Full Token-Level Cross-Attention & 2D Contact Map
- **인코더 시퀀스 보존**:
  - `ChemBERTaEncoder`: `return_sequence=True` 설정 시 풀링 이전의 원자 토큰 시퀀스 `[B, L_drug, D]`를 선형 투영하여 반환. 패딩 토큰은 `attention_mask.unsqueeze(-1)`로 zero-out 처리.
  - `ESM2Encoder`: 아미노산 토큰 시퀀스 `[B, L_target, D]`를 투영하여 반환하며 동일하게 패딩 마스킹 적용.
- **2D Contact Map 투영**:
  - `CrossAttentionFusion`: `nn.MultiheadAttention`에 `key_padding_mask`를 동적 적용하여 실제 유효 잔기/원자 간의 어텐션 가중치만을 정밀 계산.
  - 마스킹 평균 풀링(Masked Mean Pooling)으로 시퀀스 길이에 무관한 안정적 특징 벡터 추출.

### 2.2 Task F-2: Staged Fine-Tuning 최적화
- **2단계 전이학습 스케줄러**:
  - Stage 1 (Epoch 1 ~ 11): 백본 동결 하에 CrossAttention Fusion 헤드 집중 최적화 ($LR = 1.0 \times 10^{-4}$).
  - Stage 2 (Epoch 12 ~ 14): ChemBERTa 상위 2개 RoBERTa 레이어 Unfreezing ($LR = 1.0 \times 10^{-6}$, `weight_decay: 0.05`).
- **안정화 및 버그 방지**:
  - `warmup_epochs: 2` 동안의 초기 가중치 요동에 의한 조기 체크포인트 고착 방지.
  - PyTorch `optimizer.add_param_group`의 중복 파라미터 충돌 방지를 위해 엄격한 Disjoint 집합 분리 적용.

### 2.3 Task F-3: Multi-Affinity Multi-Task Architecture
- **데이터로더**: `MultiAffinityDTADataModule`을 통해 $K_d, K_i, \text{IC}_{50}$ 어세이를 단일 데이터프레임으로 병합하고 누수 없는 Cold-Drug 분할 적용.
- **손실 함수**: 부분적으로만 측정된 관측치만을 선택적으로 역전파하는 `MaskedMSELoss` 구현.
- **서빙 스키마**: `DTIMultiAffinityInferenceRequest`, `DTIMultiAffinityInferenceResponse` 구현.

---

## 3. 실증 벤치마크 및 실험적 발견 (Cold-Drug Split)

### 3.1 모델 성능 비교표

| 모델 / 파이프라인 | 학습 전략 | Cold-Drug CI (↑) | Cold-Drug MSE (↓) | Pearson r (↑) | XAI 2D Contact Map | 비고 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase B Baseline** | GINE + ProteinCNN | 0.7464 | 0.6609 | 0.6120 | ❌ (단방향) | 이전 기준 모델 |
| **Phase C Option A** | Frozen ChemBERTa + ESM-2 + Head 15에포크 | **0.7659** | **0.5780** | **0.6885** | ✅ (완전 호환) | **프로덕션 서빙 SOTA** |
| **Phase C Option B (Adv)** | Staged Fine-Tuning (Top-2 RoBERTa 레이어) | 0.7443 | 0.6404 | 0.6427 | ✅ (완전 지원) | 미세조정 실증 검증 |

### 3.2 핵심 과학적 발견: Scaffold Overfitting
- **현상**: ChemBERTa의 상위 트랜스포머 레이어를 미세조정하면 학습 손실은 `0.72`에서 `0.34`로 지속적으로 감소하지만, Validation 및 Test 일반화 성능은 오히려 감소함.
- **원인 분석**: 평가 데이터가 학습 데이터에 존재하지 않는 신규 골격(Cold-Drug)이므로, ChemBERTa의 7,700만 범용 화학 표현 공간을 완벽히 동결하고 상위 Cross-Attention 상호작용 레이어만 학습시키는 전략(Option A)이 신규 분자 구조에 대한 외삽(Extrapolation) 능력을 극대화함.
- **결론**: 프로덕션 서빙에는 Phase C Option A 모델(`models/dti/phase_c/best_model.pt`)을 주력으로 사용하며, Task F-1의 2D Contact Map 추출기와 Task F-3의 멀티태스크 스키마를 해당 서빙 엔진에 결합하는 것이 최적의 설계임.

---

## 4. 검증 결과

- **단위 테스트**: `pytest` 전체 93개 테스트 스위트 100% 통과 (`93 passed in 38.67s`)
- **코드 품질**: `ruff check` (0 errors), `ruff format` (100% 포맷팅 준수)
