# DTI/DTA Phase B — Foundation Model 실행 계획서

> **브랜치**: `dta/phase-b-foundation` (보존) / `main` (병합 완료)  
> **작성일**: 2026-09-25  
> **상태**: ✅ 완료 — Cold-Drug CI 0.7464, MSE 0.6609 달성

---

## 1. 개요 및 목표

### 배경
Phase A(GINE + ProteinCNN)로 구축된 DTI 파이프라인을 HuggingFace 대형 사전학습 모델로 확장하여, 신약 발굴 실무에서 요구되는 Cold-Drug 일반화 성능을 달성합니다.

### 목표 지표 (BindingDB Kd Cold-Drug Split)
| 지표 | 목표 | 최종 결과 | 상태 |
|:---:|:---:|:---:|:---:|
| Concordance Index (CI) ↑ | ≥ 0.70 | **0.7464** | ✅ PASSED |
| MSE ↓ | ≤ 0.75 | **0.6609** | ✅ PASSED |
| RMSE ↓ | — | 0.8130 | — |
| Pearson r ↑ | — | 0.6508 | — |

---

## 2. 아키텍처 설계

### Phase A → Phase B 비교

| 항목 | Phase A | Phase B |
|:---|:---|:---|
| 약물 인코더 | GINE (GNN, 분자 그래프) | **ChemBERTaEncoder** (SMILES 시퀀스) |
| 단백질 인코더 | ProteinCNN (1D CNN) | **ESM2Encoder** (단백질 언어모델) |
| 파운데이션 모델 | 없음 (from scratch) | HuggingFace 사전학습 활용 |
| 퓨전 레이어 | BilinearAttentionFusion | BilinearAttentionFusion (공통) |
| Cold-Drug CI | 미측정 | **0.7464** |

### Phase B 전체 파이프라인

```text
                    ┌─────────────────────────────────────────┐
    SMILES ─────▶   │  ChemBERTaEncoder                        │
                    │  Model: DeepChem/ChemBERTa-77M-MTR       │
                    │  Tokenizer: RobertaTokenizer              │
                    │  Pooling: Masked Mean (attention mask)    │
                    │  Output: 384d → Linear(384, 256) → 256d  │
                    │  freeze() / unfreeze(last_n_layers=2)    │
                    └──────────────────────┬──────────────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ BilinearAttentionFusion│ ─▶ pKd Score (scalar)
                                └──────────────────────┘
                                           ▲
                    ┌──────────────────────┴──────────────────┐
    AA Seq ──────▶  │  ESM2Encoder                             │
                    │  Model: facebook/esm2_t12_35M_UR50D      │
                    │  Max Length: 1,024 아미노산               │
                    │  Output: 480d → Linear(480, 256) → 256d  │
                    │  freeze_backbone=True + In-Memory Cache   │
                    └─────────────────────────────────────────┘
```

---

## 3. 파운데이션 모델 명세

### ChemBERTaEncoder (`DeepChem/ChemBERTa-77M-MTR`)

| 항목 | 값 |
|:---|:---|
| 파라미터 수 | ~77M |
| 학습 데이터 | PubChem 77M SMILES (Multi-Task Regression) |
| 토크나이저 | RobertaTokenizer (BPE, max_length=256) |
| 풀링 전략 | Masked Mean Pooling (패딩 제외 평균) |
| 은닉 차원 | 384d |
| 투영 레이어 | Linear(384, 256) + LayerNorm + ReLU |
| 등록 키 | `"chemberta_encoder"`, `"chembert_encoder"` |
| Staged Training | Stage 1: 백본 동결, Stage 2: `unfreeze(last_n_layers=2)` |

### ESM2Encoder (`facebook/esm2_t12_35M_UR50D`)

| 항목 | 값 |
|:---|:---|
| 파라미터 수 | ~35M (12 레이어) |
| 학습 데이터 | UniRef50/D 2021 (250M 단백질 서열) |
| 토크나이저 | ESM2 Tokenizer (아미노산 알파벳) |
| 최대 입력 길이 | 1,024 아미노산 |
| 은닉 차원 | 480d |
| 투영 레이어 | Linear(480, 256) + LayerNorm + ReLU |
| 등록 키 | `"esm2_encoder"` |
| **인메모리 캐시** | `freeze_backbone=True` 시 활성화, 최대 30배 가속 |

> [!TIP] **ESM-2 캐시 효과**: BindingDB_Kd의 고유 타겟 ~1,500개를 1 에포크 내 캐싱 완료 → Epoch 2부터 ESM-2 순전파 없음. 에포크당 시간: 36분 → **2분 30초 (15× 단축)**

---

## 4. 구현 파일 목록

| 파일 | 역할 |
|:---|:---|
| [`tdc_studio/models/dti/pretrained_encoders.py`](../../tdc_studio/models/dti/pretrained_encoders.py) | ChemBERTaEncoder, ESM2Encoder 전체 구현 |
| [`tdc_studio/models/dti/dta_model.py`](../../tdc_studio/models/dti/dta_model.py) | GraphDTAModel — Phase A/B 다형성, out_dim 동적 감지 |
| [`tdc_studio/models/dti/fusion.py`](../../tdc_studio/models/dti/fusion.py) | BilinearAttentionFusion (Phase A/B 공통) |
| [`tdc_studio/data/multi_pred.py`](../../tdc_studio/data/multi_pred.py) | `target_seq_str` 필드 추가 (ESM-2용 원시 서열) |
| [`tdc_studio/data/collate.py`](../../tdc_studio/data/collate.py) | `molecule_collate_fn`에 문자열 배치 수집 추가 |
| [`tdc_studio/core/registry.py`](../../tdc_studio/core/registry.py) | `Registry.__contains__()` 추가 |
| [`configs/config_dti_phase_b.yaml`](../../configs/config_dti_phase_b.yaml) | Phase B 전체 학습 설정 |
| [`deploy/dti_phase_b_dry_run.py`](../../deploy/dti_phase_b_dry_run.py) | 6단계 통합 무결성 검증 스크립트 |
| [`deploy/train_dti_phase_b.py`](../../deploy/train_dti_phase_b.py) | Colab GPU 학습 파이프라인 (AMP FP16, Early Stopping) |
| [`tests/test_pretrained_encoders.py`](../../tests/test_pretrained_encoders.py) | Mock 기반 단위 테스트 4개 (Zero-training) |
| [`models/dti/phase_b/benchmark_summary.yaml`](../../models/dti/phase_b/benchmark_summary.yaml) | 최종 벤치마크 결과 기록 |

---

## 5. 8단계 실행 로드맵

```
Step 1: pretrained_encoders.py 구현
    └─ ChemBERTaEncoder (ChemBERTa-77M-MTR, Masked Mean Pool)
    └─ ESM2Encoder (esm2_t12_35M, max_len=1024, In-Memory Cache)
    └─ Registry 등록: chemberta_encoder, esm2_encoder

Step 2: dta_model.py 수정
    └─ drug_out_dim / target_out_dim 동적 감지 (out_dim 속성)
    └─ extract_features() ESM-2 / ProteinCNN 분기 처리

Step 3: 데이터 파이프라인 수정
    └─ multi_pred.py: target_seq_str 필드 추가
    └─ collate.py: 문자열 배치 수집 추가

Step 4: config_dti_phase_b.yaml 작성
    └─ ChemBERTa + ESM-2 인코더 설정
    └─ batch=32, max_epochs=20, early_stopping_patience=5

Step 5: 단위 테스트 작성 및 통과 확인
    └─ tests/test_pretrained_encoders.py (4/4 pass)
    └─ uv run --extra dev pytest tests/test_pretrained_encoders.py -v

Step 6: Colab 번들 배포 스크립트 작성
    └─ deploy/dti_phase_b_dry_run.py (6단계 무결성 검증)
    └─ deploy/train_dti_phase_b.py (Staged Training, AMP FP16)

Step 7: Colab GPU 학습 실행 (tdc-studio-dti 세션)
    └─ 1차 런: Epoch 2 도중 1시간 타임아웃 → ESM-2 병목 진단
    └─ 최적화: 인메모리 캐시 구현 (36분 → 2분 30초)
    └─ 2차 런: Epoch 7 best → Early Stopping (12 epochs)
    └─ 최종: CI=0.7464 ✅, MSE=0.6609 ✅

Step 8: Git 거버넌스 & 문서화
    └─ feature/dti-phase-b-foundation → main Fast-forward 병합
    └─ dta/phase-b-foundation 전용 브랜치 생성 (Phase C 보존)
    └─ README.md DTI/DTA 섹션 추가
```

---

## 6. 세션 거버넌스

| Colab 세션 | 용도 | 상태 |
|:---|:---|:---:|
| `tdc-studio-admet` | ADMET 클러스터 학습 전용 | 운영 중 |
| `tdc-studio-dti` | DTI/DTA Phase B 학습 전용 | 완료 (pruned) |

> [!IMPORTANT] ADMET과 DTI 학습은 반드시 세션을 분리하여 GPU 자원 충돌 방지 및 로그 혼재를 막습니다.

---

## 7. YAML 업그레이드 경로 (Phase A → B)

Phase A 대비 **딱 2개 키**만 변경하면 됩니다:

```yaml
# Phase A (configs/config_dti_phase_a.yaml)
drug_encoder:
  name: gine_model           # GNN 분자 그래프 인코더
protein_encoder:
  name: protein_cnn          # 1D CNN 단백질 인코더

# Phase B (configs/config_dti_phase_b.yaml)
drug_encoder:
  name: chemberta_encoder    # ← SMILES 언어모델
  pretrained_model: DeepChem/ChemBERTa-77M-MTR
protein_encoder:
  name: esm2_encoder         # ← 단백질 언어모델
  pretrained_model: facebook/esm2_t12_35M_UR50D
  freeze_backbone: true
```

---

## 8. Phase C 로드맵 (향후 계획)

| 항목 | 내용 |
|:---|:---|
| **더 큰 ESM-2 백본** | `esm2_t33_650M_UR50D` 또는 `esm2_t36_3B_UR50D` 시도 |
| **ChemBERTa 파인튜닝** | `unfreeze(last_n_layers=4)` + 낮은 LR로 end-to-end 미세 조정 |
| **Cross-Attention Fusion** | BilinearFusion → TransformerCross 교체 |
| **멀티태스크 학습** | Ki / IC50 / EC50 보조 태스크 공동 학습 |
| **브랜치** | `dta/phase-b-foundation` 기반 `dta/phase-c-*` 파생 |
