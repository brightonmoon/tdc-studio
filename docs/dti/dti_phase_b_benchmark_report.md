# DTI/DTA Phase B — Foundation Model 기반 Drug-Target 결합력 예측 벤치마크 리포트

> **문서 목적**: 이 문서는 HuggingFace ChemBERTa(화합물) 및 ESM-2(단백질) 사전학습 파운데이션 모델을 결합한 TDC-Studio DTI/DTA Phase B 파이프라인의 아키텍처 설계, BindingDB Kd Cold-Drug 벤치마크 실측 성과, 단위 테스트 및 서빙 실행 명세를 기록한 종합 리포트입니다.

---

## 1. 개요 및 설계 아키텍처

TDC-Studio는 ADMET 예측을 넘어, **Drug-Target Affinity (DTA)** 예측을 위한 대형 사전학습(Foundation) 모델 파이프라인을 제공합니다. Phase B는 HuggingFace 기반의 ChemBERTa(화합물) + ESM-2(단백질) 파운데이션 인코더를 결합하여 **BindingDB Kd Cold-Drug 벤치마크에서 CI ≥ 0.70을 달성**하는 것을 목표로 설계되었습니다.

### 아키텍처 다이어그램

```text
                    ┌─────────────────────────┐
    SMILES ─────▶   │  ChemBERTaEncoder        │  ─────┐
                    │  DeepChem/ChemBERTa-77M  │       │   ┌──────────────────────────┐
                    │  384d → 256d Projection  │       ├─▶ │  BilinearAttentionFusion  │ ─▶ pKd Score
                    └─────────────────────────┘       │   └──────────────────────────┘
                    ┌─────────────────────────┐       │
    AA Seq ──────▶  │  ESM2Encoder             │  ─────┘
                    │  facebook/esm2_t12_35M   │
                    │  480d → 256d Projection  │
                    │  + In-Memory Cache       │
                    └─────────────────────────┘
```

### 핵심 컴포넌트 규격

| 컴포넌트 | HuggingFace 모델 ID | 출력 차원 | 주요 특징 |
|:---|:---|:---:|:---|
| **ChemBERTaEncoder** | `DeepChem/ChemBERTa-77M-MTR` | 256 | Masked Mean Pooling, `freeze()`/`unfreeze(n)` 지원 |
| **ESM2Encoder** | `facebook/esm2_t12_35M_UR50D` | 256 | max_length=1024, **고유 서열 인메모리 캐시** (15× 가속) |
| **BilinearAttentionFusion** | — | 1 | 약물·단백질 교차 어텐션 융합 후 pKd 회귀 |

> [!TIP]
> **YAML 2줄로 Phase A → B 파운데이션 업그레이드**:  
> `drug_encoder` / `protein_encoder` 설정만 변경하면 기존 Phase A (GINE + ProteinCNN) 파이프라인에서 Phase B 파운데이션 파이프라인으로 즉시 전환됩니다.

---

## 2. 🏆 BindingDB Kd Cold-Drug 벤치마크 결과

> 공식 TDC Cold-Drug Split (학습 약물과 화학 골격이 완전히 다른 신약 후보 화합물 평가) 기준의 최종 테스트 결과입니다.

| Phase | 약물 인코더 | 단백질 인코더 | Cold-Drug CI ↑ | MSE ↓ | RMSE ↓ | Pearson r ↑ |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **Phase A** | GINE (GNN) | ProteinCNN | — | — | — | — |
| **Phase B** | ChemBERTa-77M | ESM-2 35M | **0.7464 ✅** | **0.6609 ✅** | **0.8130** | **0.6508** |
| **목표 규격** | — | — | $\ge 0.70$ | $\le 0.75$ | — | — |

- 📄 **상세 결과 요약 YAML**: [`models/dti/phase_b/benchmark_summary.yaml`](../../models/dti/phase_b/benchmark_summary.yaml)
- ⚙️ **학습 설정 파일**: [`configs/config_dti_phase_b.yaml`](../../configs/config_dti_phase_b.yaml)
- 🚀 **Colab 학습 스크립트**: [`deploy/train_dti_phase_b.py`](../../deploy/train_dti_phase_b.py)
- 📋 **Phase B 마스터 계획서**: [`docs/dti/dti_phase_b_plan.md`](dti_phase_b_plan.md)

---

## 3. 학습 및 검증 가이드

### (1) 로컬 무결성 사전 점검 (Dry-run)
```bash
# 모델 레지스트리, DataModule, forward/backward 통합 검증
uv run python deploy/dti_phase_b_dry_run.py
```

### (2) Colab GPU 원격 학습 (세션 분리 권장)
```bash
# tdc-studio-dti 전용 세션으로 T4/A100 GPU 학습 실행
# (ADMET 학습과 Colab 세션 분리: tdc-studio-admet / tdc-studio-dti)
uv run python -c "
from tdc_studio.remote.colab_runner import ColabRunner
runner = ColabRunner(session='tdc-studio-dti')
runner.run_script('deploy/train_dti_phase_b.py',
                  config='configs/config_dti_phase_b.yaml')
"
```

### (3) YAML 설정 전환 예시 (Phase A → Phase B)
```yaml
# configs/config_dti_phase_b.yaml
drug_encoder:
  name: chemberta_encoder      # ← Phase A: gine_model
  pretrained_model: DeepChem/ChemBERTa-77M-MTR

protein_encoder:
  name: esm2_encoder           # ← Phase A: protein_cnn
  pretrained_model: facebook/esm2_t12_35M_UR50D
  freeze_backbone: true
```

### (4) 단위 테스트 (Zero-training, Mock 기반 빠른 검증)
```bash
uv run --extra dev pytest tests/test_pretrained_encoders.py tests/test_serving_api.py tests/test_cross_attention_fusion.py -v
# 예상 결과: 17 passed
```

### (5) 프로덕션 DTI 서빙 엔드포인트 호출 (/predict/dti)
```bash
# 1. 서빙 서버 구동 (DTI 모델 디렉토리 지정 시 자동 로딩)
MODEL_DIR=models/export/dti_phase_c uv run tdc-studio serve --port 8000

# 2. DTI 결합 친화도 추론 요청 (pKd 및 Kd nM 산출)
curl -X POST "http://localhost:8000/predict/dti" \
     -H "Content-Type: application/json" \
     -d '{
       "smiles": ["CC1=C(C(=O)N2CCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"],
       "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD"],
       "return_kd_nm": true
     }'
```

---

## 4. 브랜치 거버넌스

| 브랜치 명칭 | 운영 목적 |
|:---|:---|
| `main` | Phase B 검증 완료 코드 포함 (Fast-forward 병합 완료) |
| `dta/phase-b-foundation` | DTI/DTA Phase C 연구를 위한 전용 보존 브랜치 |
| `brightonmoon/ADMET` | ADMET 전용 개발 브랜치 |
