# TDC-Studio DTI Phase C: TODOLIST & Future Backlog

> **최종 갱신 일자**: 2026-09-27  
> **상태**: Phase C 고도화 완료 및 후속 서빙 연동 과제 이관  
> **대상 모듈**: DTA/DTI 트랙 (`tdc_studio/models/dti`, `tdc_studio/data`, `tdc_studio/serving`, `deploy`)

---

## 📌 오늘(2026-09-27) 완료된 고도화 작업 요약

Google Colab 격리 환경(`munhyeongdo4@gmail.com`, 세션 `dti-gpu`, Tesla T4 14.6GB VRAM)에서 전체 파이프라인 검증 및 학습을 완수함.

| 과제 ID | 작업명 | 상태 | 주요 성과 및 산출물 |
| :--- | :--- | :---: | :--- |
| **Task F-1** | Cross-Attention Full Token-Level Contact Map (XAI) | **완료** | • `ChemBERTaEncoder` & `ESM2Encoder`에 `return_sequence=True` 지원<br>• 패딩 토큰 Zero-out 및 Masked Multi-Head Cross-Attention 구현<br>• `models/dti/phase_c_adv/contact_map_sample.json` 추출 검증 |
| **Task F-2** | Phase C [옵션 B] 정규화 강화 Staged Fine-Tuning | **완료** | • ChemBERTa 상위 2개 RoBERTa 레이어 Unfreezing ($LR=1.0\times 10^{-6}, WD=0.05$)<br>• Colab T4 14에포크 전체 학습 수행 (Cold-Drug CI `0.7443`, MSE `0.6404`)<br>• **실증 분석**: 미지의 골격(Cold-Drug) 평가에서는 7,700만 범용 화학 표현 공간을 보존한 Frozen Option A(CI `0.7659`, MSE `0.5780`)가 미세조정 대비 일반화 능력이 우수함을 입증 |
| **Task F-3** | Phase C [옵션 C] Multi-Affinity Multi-Task 확장 | **완료** | • BindingDB $K_d + K_i + \text{IC}_{50}$ 통합 `MultiAffinityDTADataModule` 구현<br>• 부분 결측 레이블 지원 `MaskedMSELoss` 구현<br>• `GraphDTAModel` 3-head 분기 및 `DTIMultiAffinityInferenceResponse` 스키마 확장 |

---

## 🚨 [긴급 점검 및 조치] 금일 발견된 핵심 결함 및 패치 계획 (Critical Bugs & Fixes)

오늘자(2026-09-27) 코드 리뷰에서 발견된 최우선 결함 사항으로, 신규 기능 작업 전 우선 패치 필수:

1. **[DTA - 치명적] 학습 스크립트 레이블 키 불일치 (`train_dti_phase_c.py` L226, 340)**
   - **문제**: `DTADataModule`은 배치 데이터에 단수형 `batch["label"]`을 적재하지만, 학습 스크립트는 `dev_batch["labels"]`를 조회하여 첫 배치에서 즉각적인 `KeyError` 크래시 발생.
   - **조치**: `targets = dev_batch.get("labels", dev_batch.get("label")).float()` 형태로 단수/복수형 호환 안전 폴백 적용.

2. **[ADMET - 브랜치 분기 이슈] `main`과 `brightonmoon/ADMET` 브랜치 간 파일 불일치**
   - **문제**: `pbpk/engine.py`, `pbpk_pipeline.py` 등 핵심 PBPK 코드가 아직 `brightonmoon/ADMET`에만 존재하여, `main` 브랜치 단독 빌드/서빙 시 `ImportError` 유발 위험.
   - **조치**: `brightonmoon/ADMET` 브랜치의 최신 커밋을 `main`으로 안전하게 Rebase/Merge 완료.

3. **[ADMET - 테스트 깨짐] `test_tri_hybrid_serving.py` (L11) import 경로 오류**
   - **문제**: `from api import app`으로 잘못 임포트되어 있어 테스트 스위트 실행 시 즉시 `ModuleNotFoundError` 발생.
   - **조치**: `from tdc_studio.serving.app import app`으로 수정하여 회귀 테스트 정상화.

4. **[DTA - 캐시 메모리 및 동결 해제 버그] (`pretrained_encoders.py` L103, 274)**
   - **문제**: `unfreeze(last_n_layers=N)` 호출 시 전체 플래그가 `self.freeze_backbone = False`로 변경되어, 여전히 동결 상태인 하위 레이어들까지 ESM 캐시 경로를 우회하고 매 에포크마다 30배 느린 재연산 수행.
   - **조치**: 부분 동결 해제 상태(`self.partially_unfrozen = True`)를 별도 추적하고, 동결된 하위 블록의 사전 연산 캐싱은 지속 활용하도록 분기 로직 정교화.

---

## 📋 내일(2026-09-28) TODOLIST: 제안 후속 과제

### 1. [Task T-1] 프로덕션 서빙 API에 XAI 2D Contact Map 엔드포인트 연동
- **우선순위**: P1 (사용자 편의성 및 신약 후보물질 결합 부위 해석력 극대화)
- **목표**: `/predict/dti` 엔드포인트에서 추론과 동시에 원자 $\times$ 아미노산 잔기 어텐션 맵을 옵션으로 반환.
- **주요 작업 내용**:
  1. `DTIInferenceRequest`에 `return_contact_map: bool = False` 옵션 추가.
  2. 추론 시 `return_sequence=True, return_attention=True`로 실행하여 상위 K개 고강도 결합 잔기(Top-K Contact Residues, 예: Glu312, Tyr104) 및 원자 인덱스를 파싱하여 응답에 포함.
  3. PyMOL 및 3D/2D 분자 시각화 도구 연동 스키마 표준화.
- **대상 파일**:
  - `tdc_studio/serving/schema.py`
  - `tdc_studio/serving/app.py`
  - `tests/test_serving_api.py`

---

### 2. [Task T-2] Multi-Affinity Multi-Task 서빙 파이프라인 정식 연동
- **우선순위**: P2 (다각적 친화도 분석 및 앙상블 신뢰도 제공)
- **목표**: 동일 약물-타깃 쌍에 대해 $K_d, K_i, \text{IC}_{50}$ 3종 동시 예측 서빙 엔드포인트 구축.
- **주요 작업 내용**:
  1. `/predict/dti/multi-affinity` 엔드포인트 신설.
  2. 3개 결합 지표 간 물리화학적 상관관계 일관성을 검증하는 신뢰도 지표(Affinity Consistency Score) 산출.
  3. 클라이언트용 통합 테스트 케이스 추가.
- **대상 파일**:
  - `tdc_studio/serving/app.py`
  - `tdc_studio/serving/schema.py`
  - `tests/test_tri_hybrid_serving.py`

---

### 3. [Task T-3] Colab 세션 유휴 자원 정리 및 자동 회수 가이드
- **우선순위**: P3 (컴퓨팅 자원 및 크레딧 관리)
- **목표**: 작업 종료 후 불필요한 Colab GPU 인스턴스 점유를 방지하고 자동 회수 체계 정착.
- **주요 작업 내용**:
  1. `scripts/colab_cleanup.ps1` 스크립트를 작성하여 사용 완료된 세션(`dti-gpu` 등) 일괄 `colab stop` 및 `unassign` 처리.
  2. 로컬 계정 전환 및 세션 현황 대시보드 스크립트 정비.
- **대상 파일**:
  - `scripts/colab_cleanup.ps1`
  - `scripts/colab_switch.ps1`
