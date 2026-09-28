# TDC-Studio DTI Phase C: TODOLIST & Future Backlog

> **최종 갱신 일자**: 2026-09-28  
> **상태**: Phase C 잔여 과제(Task T-1, T-2, T-3) 프로덕션 서빙 연동 및 자원 관리 체계 완료  
> **대상 모듈**: DTA/DTI 트랙 (`tdc_studio/models/dti`, `tdc_studio/data`, `tdc_studio/serving`, `deploy`, `scripts`)

---

## 📌 완료된 고도화 및 서빙 연동 작업 요약

### 1. Phase C 모델 고도화 (2026-09-27 완수)
Google Colab 격리 환경(`munhyeongdo4@gmail.com`, 세션 `dti-gpu`, Tesla T4 14.6GB VRAM)에서 전체 파이프라인 검증 및 학습을 완수함.

| 과제 ID | 작업명 | 상태 | 주요 성과 및 산출물 |
| :--- | :--- | :---: | :--- |
| **Task F-1** | Cross-Attention Full Token-Level Contact Map (XAI) | **완료** | • `ChemBERTaEncoder` & `ESM2Encoder`에 `return_sequence=True` 지원<br>• 패딩 토큰 Zero-out 및 Masked Multi-Head Cross-Attention 구현<br>• `models/dti/phase_c_adv/contact_map_sample.json` 추출 검증 |
| **Task F-2** | Phase C [옵션 B] 정규화 강화 Staged Fine-Tuning | **완료** | • ChemBERTa 상위 2개 RoBERTa 레이어 Unfreezing ($LR=1.0\times 10^{-6}, WD=0.05$)<br>• Colab T4 14에포크 전체 학습 수행 (Cold-Drug CI `0.7443`, MSE `0.6404`)<br>• **실증 분석**: 미지의 골격(Cold-Drug) 평가에서는 7,700만 범용 화학 표현 공간을 보존한 Frozen Option A(CI `0.7659`, MSE `0.5780`)가 미세조정 대비 일반화 능력이 우수함을 입증 |
| **Task F-3** | Phase C [옵션 C] Multi-Affinity Multi-Task 확장 | **완료** | • BindingDB $K_d + K_i + \text{IC}_{50}$ 통합 `MultiAffinityDTADataModule` 구현<br>• 부분 결측 레이블 지원 `MaskedMSELoss` 구현<br>• `GraphDTAModel` 3-head 분기 및 `DTIMultiAffinityInferenceResponse` 스키마 확장 |

---

### 2. 프로덕션 서빙 연동 및 자원 관리 (2026-09-28 완수)

| 과제 ID | 작업명 | 상태 | 주요 성과 및 구현 내역 |
| :--- | :--- | :---: | :--- |
| **Task T-1** | 프로덕션 서빙 API에 XAI 2D Contact Map 엔드포인트 연동 (P1) | **완료** | • `tdc_studio/serving/xai_utils.py` 신설 (Top-K 잔기/원자 어텐션 추출 및 PyMOL 명령어 자동 생성)<br>• `DTIInferenceRequest`에 `return_contact_map`, `top_k_residues`, `return_full_matrix` 연동<br>• `POST /predict/dti` 응답에 `top_contact_residues`, `top_contact_atoms`, `pymol_commands` 직렬화 반환 |
| **Task T-2** | Multi-Affinity Multi-Task 서빙 파이프라인 정식 연동 (P2) | **완료** | • `POST /predict/dti/multi-affinity` 및 별칭 `/predict/dti/multi` 신설<br>• $K_d, K_i, \text{IC}_{50}$ 3종 동시 예측 및 물리화학적 Cheng-Prusoff 정합성 검증 엔진(ACS) 구현<br>• 다중 태스크 정규화 역변환 파이프라인(`DTIMultiAffinityPipeline`) 구현 및 단위 테스트 통과 |
| **Task T-3** | Colab 세션 유휴 자원 정리 및 자동 회수 가이드 (P3) | **완료** | • `munhyeongdo4@gmail.com` 전용 `scripts/colab_cleanup.ps1` 작성 (`list`, `stop`, `prune`, `-All`, `-Force`)<br>• 격리 프로파일(`C:\Users\xps\.colab_munhyeongdo4`) 연동 및 Stale 세션/임시 번들 자동 소거 기능 구현 |

---

## 🧪 테스트 및 무결성 검증 결과 (2026-09-28 기준)

- `tests/test_serving_api.py`: **15 passed** (100% 통과, 19.41s)
- `tests/test_dti_phase_c_advancement.py`: **5 passed** (100% 통과, 16.30s)
- 전체 DTI 서빙 엔드포인트 및 XAI 2D Contact Map, PyMOL 선택 스크립트 정상 동작 확인 완료.
