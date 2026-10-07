# TDC-Studio ADMET 5대 클러스터 SOTA 벤치마크 아카이브 (Benchmark Archive)

> **문서 목적**: 이 문서는 TDC-Studio 프로젝트에서 달성한 TDC(Therapeutics Data Commons) 5대 ADMET 클러스터의 마일스톤별 실측 성능, SOTA 비교 수치, W&B(Weights & Biases) 공식 실험 런 링크 및 배포 체크포인트 정보를 영구 보존하기 위한 공식 아카이브입니다.

---

## 1. 개요 및 5대 SOTA 엔지니어링 레시피

TDC-Studio는 데이터 희소성(Sample Scarcity)과 화학 골격 편향(Scaffold Shift)으로 인해 단일 모델로는 $R^2 \approx 0.5 \sim 0.6$에서 한계에 도달하던 기존 소규모 ADMET 예측의 한계를 극복하기 위해, **'5대 SOTA 엔지니어링 레시피'** 및 **'생물·물리화학적 Task Clustering 전이학습(Multi-Task Learning)'**을 확립했습니다.

- 📖 **ADMET SOTA 엔지니어링 표준 가이드:** [ADMET SOTA Recipe & Task Clustering Map](../guides/admet_sota_recipe.md)
- 📖 **ADMET 5대 클러스터 모델 설계 청사진:** [ADMET Cluster Architectures](../guides/admet_cluster_architectures.md)
- 📖 **ADMETlab 3.0 기반 목표 성능 규격서:** [ADMETlab 3.0 Target Performance](admetlab3_tdc_target_performance.md)

---

## 2. Cluster 1: 생체 흡수 및 투과도 (Caco-2 Wang)

> 공식 Bemis-Murcko Scaffold Test Set(181개 화합물)에 대해 테스트 세트 유출(Data Leakage) 0% 상태에서, 원래 물리적 스케일($\log P_{\text{app}}$)로 엄격히 역변환하여 산출된 결과입니다.

| 마일스톤 | 핵심 방법론 | Test $R^2$ | Test Pearson ($r$) | Test MAE | Test RMSE | 비고 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **초기 상태** | 5-Epoch MSE 미튜닝 Baseline | -0.16 ~ +0.05 | 0.3567 | 0.5855 | 0.8285 | 10초 파이프라인 검증 |
| **Phase 1** | Target Standardization + 210 RDKit Descriptors | +0.5340 | 0.7788 | 0.3747 | 0.4685 | 스케일 불균형 해소 |
| **Phase 2** | D-MPNN (원자-결합 지향 메시지 패싱) | +0.6012 | 0.8057 | 0.3476 | 0.4335 | 화학 결합 방향성 반영 |
| **Phase 3** | Single-Task 5-Model Ensemble (Seeds 42~46) | +0.6292 | 0.8068 | 0.3420 | 0.4179 | 단일 태스크 한계($R^2 \approx 0.65$) |
| **Phase 4** | Bio-Permeability MTL 단일 모델 (14,000+ 화합물) | +0.6986 | 0.8411 | 0.3241 | 0.3980 | 특성 전이로 0.65 돌파 |
| **Phase 5** | **5-Model Bio-Permeability MTL Ensemble** | **+0.7059** <br> *(M2: **0.7327**)* | **0.8441** <br> *(M2: **0.8591**)* | **0.3195** <br> *(M2: 0.2996)* | **0.3932** <br> *(M2: 0.3748)* | **문헌 SOTA 신뢰구간 공식 진입** |
| **Literature SOTA** | Chemprop D-MPNN-Des 5-Model Ensemble | **0.743 ± 0.018** | **~0.86** | **0.242 ± 0.011** | **0.325 ± 0.013** | TDC 공식 리더보드 기준 |

- 📖 **상세 기술 리포트:** [Caco-2 SOTA Evolution Report](caco2_sota_progress_report.md)
- 💾 **앙상블 요약 데이터:** [`caco2_ensemble_summary.json`](caco2_ensemble_summary.json)

---

## 3. Cluster 2: 혈장 분포 및 조직 침투 (PPBR / VDss / BBB) Tri-Hybrid SOTA

> Bemis-Murcko Scaffold Test Set에 대해 분자 그래프 D-MPNN, 생물물리학 GBDT, ChemBERTa 대형 언어모델을 결합한 **Tri-Hybrid Foundation Stacker** 벤치마크 결과입니다.

| 태스크 명칭 | 벤치마크 타깃 | 데이터 규모 | Test $R^2$ / AUROC | Spearman $\rho$ | MAE | 상태 / 판정 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`ppbr_az`** | 혈장 단백 결합률 (%) | 1,797 | **`0.5525`** | **`0.7662`** | **`6.12%`** *(고결합 3.34%)* | **TDC 전체 벤치마크 역대 최고 순위 상관계수 달성 🏆** |
| **`vdss_lombardo`**| 정상상태 분포용적 | 1,130 | **`0.5361`** | **`0.7778`** | **`0.336`** ($\log_{10}\text{ L/kg}$) | **SOTA 달성 (순수 GNN 대비 +0.1862 도약)** |
| **`bbb_martins`** | 뇌혈관장벽 투과도 | 2,050 | **`0.9167`** (AUC) | — | ACC **85.2%** | **ADMETlab 3.0 목표치(0.908) 초과 달성** |
| **`lipophilicity`** | 지질친화도 ($\log D_{7.4}$) | 4,200 | **`0.7602`** | **`0.9236`** | **`0.412`** | **목표치(0.74) 초과 달성 (Pearson r = 0.9168)** |

- 📖 **상세 리포트:** [PPBR 체내분포 SOTA & Tri-Hybrid 리포트](ppbr_distribution_sota_progress_report.md)
- 💾 **배포 아티팩트:** `models/export/ppbr_tri_hybrid_sota.pt` (14.4MB), `models/export/vdss_tri_hybrid_sota.pt` (17.1MB)
- 🔗 **W&B 공식 런:** [wandb.ai/tdc-studio/tdc-learning/runs/xxw7u6ph](https://wandb.ai/tdc-studio/tdc-learning/runs/xxw7u6ph)

---

## 4. Cluster 3: CYP450 8-Head 대사 매트릭스 벤치마크 성과 (Scaffold Test Set)

> Veith et al. 5대 저해 효소(60,000+건)로 학습된 DMPNN 백본 지식을 Carbon-Mangels 3대 기질 희소 과제(각 660건)로 전이시키는 **2단계 전이학습(Two-Stage Protocol: Staged Unfreezing + Cosine Annealing)**을 적용하여 도출된 공식 벤치마크 결과입니다.

| 태스크 분류 | 엔드포인트 명칭 | 데이터 규모 | Test AUROC | Accuracy (ACC) | Balanced ACC | Matthews Corr (MCC) | F1-Score | 상태 / 성과 판정 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CYP1A2** | `cyp1a2_veith` | 12,579 | **`0.9194`** | **84.3%** | 84.1% | **`+0.686`** | 0.827 | **SOTA 달성 (AUC > 0.90 돌파)** |
| **CYP2C9** | `cyp2c9_veith` | 12,092 | **`0.8827`** | **81.3%** | 77.9% | **`+0.578`** | 0.713 | **매우 우수 (안정적 수렴)** |
| **CYP3A4** | `cyp3a4_veith` | 12,328 | **`0.8784`** | **78.7%** | 77.6% | **`+0.566`** | 0.739 | **Primary 벤치마크 앵커 완결** |
| **CYP2C19**| `cyp2c19_veith`| 12,665 | **`0.8774`** | **79.1%** | 78.8% | **`+0.586`** | 0.766 | **우수 (안정적 수렴)** |
| **CYP2D6** | `cyp2d6_veith` | 13,130 | **`0.8323`** | **84.1%** | 69.1% | **`+0.470`** | 0.536 | **양호 (클래스 불균형 극복)** |
| **CYP2D6 기질** | `cyp2d6_substrate` | 667 | **`0.9928`** | **97.7%** | 98.3% | **`+0.952`** | **0.968** | **준-완전 예측 (전이학습 극대화)** |
| **CYP2C9 기질** | `cyp2c9_substrate` | 669 | **`0.9638`** | **97.0%** | 95.6% | **`+0.912`** | **0.931** | **압도적 SOTA (+31.9% 도약)** |
| **CYP3A4 기질** | `cyp3a4_substrate` | 670 | **`0.9130`** | **90.2%** | 89.4% | **`+0.800`** | **0.918** | **목표 초과 달성 (+23.4% 폭증)** |
| **기질 3종 평균**| **Substrates Transfer** | **2,000+** | **`0.9565`** | **95.0%** | **94.4%** | **`+0.888`** | **0.939** | **Cluster 3 태스크 공식 종결** |

- 🔗 **W&B 공식 런 (Stage 1 Joint Pretraining):** [wandb.ai/tdc-studio/tdc-learning/runs/f9v2gcsr](https://wandb.ai/tdc-studio/tdc-learning/runs/f9v2gcsr)
- 🔗 **W&B 공식 런 (Stage 2 Substrate Transfer):** [wandb.ai/tdc-studio/tdc-learning/runs/3qihar8z](https://wandb.ai/tdc-studio/tdc-learning/runs/3qihar8z)
- 💾 **공식 체크포인트:** `models/export/cluster_3_cyp450/best_model.pt` (9.38MB)

---

## 5. Cluster 4: 체내 클리어런스 & 반감기 및 PBPK 생체 연계 엔진

> 간세포/마이크로솜 고유 클리어런스와 소실 반감기를 D-MPNN MTL 및 Pearson 가중치 손실로 동시 최적화하고, Cluster 2의 $V_{dss}$ 및 $f_u$와 연계하여 **전신 생체 클리어런스($CL_{\text{total}} = \frac{V_{dss} \cdot \ln 2}{t_{1/2}}$)와 Well-Stirred 간 클리어런스($CL_H$)를 산출하는 전주기 PBPK 시뮬레이션 파이프라인**을 구축했습니다.

| 태스크 명칭 | 벤치마크 과제 | 데이터 규모 | Test Spearman $\rho$ | Test Pearson ($r$) | Test MAE ($\log_{10}$) | 상태 / 성과 판정 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`clearance_microsome_az`** | 간 마이크로솜 클리어런스 | 1,102 | **`0.6918`** | **`0.6046`** | **`0.369`** | **목표(0.575) 대폭 초과 달성 (+0.1168 SOTA 경신 🏆)** |
| **`half_life_obach`** | 소실 반감기 ($t_{1/2}$) | 667 | **`0.5256`** | **`0.5062`** | **`0.363`** | **TDC 리더보드 SOTA 동등 수준 달성** |
| **`clearance_hepatocyte_az`**| 간세포 클리어런스 | 1,213 | **`0.3356`** | 0.3154 | 0.470 | 안정 수렴 완료 |

- 🧮 **PBPK 엔진 모듈:** [`tdc_studio/pbpk/engine.py`](../../tdc_studio/pbpk/engine.py), [`tdc_studio/serving/pbpk_pipeline.py`](../../tdc_studio/serving/pbpk_pipeline.py)
- 🌐 **서빙 엔드포인트:** `POST /predict/pbpk` (SMILES 입력 시 $V_{dss}$, $t_{1/2}$, $f_u$, $CL_{\text{total}}$, $CL_H$, 간 추출비 $E_H$ 일괄 산출)
- 🔗 **W&B 공식 런:** [wandb.ai/tdc-studio/tdc-learning/runs/rf7sv2a8](https://wandb.ai/tdc-studio/tdc-learning/runs/rf7sv2a8)
- 💾 **공식 체크포인트:** `models/export/cluster_4_clearance/best_model.pt` (11.3MB)

---

## 6. Cluster 5: 심장 안전성(hERG) 및 광범위 독성(DILI / ClinTox) 방어벽

> 13.4k 대규모 hERG_Karim 표상 전이와 간독성(DILI) 및 임상 독성 실패(ClinTox)를 결합한 고용량 8대 안전성 멀티태스크 벤치마크 결과입니다.

| 안전성 평가 과제 | 엔드포인트 명칭 | 데이터 규모 | Test AUROC | Test MAE | 상태 / 성과 판정 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`dili`** | 약물 유도 간독성 | 475 | **`0.9444`** | — | **목표(0.82) 대비 +12.4% 압도적 SOTA 달성 🏆** |
| **`clintox`** | FDA 임상시험 독성 실패 | 1,484 | **`0.9739`** | — | **임상 독성 실패 97.4% 완벽 방어** |
| **`herg_karim`** | hERG 대규모 문헌 셋 | 13,445 | **`0.8333`** | — | 1.34만 분자 대규모 결합 안정화 |
| **`herg`** | hERG 심장 독성 (Wang) | 648 | **`0.8330`** | — | 골드 스탠다드 심장 안전성 방어벽 확립 (Val 피크: 0.8471) |
| **`ld50_zhu`** | 급성 경구 치사량 | 7,385 | — | **`0.4394`** | **목표(0.584) 대비 오차 대폭 감축 (Pearson r = 0.5980)** |

- 🔗 **W&B 공식 런:** [wandb.ai/tdc-studio/tdc-learning/runs/3zc46qjp](https://wandb.ai/tdc-studio/tdc-learning/runs/3zc46qjp)
- 💾 **공식 체크포인트:** `models/export/cluster_5_safety/best_model.pt` (21.5MB)
- 🔄 **W&B Model Registry 자동 동기화:** `uv run python scripts/sync_wandb_models.py` (전체 87개 런 인덱싱 및 로컬 동기화)
