# TDC 22+ ADMET 핵심 지표 선정 근거 및 PBPK 생체역학 설계 철학

> **문서 목적**: 이 문서는 Therapeutics Data Commons (TDC) 내 38종 벤치마크 데이터셋 중 왜 22종(현재 25종)의 핵심 지표를 선별하였는지, 비선정/통합 데이터셋의 의약화학적·임상 규제적 사유는 무엇인지, 그리고 이들 지표가 어떻게 14-구획 PBPK 생체역학 엔진으로 유기적으로 연결되는지를 설명하는 설계 철학 문서입니다.

---

## 1. 개요 및 설계 배경

TDC(Therapeutics Data Commons)의 `single_pred` 영역에는 **ADME 27종 + Tox 11종 = 총 38종의 벤치마크 데이터셋**이 존재합니다.  
TDC-Studio가 이 38종 전체를 단순 나열하지 않고 **'22종 핵심 지표 체계(현재 25종으로 확장)'**로 선별·압축하여 프로덕션 엔진을 구축한 이유는 **의약화학적 실무, FDA/ICH 임상 규제 가이드라인, 그리고 14-구획 PBPK 생체역학 수식의 완결성**을 충족하기 위함입니다.

---

## 2. 22종(C1~C5) 핵심 지표 구성

TDC-Studio는 전체 ADMET 과정을 생체 내 약물 이동 경로에 따라 5대 클러스터로 모듈화하여 관리합니다:

- **C1. 흡수 (Absorption, 6종)**:
  - `caco2_wang`: 인간 장 상피세포 겉보기 투과도 ($P_{\text{app}}$)
  - `lipophilicity_astrazeneca`: 옥탄올/물 분배계수 ($\log D_{7.4}$)
  - `solubility_aqsoldb`: 열역학적 수용해도 ($\log S$)
  - `hia_hou`: 인간 소장 흡수율 (Human Intestinal Absorption)
  - `bioavailability_ma`: 경구 생체이용률 (Oral Bioavailability)
  - `pgp_broccatelli`: P-당단백질(P-gp) 유출 펌프 기질/저해 여부

- **C2. 분포 (Distribution, 3종)**:
  - `ppbr_az`: 혈장 단백 결합률 (Plasma Protein Binding Rate, 비결합 분율 $f_u = 1 - \text{PPBR}$)
  - `vdss_lombardo`: 정상상태 분포용적 ($V_{dss}$)
  - `bbb_martins`: 뇌혈관장벽 투과도 (Blood-Brain Barrier Penetration)

- **C3. 대사 (Metabolism, 8종)**:
  - CYP450 5대 저해 효소: `cyp1a2_veith`, `cyp2c9_veith`, `cyp2c19_veith`, `cyp2d6_veith`, `cyp3a4_veith`
  - CYP450 3대 기질 효소: `cyp2c9_substrate`, `cyp2d6_substrate`, `cyp3a4_substrate`

- **C4. 배설 (Excretion, 3종)**:
  - `clearance_microsome_az`: 인간 간 마이크로솜 고유 청소율 ($CL_{\text{int,app}}$)
  - `clearance_hepatocyte_az`: 인간 간세포 청소율 ($CL_{\text{int}}$)
  - `half_life_obach`: 혈중 소실 반감기 ($t_{1/2}$)

- **C5. 독성 (Toxicity, 2+3종)**:
  - 핵심 규제 2종: `herg` (Wang et al. 심장 독성), `ames` (유전/돌연변이 독성)
  - 확장 안전성 3종: `dili` (약물 유도 간독성), `clintox` (FDA 임상시험 독성 실패), `ld50_zhu` (급성 경구 치사량)

> **"22종" 명칭의 유래**: 흡수(6) + 분포(3) + 대사(8) + 배설(3) + 독성(2: hERG, AMES) = 22종으로 아키텍처가 확정되었으며, 이후 안전성 강화로 현재 25종 풀 패널이 서빙됩니다.

---

## 3. TDC 38종 중 비선정/통합 데이터셋의 4대 사유

TDC 내 38종 중 단일 모델로 독립 서빙하지 않고 제외하거나 사전학습/전이학습으로 통합한 데이터셋의 구체적인 근거는 다음과 같습니다:

| 제외/통합 대상 데이터셋 | 선정된 대표 표준 지표 | 제외 및 통합 사유 (의약화학 및 규제 기준) |
| :--- | :--- | :--- |
| `herg_karim`, `herg_central` | **`herg` (Wang et al.)** | **동일 타깃 중복 해소 및 2-Stage 전이학습 통합**: 동일 hERG K+ 채널 차단 위험을 중복 노출하지 않고, `herg_karim`(13.4k) 사전학습 후 `herg`(Wang, 648)로 미세조정하는 2-Stage 모델로 단일화 (AUROC 0.8568 달성). |
| `b3db_classification`, `b3db_regression` | **`bbb_martins`** | **골드스탠다드 채택**: B3DB는 Martins 셋의 변형/확장셋으로, 문헌 인용도와 벤치마크 신뢰도가 가장 높은 `bbb_martins`를 대표 지표로 선정. |
| `rlm` (Rat Liver Microsome) | **`clearance_microsome_az` (인간)** | **인간(Human) vs 동물(Rat) 종간 차이(Species Divergence) 배제**: 쥐와 인간의 대사 효소 발현 차이로 인한 왜곡을 방지하고, Human PBPK 임상 외삽에 집중하기 위해 인간 유래 마이크로솜/간세포 데이터만 채택. |
| `pampa_ncats`, `approved_pampa_ncats` | **`caco2_wang`** | **생체 세포막 표준성**: PAMPA는 인공막 수동 확산만 측정하는 반면, Caco-2는 능동 수송체(Transporter)와 수동 확산이 모두 공존하는 FDA 공인 경구 흡수 모델임. |
| `tox21` (12 어세이), `toxcast` (617 어세이) | **`herg`, `ames`, `dili`, `clintox`** | **임상 규제 필수 독성 집중**: Tox21/ToxCast는 환경 스크리닝용 시험관내 반응 풀인 반면, 신약 승인(IND/NDA)에 직결되는 핵심 독성은 ICH 가이드라인의 심장독성(S7B: hERG), 유전독성(S2: AMES), 간독성(DILI)임. |
| `hydrationfreeenergy_freesolv` | **`solubility_aqsoldb`** | **실측 물리화학 지표 우선**: 640여 개 계산 자유에너지보다 9,982개 실제 수용해도 실측치를 사용하는 것이 생체이용률 예측에 직접적임. |

---

## 4. PBPK 생체역학 파이프라인 (Mechanistic Chain)

선정된 22+ 지표는 독립적인 점수 출력이 아니라, **14-구획 ODE 생체약동학 엔진의 필수 입력 파라미터로 상호 연결**됩니다:

```text
[C1 흡수] Lipophilicity + Caco-2 + HIA  ──▶ 경구 흡수율(ka) & 생체이용률(F)
[C2 분포] PPBR (fu) + VDss              ──▶ 혈장-조직 분배계수(Kp) & 정상상태 분포용적
[C3 대사] CYP450 8-Head Isoforms         ──▶ 주요 간 대사 경로 및 약물상호작용(DDI) 억제상수(Ki)
[C4 배설] Microsome + Hepatocyte Cl     ──▶ 고유 청소율(CLint) ──▶ 전신 청소율(CLtotal)
                                                                      │
                                                ┌─────────────────────┘
                                                ▼
                                    제거 반감기 t1/2 = (0.693 × Vdss) / CLtotal
                                    혈중 농도-시간 곡선 (Cplasma - t) 시뮬레이션 완결
```

### 14-구획 ODE 연계 수식

1. **비결합 분율 ($f_u$)**:
   $$f_u = 1 - \frac{\text{PPBR}}{100}$$

2. **Well-Stirred 모델 기반 간 청소율 ($CL_H$)**:
   $$CL_H = \frac{Q_H \cdot f_u \cdot CL_{\text{int}}}{Q_H + f_u \cdot CL_{\text{int}}}$$
   *(여기서 $Q_H$는 인간 간 혈류량 $\approx 20.7 \text{ mL/min/kg}$)*

3. **간 추출비 ($E_H$)**:
   $$E_H = \frac{CL_H}{Q_H}$$

4. **전신 청소율 ($CL_{\text{total}}$) 및 제거 반감기 ($t_{1/2}$)**:
   $$CL_{\text{total}} = CL_H + CL_R$$
   $$t_{1/2} = \frac{V_{dss} \cdot \ln 2}{CL_{\text{total}}}$$

- 🧮 **관련 모듈 소스코드:**
  - PBPK ODE 시뮬레이션 엔진: [`tdc_studio/pbpk/engine.py`](../../tdc_studio/pbpk/engine.py)
  - ADMET + PBPK 종단간 서빙 파이프라인: [`tdc_studio/serving/pbpk_pipeline.py`](../../tdc_studio/serving/pbpk_pipeline.py)
