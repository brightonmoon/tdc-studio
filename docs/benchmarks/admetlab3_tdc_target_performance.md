# TDC ADMET Benchmark Target Performance Specifications & Data Loader Reference
> **Benchmark Reference:** ADMETlab 3.0 (*Nucleic Acids Research*, 2024, [doi:10.1093/nar/gkae236](https://academic.oup.com/nar/article/52/W1/W422/7640525), Supplementary Tables 4 & 5) and Therapeutics Data Commons (TDC) Benchmark Leaderboards (Huang et al., 2021)

---

## 1. 개요 및 벤치마크 기준

ADMETlab 3.0은 최신 분자 딥러닝 아키텍처(D-MPNN, D-MPNN-Des, MGA)를 통해 약물 개발에 필수적인 ADMET 전 영역의 벤치마크 지표를 체계적으로 수립했습니다.
TDC-Studio는 본 논문(NAR 2024) 및 공식 부록(`gkae236_supplemental_file.html`)에 명시된 **DMPNN-Des 공식 테스트 세트 실측치(Mean ± Std)**와 TDC Leaderboard 실측치를 결합하여, **TDC 전 데이터셋에 대한 공식 목표 성능치(Target SOTA Threshold)** 및 **TDC Python 데이터 로더 규격**을 명시합니다.

- **권장 스플릿 (Dataset Split):** 신약 개발의 일반화 검증을 위해 **Scaffold Split (Bemis-Murcko Scaffold)**을 최우선 기준으로 하며, 보조적으로 `Random Split`을 지원합니다.

---

## 2. 클러스터별 TDC 데이터셋 목표 성능 및 로더 규격

### Cluster 1: Bio-Permeability & Oral Absorption (생체 투과 및 경구 흡수)
*수동 확산(Fick's law), 막 분배(LogD), 수용성 용해도(LogS) 및 P-gp 유출 펌프의 상호작용*

| TDC 데이터셋 명칭 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 $R^2$ (Test) | 목표 RMSE | 목표 MAE | 목표 ROC-AUC | 목표 ACC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`caco2_wang`** | Caco-2 Permeability | 회귀 | 906 | **0.743 ± 0.018** | **0.325 ± 0.013** | **0.242 ± 0.011** | - | - |
| **`lipophilicity_astrazeneca`**| logD7.4 | 회귀 | 4,200 | **0.902 ± 0.004** | **0.398 ± 0.008** | **0.290 ± 0.006** | - | - |
| **`solubility_aqsoldb`** | logS | 회귀 | 9,982 | **0.877 ± 0.013** | **0.746 ± 0.027** | **0.495 ± 0.010** | - | - |
| **`hia_hou`** | HIA | 분류 | 578 | - | - | - | **0.897 ± 0.054** | **0.911 ± 0.018** |
| **`bioavailability_ma`** | F20% / F30% | 분류 | 640 | - | - | - | **0.909 ± 0.047** | **0.857 ± 0.030** |
| **`pgp_broccatelli`** | Pgp-inhibitor | 분류 | 1,218 | - | - | - | **0.915 ± 0.011** | **0.850 ± 0.010** |

```python
# [Cluster 1 TDC Data Loaders]
from tdc.single_pred import ADME

# 1. Caco-2 Permeability
caco2_data = ADME(name = 'Caco2_Wang')
caco2_split = caco2_data.get_split(method = 'scaffold', seed = 42)

# 2. Lipophilicity (logD 7.4)
lipo_data = ADME(name = 'Lipophilicity_AstraZeneca')
lipo_split = lipo_data.get_split(method = 'scaffold', seed = 42)

# 3. Solubility (logS)
sol_data = ADME(name = 'Solubility_AqSolDB')
sol_split = sol_data.get_split(method = 'scaffold', seed = 42)

# 4. Human Intestinal Absorption & Bioavailability
hia_data = ADME(name = 'HIA_Hou')
hia_split = hia_data.get_split(method = 'scaffold', seed = 42)

bioav_data = ADME(name = 'Bioavailability_Ma')
bioav_split = bioav_data.get_split(method = 'scaffold', seed = 42)

# 5. P-glycoprotein Efflux
pgp_data = ADME(name = 'Pgp_Broccatelli')
pgp_split = pgp_data.get_split(method = 'scaffold', seed = 42)
```

---

### Cluster 2: Plasma Distribution & Tissue Penetration (혈장 분포 및 조직 투과)
*Free Drug Hypothesis: 알부민 결합률(PPBR)이 비결합 분율(fu)을 결정하고, 뇌장벽(BBB) 및 전신 조직(VDss) 침투를 지배*

| TDC 데이터셋 명칭 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 $R^2$ (Test) | 목표 RMSE | 목표 MAE | 목표 ROC-AUC | 목표 ACC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ppbr_az`** | PPB (결합률 %) | 회귀 | 1,797 | **0.824 ± 0.031** | **11.382 ± 0.648** | **5.976 ± 0.429%** | - | - |
| **`vdss_lombardo`** | VDss (분포용적) | 회귀 | 1,130 | **0.760 ± 0.045** | **0.301 ± 0.024** | **0.162 ± 0.010** | - | - |
| **`bbb_martins`** | BBB Penetration | 분류 | 2,050 | - | - | - | **0.908 ± 0.004** | **0.836 ± 0.014** |

```python
# [Cluster 2 TDC Data Loaders]
from tdc.single_pred import ADME

# 1. Plasma Protein Binding Rate (PPBR)
ppbr_data = ADME(name = 'PPBR_AZ')
ppbr_split = ppbr_data.get_split(method = 'scaffold', seed = 42)

# 2. Volume of Distribution at steady state (VDss)
vdss_data = ADME(name = 'VDss_Lombardo')
vdss_split = vdss_data.get_split(method = 'scaffold', seed = 42)

# 3. Blood-Brain Barrier Penetration (BBB)
bbb_data = ADME(name = 'BBB_Martins')
bbb_split = bbb_data.get_split(method = 'scaffold', seed = 42)
```

---

### Cluster 3: Cytochrome P450 Metabolism (약물 대사 효소 패널)
*5대 저해 효소 앵커(각 12,000+ 화합물) 및 3대 소규모 기질(Carbon-Mangels) turnover 패널*

| TDC 데이터셋 명칭 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 ROC-AUC (Test) | 목표 ACC (Test) | 목표 MCC (Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`cyp1a2_veith`** | CYP1A2 inhibitor | 분류 | 12,574 | **0.942 ± 0.003** | **0.874 ± 0.003** | **0.748 ± 0.006** |
| **`cyp2c19_veith`**| CYP2C19 inhibitor | 분류 | 12,092 | **0.915 ± 0.005** | **0.843 ± 0.009** | **0.689 ± 0.016** |
| **`cyp2c9_veith`** | CYP2C9 inhibitor | 분류 | 12,092 | **0.917 ± 0.009** | **0.854 ± 0.009** | **0.682 ± 0.020** |
| **`cyp2d6_veith`** | CYP2D6 inhibitor | 분류 | 13,130 | **0.886 ± 0.010** | **0.869 ± 0.012** | **0.575 ± 0.033** |
| **`cyp3a4_veith`** | CYP3A4 inhibitor | 분류 | 12,328 | **0.916 ± 0.005** | **0.834 ± 0.009** | **0.667 ± 0.020** |
| **`cyp2c9_substrate_carbonmangels`**| CYP2C9 substrate | 분류 | 666 | **0.782 ± 0.042** | **0.720 ± 0.049** | **0.433 ± 0.100** |
| **`cyp2d6_substrate_carbonmangels`**| CYP2D6 substrate | 분류 | 664 | **0.844 ± 0.057** | **0.754 ± 0.051** | **0.514 ± 0.099** |
| **`cyp3a4_substrate_carbonmangels`**| CYP3A4 substrate | 분류 | 667 | **0.798 ± 0.034** | **0.720 ± 0.012** | **0.444 ± 0.027** |

```python
# [Cluster 3 TDC Data Loaders]
from tdc.single_pred import ADME

# 1. 5 High-Volume CYP Inhibition Assays (Veith et al.)
cyp3a4_inh = ADME(name = 'CYP3A4_Veith').get_split(method = 'scaffold', seed = 42)
cyp2d6_inh = ADME(name = 'CYP2D6_Veith').get_split(method = 'scaffold', seed = 42)
cyp2c9_inh = ADME(name = 'CYP2C9_Veith').get_split(method = 'scaffold', seed = 42)
cyp2c19_inh = ADME(name = 'CYP2C19_Veith').get_split(method = 'scaffold', seed = 42)
cyp1a2_inh = ADME(name = 'CYP1A2_Veith').get_split(method = 'scaffold', seed = 42)

# 2. 3 Low-Volume CYP Substrate Assays (Carbon-Mangels et al.)
cyp2c9_sub = ADME(name = 'CYP2C9_Substrate_CarbonMangels')
split_cyp2c9 = cyp2c9_sub.get_split(method = 'scaffold', seed = 42)

cyp2d6_sub = ADME(name = 'CYP2D6_Substrate_CarbonMangels')
split_cyp2d6 = cyp2d6_sub.get_split(method = 'scaffold', seed = 42)

cyp3a4_sub = ADME(name = 'CYP3A4_Substrate_CarbonMangels')
split_cyp3a4 = cyp3a4_sub.get_split(method = 'scaffold', seed = 42)
```

---

### Cluster 4: Pharmacokinetic Clearance & Elimination (약물 제거 및 소실 반감기)
*간세포/마이크로솜 클리어런스($CL_{\text{int}}$) 및 체내 소실 반감기($t_{1/2}$)*

| TDC 데이터셋 명칭 | 벤치마크 기준 과제 | 유형 | 샘플 수 | 1차 목표 지표 (TDC SOTA) | 2차 지표 (ADMETlab 3.0) |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`clearance_hepatocyte_az`** | Clearance (Hepatocyte AZ) | 회귀 | 1,213 | **Spearman $\rho \ge 0.403 \pm 0.025$**, MAE $\le 38.4$ | $R^2 \ge 0.667$, RMSE $\le 2.912$ |
| **`clearance_microsome_az`** | Clearance (Microsome AZ) | 회귀 | 1,102 | **Spearman $\rho \ge 0.575 \pm 0.019$**, MAE $\le 26.2$ | $R^2 \ge 0.667$, RMSE $\le 2.912$ |
| **`half_life_obach`** | T1/2 (Elimination half-life) | 회귀 | 667 | **Spearman $\rho \ge 0.538 \pm 0.031$**, MAE $\le 12.2$ | $R^2 = 0.653 \pm 0.070$, MAE $= 0.420$ |

```python
# [Cluster 4 TDC Data Loaders]
from tdc.single_pred import ADME

# 1. Intrinsic Clearance in Human Hepatocytes
clearance_hep = ADME(name = 'Clearance_Hepatocyte_AZ')
split_hep = clearance_hep.get_split(method = 'scaffold', seed = 42)

# 2. Intrinsic Clearance in Human Liver Microsomes
clearance_mic = ADME(name = 'Clearance_Microsome_AZ')
split_mic = clearance_mic.get_split(method = 'scaffold', seed = 42)

# 3. Elimination Half-Life (Obach et al.)
half_life = ADME(name = 'Half_Life_Obach')
split_hl = half_life.get_split(method = 'scaffold', seed = 42)
```

---

### Cluster 5: Cardiac Safety & Broad Toxicity (심장 안전성 및 독성 프로파일링)
*hERG 칼륨 채널 차단, 급성 치사량(LD50), Ames 돌연변이성, 간독성(DILI) 및 Tox21 다중 어세이*

| TDC 데이터셋 명칭 | 대응 과제 | 유형 | 샘플 수 (TDC) | 목표 지표 1 (ROC-AUC / MAE) | 목표 지표 2 (ACC / $R^2$) | 목표 MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`herg`** | hERG Blocker (Wang et al.) | 분류 | 648 | **ROC-AUC $\ge 0.887 \pm 0.013$** | **ACC $\ge 0.825$** | **MCC $\ge 0.610$** |
| **`hERG_Karim`** | hERG Blockers (대규모 문헌 셋) | 분류 | 13,845 | **ROC-AUC $\ge 0.937 \pm 0.006$** | **ACC $\ge 0.828 \pm 0.017$** | **MCC $\ge 0.680 \pm 0.026$** |
| **`herg_central` (`hERG_inhib`)** | hERG 10µM Blocker ($IC_{50} < 10\mu\text{M}$) | 분류 | **306,893** *(ADMETlab 서브셋: 9,876)* | **ROC-AUC $\ge 0.885 \pm 0.015$** *(ADMETlab: 0.840)* | **ACC $\ge 0.812$** *(ADMETlab: 0.691)* | **MCC $\ge 0.540$** *(ADMETlab: 0.426)* |
| **`herg_central` (`hERG_at_10uM`)** | 10µM 농도 저해율 (%) | 회귀 | **306,893** | **MAE $\le 15.20 \pm 0.85$ %** | **$R^2 \ge 0.650 \pm 0.030$** | RMSE $\le 21.40$ % |
| **`herg_central` (`hERG_at_1uM`)** | 1µM 농도 저해율 (%) | 회귀 | **306,893** | **MAE $\le 12.80 \pm 0.62$ %** | **$R^2 \ge 0.682 \pm 0.025$** | RMSE $\le 18.10$ % |
| **`ld50_zhu`** | Acute Oral Toxicity LD50 | 회귀 | 7,385 | **MAE $\le 0.584 \pm 0.012$** | **$R^2 \ge 0.624 \pm 0.021$** | RMSE $\le 0.812$ |
| **`dili`** | DILI (Drug-Induced Liver Injury) | 분류 | 475 | **ROC-AUC $\ge 0.860 \pm 0.052$** | **ACC $\ge 0.787 \pm 0.075$** | **MCC $\ge 0.590 \pm 0.143$** |
| **`ames`** | AMES Mutagenicity | 분류 | 7,255 | **ROC-AUC $\ge 0.882 \pm 0.007$** | **ACC $\ge 0.785 \pm 0.015$** | **MCC $\ge 0.576 \pm 0.028$** |
| **`skin_reaction`** | Skin Sensitization | 분류 | 404 | **ROC-AUC $\ge 0.787 \pm 0.043$** | **ACC $\ge 0.780 \pm 0.062$** | **MCC $\ge 0.491 \pm 0.152$** |
| **`carcinogens_lagunin`**| Carcinogenicity | 분류 | 280 | **ROC-AUC $\ge 0.715 \pm 0.029$** | **ACC $\ge 0.668 \pm 0.040$** | **MCC $\ge 0.349 \pm 0.078$** |
| **`clintox`** | FDA Clinical Trial Failure | 분류 | 1,484 | **ROC-AUC $\ge 0.865 \pm 0.011$** | **ACC $\ge 0.761 \pm 0.022$** | **MCC $\ge 0.534 \pm 0.036$** |
| **`tox21` (NR-AR)** | Nuclear Receptor - AR | 분류 | 7,831 | **ROC-AUC $\ge 0.883 \pm 0.016$** | **ACC $\ge 0.956 \pm 0.016$** | **MCC $\ge 0.534 \pm 0.076$** |
| **`tox21` (NR-AhR)** | Nuclear Receptor - AhR | 분류 | 7,831 | **ROC-AUC $\ge 0.924 \pm 0.011$** | **ACC $\ge 0.862 \pm 0.040$** | **MCC $\ge 0.550 \pm 0.047$** |
| **`tox21` (SR-MMP)** | Stress Response - MMP | 분류 | 7,831 | **ROC-AUC $\ge 0.941 \pm 0.016$** | **ACC $\ge 0.857 \pm 0.048$** | **MCC $\ge 0.604 \pm 0.062$** |
| **`tox21` (SR-p53)** | Stress Response - p53 | 분류 | 7,831 | **ROC-AUC $\ge 0.890 \pm 0.027$** | **ACC $\ge 0.882 \pm 0.034$** | **MCC $\ge 0.415 \pm 0.036$** |

> [!NOTE]
> **hERG 데이터셋 3종 크기 차이 및 ADMETlab 3.0 (9,876개)의 출처 분석**
> - **`hERG` (Wang et al., 648개)**: 문헌에서 정밀 검증된 전통적인 소규모 벤치마크.
> - **`hERG_Karim` (13,845개)**: ChEMBL 및 문헌 데이터를 통합한 대규모 데이터셋으로, ADMETlab 3.0 논문의 `hERG Blocker (13,845개, ROC-AUC 0.937)`와 정확히 일치함.
> - **`herg_central` (TDC 공식, 306,893개)**: 미국 NIH NCATS에서 IonWorks 고속 자동 패치클램프로 측정한 **30.7만 개 HTS 전체 라이브러리** (Du et al., *Nat. Biotech.* 2011).
> - **ADMETlab 3.0의 `hERG Blocker 10um` (9,876개)**: ADMETlab 연구진은 30만 개 노이즈 원시 HTS 데이터를 전부 쓰지 않고, 10µM 기준치 검증 및 화학적 정제를 거친 **9,876개 서브셋(양성 5,090 / 음성 4,786)**을 벤치마크로 사용함. TDC에서 제공되는 306,893개 원본 데이터를 활용할 경우 더 방대한 화학 공간을 탐색할 수 있음.

```python
# [Cluster 5 TDC Data Loaders]
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

# 1. Standard hERG (Wang et al., 648 drugs)
herg_data = Tox(name = 'hERG')
split_herg = herg_data.get_split(method = 'scaffold', seed = 42)

# 2. Large-scale hERG (Karim et al., 13,845 drugs)
herg_karim_data = Tox(name = 'hERG_Karim')
split_karim = herg_karim_data.get_split(method = 'scaffold', seed = 42)

# 3. Massive hERG Central (306,893 drugs, 3 available target labels)
# Available labels: ['hERG_at_1uM', 'hERG_at_10uM', 'hERG_inhib']
herg_central_labels = retrieve_label_name_list('herg_central')

# 3-1. Binary Classification: hERG_inhib (whether IC50 < 10uM, 1 = blocker, 0 = non-blocker)
herg_central_inhib = Tox(name = 'herg_central', label_name = 'hERG_inhib')
split_central_inhib = herg_central_inhib.get_split(method = 'scaffold', seed = 42)

# 3-2. Continuous Regression: hERG_at_10uM (% inhibition at 10 uM)
herg_central_10um = Tox(name = 'herg_central', label_name = 'hERG_at_10uM')
split_central_10um = herg_central_10um.get_split(method = 'scaffold', seed = 42)

# 3-3. Continuous Regression: hERG_at_1uM (% inhibition at 1 uM)
herg_central_1um = Tox(name = 'herg_central', label_name = 'hERG_at_1uM')
split_central_1um = herg_central_1um.get_split(method = 'scaffold', seed = 42)

# 4. Acute Toxicity LD50 (Zhu et al.)
ld50_data = Tox(name = 'LD50_Zhu')
split_ld50 = ld50_data.get_split(method = 'scaffold', seed = 42)

# 5. Organ Toxicity: DILI, Ames, Skin Reaction, Carcinogens
dili_data = Tox(name = 'DILI').get_split(method = 'scaffold', seed = 42)
ames_data = Tox(name = 'AMES').get_split(method = 'scaffold', seed = 42)
skin_data = Tox(name = 'Skin_Reaction').get_split(method = 'scaffold', seed = 42)
carc_data = Tox(name = 'Carcinogens_Lagunin').get_split(method = 'scaffold', seed = 42)
clintox_data = Tox(name = 'ClinTox').get_split(method = 'scaffold', seed = 42)

# 6. Multi-Assay Tox21 (retrieve_label_name_list required)
tox21_labels = retrieve_label_name_list('Tox21')
tox21_data = Tox(name = 'Tox21', label_name = tox21_labels[0])
split_tox21 = tox21_data.get_split(method = 'scaffold', seed = 42)
```
