# TDC ADMET Benchmark Target Performance Specifications
> **Benchmark Reference:** ADMETlab 3.0 (*Nucleic Acids Research*, 2024, [doi:10.1093/nar/gkae236](https://academic.oup.com/nar/article/52/W1/W422/7640525), Supplementary Tables 4 & 5)

---

## 1. 개요 및 벤치마크 기준

ADMETlab 3.0은 최신 분자 딥러닝 아키텍처(D-MPNN, D-MPNN-Des, MGA)를 통해 약물 개발에 필수적인 ADMET 전 영역의 벤치마크 지표를 체계적으로 수립했습니다.
TDC-Studio는 본 논문(NAR 2024) 및 공식 부록(`gkae236_supplemental_file.html`)에 명시된 **DMPNN-Des 공식 테스트 세트 실측치(Mean ± Std)**를 TDC 데이터셋의 **공식 목표 성능치(Target SOTA Threshold)**로 설정합니다.

---

## 2. 클러스터별 TDC 데이터셋 목표 성능치 일람표

### Cluster 1: Bio-Permeability & Oral Absorption (생체 투과 및 경구 흡수)
*수동 확산(Fick's law), 막 분배(LogD), 수용성 용해도(LogS) 및 P-gp 유출 펌프의 상호작용*

| TDC 데이터셋 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 $R^2$ (Test) | 목표 RMSE | 목표 MAE | 목표 ROC-AUC | 목표 ACC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`caco2_wang`** | Caco-2 Permeability | 회귀 | 906 | **0.743 ± 0.018** | **0.325 ± 0.013** | **0.242 ± 0.011** | - | - |
| **`lipophilicity_astrazeneca`**| logD7.4 | 회귀 | 4,200 | **0.902 ± 0.004** | **0.398 ± 0.008** | **0.290 ± 0.006** | - | - |
| **`solubility_aqsoldb`** | logS | 회귀 | 9,982 | **0.877 ± 0.013** | **0.746 ± 0.027** | **0.495 ± 0.010** | - | - |
| **`hia_hou`** | HIA (Human Intestinal Absorption) | 분류 | 578 | - | - | - | **0.897 ± 0.054** | **0.911 ± 0.018** |
| **`bioavailability_ma`** | F20% / F30% Oral Bioavailability| 분류 | 640 | - | - | - | **0.909 ± 0.047** | **0.857 ± 0.030** |
| **`pgp_broccatelli`** | Pgp-inhibitor | 분류 | 1,218 | - | - | - | **0.915 ± 0.011** | **0.850 ± 0.010** |
| *(확장)* `pgp_substrate` | Pgp-substrate | 분류 | 1,000+ | - | - | - | **0.892 ± 0.019** | **0.815 ± 0.034** |
| *(물리화학)* `mdck_permeability`| MDCK Permeability | 회귀 | 1,000+ | **0.700 ± 0.062** | **0.293 ± 0.034** | **0.205 ± 0.010** | - | - |

---

### Cluster 2: Plasma Distribution & Tissue Penetration (혈장 분포 및 조직 투과)
*Free Drug Hypothesis: 알부민 결합률(PPBR)이 비결합 분율(fu)을 결정하고, 뇌장벽(BBB) 및 전신 조직(VDss) 침투를 지배*

| TDC 데이터셋 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 $R^2$ (Test) | 목표 RMSE | 목표 MAE | 목표 ROC-AUC | 목표 ACC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ppbr_az`** | PPB (Plasma Protein Binding %) | 회귀 | 1,797 | **0.824 ± 0.031** | **11.382 ± 0.648** | **5.976 ± 0.429** | - | - |
| **`vdss_lombardo`** | VDss (Volume of Distribution) | 회귀 | 1,130 | **0.760 ± 0.045** | **0.301 ± 0.024** | **0.162 ± 0.010** | - | - |
| **`bbb_martins`** | BBB Penetration | 분류 | 2,050 | - | - | - | **0.908 ± 0.004** | **0.836 ± 0.014** |
| *(참조)* `fu` | Fraction Unbound in Plasma | 회귀 | 1,500+ | **0.894 ± 0.024** | **0.229 ± 0.027** | **0.135 ± 0.012** | - | - |

---

### Cluster 3: Cytochrome P450 Metabolism (약물 대사 효소 패널)
*5대 저해 효소 앵커(각 12,000+ 화합물) 및 3대 소규모 기질 turnover 패널*

| TDC 데이터셋 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 ROC-AUC (Test) | 목표 ACC (Test) | 목표 MCC (Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`cyp1a2_veith`** | CYP1A2 inhibitor | 분류 | 12,574 | **0.942 ± 0.003** | **0.874 ± 0.003** | **0.748 ± 0.006** |
| **`cyp2c19_veith`**| CYP2C19 inhibitor | 분류 | 12,092 | **0.915 ± 0.005** | **0.843 ± 0.009** | **0.689 ± 0.016** |
| **`cyp2c9_veith`** | CYP2C9 inhibitor | 분류 | 12,092 | **0.917 ± 0.009** | **0.854 ± 0.009** | **0.682 ± 0.020** |
| **`cyp2d6_veith`** | CYP2D6 inhibitor | 분류 | 13,130 | **0.886 ± 0.010** | **0.869 ± 0.012** | **0.575 ± 0.033** |
| **`cyp3a4_veith`** | CYP3A4 inhibitor | 분류 | 12,328 | **0.916 ± 0.005** | **0.834 ± 0.009** | **0.667 ± 0.020** |
| **`cyp2c9_substrate_carbonmangels`**| CYP2C9 substrate | 분류 | 666 | **0.782 ± 0.042** | **0.720 ± 0.049** | **0.433 ± 0.100** |
| **`cyp2d6_substrate_carbonmangels`**| CYP2D6 substrate | 분류 | 664 | **0.844 ± 0.057** | **0.754 ± 0.051** | **0.514 ± 0.099** |
| **`cyp3a4_substrate_carbonmangels`**| CYP3A4 substrate | 분류 | 667 | **0.798 ± 0.034** | **0.720 ± 0.012** | **0.444 ± 0.027** |

---

### Cluster 4: Pharmacokinetic Clearance & Elimination (약물 제거 및 소실 반감기)
*간 클리어런스($CL_{\text{int}}$), 전신 청정율($CL_{\text{plasma}} = f_u \cdot CL_{\text{int}}$) 및 반감기($t_{1/2} = \frac{0.693 V_{\text{d}}}{CL}$)*

| TDC 데이터셋 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 $R^2$ (Test) | 목표 RMSE | 목표 MAE | 목표 ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`half_life_obach`** | T1/2 (Elimination half-life) | 회귀 | 667 | **0.653 ± 0.070** | **0.877 ± 0.111** | **0.420 ± 0.026** | - |
| **`clearance_microsome_az`** | CL-plasma / HLM Clearance | 회귀 | 1,102 | **0.667 ± 0.046** | **2.912 ± 0.311** | **1.783 ± 0.160** | - |
| **`clearance_hepatocyte_az`**| CL-plasma / Hepatocyte | 회귀 | 1,213 | **0.667 ± 0.046** | **2.912 ± 0.311** | **1.783 ± 0.160** | - |
| *(참조 보조)* `hlm_stability` | HLM Metabolic Stability | 분류 | 3,500+ | - | - | - | **0.882 ± 0.007** |

---

### Cluster 5: Cardiac Safety & Broad Toxicity (심장 안전성 및 독성 프로파일링)
*hERG 칼륨 채널 차단, Ames 돌연변이성, 간독성(DILI) 및 12대 Tox21 핵수용체/스트레스 반응*

| TDC 데이터셋 | ADMETlab 3.0 대응 과제 | 유형 | 샘플 수 | 목표 ROC-AUC (Test) | 목표 ACC (Test) | 목표 MCC (Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`herg`** | hERG Blockers (cardiotoxicity) | 분류 | 648 | **0.937 ± 0.006** | **0.828 ± 0.017** | **0.680 ± 0.026** |
| **`dili`** | DILI (Drug-Induced Liver Injury) | 분류 | 475 | **0.860 ± 0.052** | **0.787 ± 0.075** | **0.590 ± 0.143** |
| **`ames`** | AMES Mutagenicity | 분류 | 7,255 | **0.882 ± 0.007** | **0.785 ± 0.015** | **0.576 ± 0.028** |
| **`skin_reaction`** | Skin Sensitization | 분류 | 404 | **0.787 ± 0.043** | **0.780 ± 0.062** | **0.491 ± 0.152** |
| **`carcinogens_lagunin`**| Carcinogenicity | 분류 | 280 | **0.715 ± 0.029** | **0.668 ± 0.040** | **0.349 ± 0.078** |
| **`tox21` (NR-AR)** | Nuclear Receptor - AR | 분류 | 7,831 | **0.883 ± 0.016** | **0.956 ± 0.016** | **0.534 ± 0.076** |
| **`tox21` (NR-AhR)** | Nuclear Receptor - AhR | 분류 | 7,831 | **0.924 ± 0.011** | **0.862 ± 0.040** | **0.550 ± 0.047** |
| **`tox21` (SR-MMP)** | Stress Response - MMP | 분류 | 7,831 | **0.941 ± 0.016** | **0.857 ± 0.048** | **0.604 ± 0.062** |
| **`tox21` (SR-p53)** | Stress Response - p53 | 분류 | 7,831 | **0.890 ± 0.027** | **0.882 ± 0.034** | **0.415 ± 0.036** |
| **`clintox`** | FDA Clinical Trial Failure | 분류 | 1,484 | **0.865 ± 0.011** | **0.761 ± 0.022** | **0.534 ± 0.036** |
