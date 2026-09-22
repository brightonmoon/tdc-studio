# Caco-2 Permeability Benchmark Evolution Report (Phase 1 ~ Phase 3)

## 1. Executive Summary

This report documents the iterative engineering progression on the TDC `caco2_wang` benchmark (906 molecules total, scaffold split: 634 train, 91 valid, 181 test), moving from an initial baseline of $R^2 \approx 0.0$ to a single-model peak of $R^2 = 0.6503$ (Pearson $r = 0.8210$) and a 5-model ensemble score of $R^2 = 0.6292$ (Val Pearson $r = 0.8303$) on NVIDIA T4 GPU via Google Colab.

---

## 2. Benchmark Evolution Across Phases

| Phase / Milestone | Core Methodology | Test $R^2$ | Test Pearson ($r$) | Test MAE | Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Initial Pipeline** | 5-Epoch MSE un-tuned Graph Transformer (pipeline smoke check) | -0.16 ~ +0.05 | 0.3567 | 0.5855 | 0.8285 |
| **Normal 50-Epoch HPO** | 50 Epochs, Cosine Annealing, Smooth L1 Loss, Composite Metric | +0.5009 (Val) | 0.7356 | 0.3993 | 0.5427 |
| **Phase 1** | Target Standardization + 210 RDKit 2D Physico-chemical Descriptors | +0.5340 | 0.7788 | 0.3747 | 0.4685 |
| **Phase 2** | D-MPNN (Directed Message Passing on Chemical Bonds) + Descriptors | +0.6012 | 0.8057 | 0.3476 | 0.4335 |
| **Phase 3** | 5-Model D-MPNN-Des Ensemble (Seeds: 42, 43, 44, 45, 46) | **+0.6292** *(M1: 0.6503)* | **0.8068** *(Val: 0.8303)* | **0.3420** | **0.4179** |
| **Literature SOTA** | Chemprop D-MPNN-Des (Yang et al. 2019 / Heid et al. 2024) | **0.743 ± 0.018** | ~0.86 | 0.242 ± 0.011 | 0.325 ± 0.013 |

---

## 3. Detailed Results of Phase 3 (5-Model Ensemble)

```
             ★ Ensemble Benchmark Results: CACO2_WANG (N=5 Models)              
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric ┃  M#1   ┃  M#2   ┃  M#3   ┃  M#4   ┃  M#5   ┃  Indiv Mean ┃ ★ Ensemble ┃ Literature ┃
┃        ┃  (42)  ┃  (43)  ┃  (44)  ┃  (45)  ┃  (46)  ┃   ± Std     ┃  Average   ┃    SOTA    ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ R²     │ 0.6503 │ 0.6046 │ 0.5621 │ 0.5965 │ 0.6381 │ 0.610±0.031 │   0.6292   │ 0.743±0.018│
│ RMSE   │ 0.4059 │ 0.4316 │ 0.4542 │ 0.4360 │ 0.4129 │ 0.428±0.017 │   0.4179   │ 0.325±0.013│
│ MAE    │ 0.3345 │ 0.3547 │ 0.3675 │ 0.3485 │ 0.3357 │ 0.348±0.012 │   0.3420   │ 0.242±0.011│
│ Pearson│ 0.8210 │ 0.7879 │ 0.7919 │ 0.7938 │ 0.8070 │ 0.800±0.012 │   0.8068   │   ~0.86    │
│Spearman│ 0.7949 │ 0.7414 │ 0.7499 │ 0.7375 │ 0.7606 │ 0.757±0.020 │   0.7636   │   ~0.83    │
└────────┴────────┴────────┴────────┴────────┴────────┴─────────────┴━━━━━━━━━━━━┴━━━━━━━━━━━━┘
```

* **W&B Run URI:** `https://wandb.ai/tdc-studio/tdc-learning/runs/omnknrra`

---

## 4. Key Engineering Insights & Next Phase (Phase 4)

1. **Root Cause of Single-Task Plateau ($R^2 \approx 0.65$):**
   - Sample complexity: 634 molecules is insufficient to generalize over complex, unconstrained scaffold shifts without external biological priors.
2. **Next Frontier (Phase 4: Bio-Permeability Multi-Task Learning):**
   - Jointly train with correlated ADMET endpoints:
     - `lipophilicity_astrazeneca` (4,200 compounds): Membrane partitioning driver.
     - `solubility_aqsoldb` (9,982 compounds): Aqueous dissolution bottleneck.
     - `hia_hou` (578 compounds): In vivo intestinal absorption counterpart.
   - Expanding total training compounds from 634 to 15,000+ to break through the 0.75+ SOTA barrier.
