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
| **Phase 3** | 5-Model Single-Task D-MPNN-Des Ensemble (Seeds 42~46) | +0.6292 *(M1: 0.6503)* | 0.8068 *(M1: 0.8210)* | 0.3420 | 0.4179 |
| **Phase 4 (MTL)** | **Bio-Permeability Multi-Task D-MPNN-Des (14,000+ compounds)** | **+0.6986** | **0.8411** | **0.3241** | **0.3980** |
| **Literature SOTA** | Chemprop D-MPNN-Des 5-Model Ensemble (Yang et al. 2019) | **0.743 ± 0.018** | ~0.86 | 0.242 ± 0.011 | 0.325 ± 0.013 |

---

## 3. Detailed Results of Phase 4 (Bio-Permeability Multi-Task Learning)

```
      ★ Multi-Task Benchmark Results: CACO2_WANG (Single Model, Seed 42)       
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric       ┃ Phase 3 ST (Peak) ┃ Phase 4 MTL (Test) ┃ Literature SOTA ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ R²           │ 0.6503            │ 0.6986 (+0.0483)   │ 0.743±0.018     │
│ RMSE         │ 0.4059            │ 0.3980 (-0.0079)   │ 0.325±0.013     │
│ MAE          │ 0.3345            │ 0.3241 (-0.0104)   │ 0.242±0.011     │
│ Pearson (r)  │ 0.8210            │ 0.8411 (+0.0201)   │ ~0.86           │
│ Spearman (ρ) │ 0.7949            │ 0.8242 (+0.0293)   │ ~0.83           │
└──────────────┴───────────────────┴────────────────────┴─────────────────┘
```

* **Best Validation $R^2$:** 0.6717 (Epoch 36, up from ~0.50 in Single-Task)
* **W&B Run URI:** [train_caco2_bio_permeability_mtl (Run 5v9sunex)](https://wandb.ai/tdc-studio/tdc-learning/runs/5v9sunex)
* **Training Time:** ~150s on Colab NVIDIA T4 GPU (50 epochs)

---

## 4. Key Engineering Insights & Next Phase (Phase 5)

1. **Impact of Bio-Permeability Multi-Task Learning:**
   - Single-task Caco-2 training is fundamentally bottlenecked by 634 training scaffolds.
   - Auxiliary biophysical learning across Lipophilicity ($\log D_{7.4}$), Aqueous Solubility ($\log S$), and Human Intestinal Absorption (HIA) supplied 14,000+ compounds of structural diversity, raising the base feature representation capacity.
   - Jumped from $R^2 = 0.6503 \rightarrow 0.6986$ and Pearson $r = 0.8210 \rightarrow 0.8411$ on the identical unstandardized 181-compound test set with zero data leakage.

2. **Next Frontier (Phase 5: 5-Model Multi-Task Ensemble):**
   - Literature SOTA ($R^2 = 0.743 \pm 0.018$) is established using a 5-model ensemble in Chemprop.
   - Since our single MTL model already achieves $R^2 = 0.6986$, a 5-seed consensus ensemble (Seeds 42, 43, 44, 45, 46) is positioned to reduce individual model variance and reach the target $R^2 \ge 0.74$ SOTA zone.

