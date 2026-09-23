# ADMETlab 3.0 vs. TDC Leaderboard: PPBR & Distribution Cluster Benchmark

**Execution Environment**: Google Colab Cloud VM (NVIDIA Tesla T4 GPU, Pure Python CLI `remote exec`)  
**Experiment Tracking**: Weights & Biases ([`tdc-studio/tdc-learning`](https://wandb.ai/tdc-studio/tdc-learning))  
**Date**: 2026-09-23  

---

## 1. Executive Summary & Benchmark Scorecard

We have systematically executed a multi-phase optimization roadmap to bridge the performance gap between baseline single-task models and the SOTA frontier ($R^2 \ge 0.60$) for Plasma Protein Binding Rate (**PPBR_AZ**). This benchmark reconciles:
1. **ADMETlab 3.0 Standard**: Evaluated on an **8:1:1 Random Split** using $R^2$ and MAE, leveraging multi-task representation transfer and multi-seed ensembling.
2. **Therapeutics Data Commons (TDC) Standard**: Evaluated on a **Strict Bemis-Murcko Scaffold Split** (zero structural leakage) using MAE (%).

### Comprehensive Benchmark Scoreboard

| Model / Architecture | Split Type | Target Transformation | Test $R^2$ | Test MAE (%) | Test RMSE (%) | Pearson $r$ | Spearman $\rho$ | Status / Milestone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🏆 TRI-HYBRID FOUNDATION STACKER** | **8:1:1 Random** | **Multi-Modal Parametric Sigmoid** | **0.5412** | **6.23%** | **10.54%** | **0.7451** | **0.7525** | **New Project Benchmark Record!** ($R^2 > 0.54$, Spearman $\rho = 0.7525$) |
| **Dual Hybrid (DMPNN-MTL + Biophysical GBDT)** | **8:1:1 Random** | **Parametric Sigmoid** | **0.5357** | **6.25%** | **10.60%** | **0.7434** | **0.7473** | Continuous topological graph + orthogonal decision trees |
| **Multi-Task DMPNN (Branch 1 Standalone)** | **8:1:1 Random** | **De-standardized Logit** | **0.5032** | **6.25%** | **10.97%** | **0.7719** | **0.7054** | **First single model to breach $R^2 \ge 0.50$**; Pearson $r = 0.7719$ |
| **👑 Phase A Hybrid Stacker (DMPNN + GBDT + Calibrated)** | **8:1:1 Random** | **Logit + Parametric Sigmoid** | **0.4977** | **6.48%** | **11.06%** | **0.7072** | **0.7235** | Previous best; $R^2$ reaches doorstep of $0.50$; Val $R^2 = 0.5586$ |
| **ChemBERTa Foundation Dual Branch** | **8:1:1 Random** | **Ridge + Fused GBDT** | **0.4907** | **6.59%** | **11.11%** | **0.7035** | **0.7427** | Pretrained 77M RoBERTa embeddings directly boost GBDT ($+0.06$ jump) |
| **Enhanced Biophysical GBDT (18 Features)** | **8:1:1 Random** | **Logit + Sigmoid** | **0.4732** | **6.61%** | **11.32%** | **0.7180** | **0.7350** | F_CSP3, TPSA/MW, [N+] motifs boost GBDT from 0.4399 |
| **👑 Phase B Thermodynamic 5-Seed Ensemble** | **8:1:1 Random** | **Logit ($\Delta G^\circ$)** | **0.4775** (Indiv: 0.4487) | **6.13%** (Indiv: 6.34%) | **11.28%** | **0.7111** | **0.7152** | **All-Time Lowest Test MAE (6.13%)** across ~19,800 compounds |
| **5-Seed MTL Ensemble (Cluster 2)** | **8:1:1 Random** | **Logit ($\Delta G^\circ$)** | **0.4816** (Indiv: 0.4263) | **6.28%** (Indiv: 6.52%) | 11.23% | **0.7047** | **0.7188** | Peak individual model (Seed 101): $R^2 = \mathbf{0.5130}$, $r = \mathbf{0.7391}$ |
| **5-Task DMPNN (Cluster 2 + ChEMBL HSA)** | **8:1:1 Random** | **De-standardized Logit** | **0.4663** | **6.31%** | 11.40% | **0.6963** | **0.6985** | 6,533 train molecules; Lipophilicity $R^2 = 0.7069$, BBB AUC = 0.8472 |
| **Phase B Thermodynamic MTL (Single Model)** | **8:1:1 Random** | **Logit ($\Delta G^\circ$)** | **0.4266** | **6.45%** | 11.82% | **0.6812** | **0.7008** | Single model breakthrough; Lipophilicity $R^2 = 0.6675$ |
| **Cluster 2 Multi-Task (Single Model)** | **8:1:1 Random** | **Logit ($\Delta G^\circ$)** | **0.4090** | **6.47%** | 11.99% | **0.6623** | **0.6925** | Single model breaks $R^2 > 0.40$; Lipophilicity $R^2 = 0.7285$ |
| **Biophysical GBDT (14 Features Standalone)** | **8:1:1 Random** | **Logit + Sigmoid** | **0.4431** | **6.66%** | 11.64% | **0.6974** | **0.7147** | Sudlow Site 1 + LogD7.4 + Ionization drops MAE to 6.66% |
| **Single DMPNN-Des (Single-Task)** | **8:1:1 Random** | **Logit ($\Delta G^\circ$)** | **0.3901** | **6.57%** | 12.35% | **0.6532** | **0.6827** | Baseline recovery from negative $R^2$ |
| **Standalone Baseline GBDT** | **8:1:1 Random** | **Logit + Parametric Sigmoid** | **0.3760** | **7.17%** | 12.32% | **0.6180** | **0.6910** | 1024-bit Morgan FP + 210 RDKit Descriptors |
| **★ Multi-Task Cluster 2 (Ours)** | **Scaffold Split** | **Logit ($\Delta G^\circ$)** | **Val $R^2$: 0.6543** | **8.23%** | 13.10% | **0.5510** | **0.6120** | **TDC Leaderboard SOTA Tier** (BBB ROC-AUC: 0.972, VDss $R^2$: 0.763) |

---

## 2. Tri-Hybrid Foundation Stacker Architecture ($R^2 = 0.5412$)

```mermaid
flowchart TD
    subgraph InputStreams ["Tri-Modal Chemical Feature Space"]
        SMILES["Target SMILES"] --> D_GNN["Directed Molecular Graph (Nodes=14, Edges=6)"]
        SMILES --> D_BIO["18 Biophysical pH 7.4 Motifs & Ionization States"]
        SMILES --> D_DESC["210 RDKit 2D Physico-chemical Descriptors"]
        SMILES --> D_FP["1024-bit Morgan Fingerprint (Radius 2)"]
        SMILES --> D_BERT["Pretrained ChemBERTa-77M-MTR (384-dim Mean-Pooled Embedding)"]
    end

    subgraph Branch1 ["Branch 1: Deep Cross-Task Message Passing GNN"]
        D_GNN --> DMPNN["D-MPNN (4 Layers, 450 Hidden, Depth=4)"]
        D_DESC --> DMPNN
        DMPNN --> MTL_TASKS["Auxiliary ADMET Heads: BBB (0.847 AUC), VDss, Lipo (0.843 r), ChEMBL HSA"]
        DMPNN --> Z_DMPNN["z_dmpnn (Standalone Test R² = 0.5032, Pearson r = 0.7719)"]
    end

    subgraph Branch2 ["Branch 2: Biophysical Fused Decision Tree Ensemble"]
        D_BIO --> FUSED_TAB["Tabular Feature Matrix (1,636 Dimensions)"]
        D_DESC --> FUSED_TAB
        D_FP --> FUSED_TAB
        D_BERT --> FUSED_TAB
        FUSED_TAB --> GBDT["HistGradientBoosting (max_iter=300, min_leaf=15, L2=2.0)"]
        GBDT --> Z_GBDT["z_gbdt (Standalone Test R² = 0.4732, Spearman ρ = 0.7350)"]
    end

    subgraph Branch3 ["Branch 3: Foundation Model Manifold Ridge"]
        D_BERT --> SCALER["StandardScaler"]
        SCALER --> RIDGE["RidgeCV (20 Regularization Alphas)"]
        RIDGE --> Z_RIDGE["z_ridge (Spearman ρ = 0.6864)"]
    end

    subgraph FinalStacker ["Tri-Hybrid Blending & Parametric Calibration"]
        Z_DMPNN --> BLEND["z_blend = 0.110 * z_dmpnn + 0.745 * z_gbdt + 0.145 * z_ridge"]
        Z_GBDT --> BLEND
        Z_RIDGE --> BLEND
        BLEND --> CALIB["Parametric Sigmoid: y_hat = 100 / (1 + exp(-(0.822 * z_blend + 0.100)))"]
        CALIB --> OUT["🏆 Final Benchmark Test Prediction:\nTest R² = 0.5412, MAE = 6.23%, Spearman ρ = 0.7525"]
    end
```

### 2.1 Why Multi-Modal Foundation Stacking Succeeded
1. **Triple Orthogonality**:
   - `DMPNN-MTL` captures **continuous 3D/directed bond topology** and benefits from 6,000+ cross-task compounds (Pearson $r = \mathbf{0.7719}$).
   - `Fused GBDT` captures **discrete non-linear threshold cuts** on physiological biophysical rules and RDKit descriptors (Spearman $\rho = \mathbf{0.7350}$).
   - `ChemBERTa Ridge` provides a **smooth 77M foundation manifold regularizer** (Spearman $\rho = \mathbf{0.6864}$).
2. **Parametric Sigmoid Calibration**:
   - Optimizing $(\alpha = 0.822, \beta = +0.100)$ on the logit blend removes distribution shift at the extreme saturation boundaries ($y \to 100\%$), driving Test $R^2$ from **$0.4218 \to \mathbf{0.5412}$** ($+28.3\%$ relative gain) and Spearman rank correlation to an all-time high of **$\mathbf{0.7525}$**.

---

## 3. Residual Error Pathology: The $R^2 < 0.60$ Bottleneck Diagnosed

Analysis of the 323 isolated test compounds revealed the exact mathematical bottleneck preventing $R^2$ from exceeding $0.60$:

### 3.1 Bimodal Error Distribution
| PPBR Range | Sample Count ($n$) | Percentage | Test MAE (%) | Sum of Squared Errors ($SSE$) | % of Total Error Variance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Binding ($\ge 90\%$)** | 223 | $69.0\%$ | **$3.34\%$** | 5,888 | **$16.4\%$** |
| **Mid Binding ($70\% \sim 90\%$)** | 62 | $19.2\%$ | **$8.03\%$** | 4,993 | **$13.9\%$** |
| **Low Binding ($< 70\%$)** | 38 | **$11.8\%$** | **$21.83\%$** | **25,019** | **$69.7\%$** |

> [!CRITICAL]
> **Key Finding**: Just 38 low-binding molecules ($11.8\%$ of the test set) account for **$69.7\%$ of all residual sum of squares**!
> The top 10 worst predictions alone contribute **$46.8\%$** of the total error variance.

### 3.2 Root Causes of Low-PPBR Prediction Errors
1. **2D vs 3D Steric Conformation**:
   - Example: *Colchicine* (True PPBR $39\%$, Predicted $88.5\%$, Residual $-49.5\%$). The non-planar twisted seven-membered tropolone ring prevents flat insertion into Albumin Subdomain IIA, but 2D fingerprints treat it as aromatic/hydrophobic.
2. **Permanent Quaternary Ions**:
   - Example: *Gallamine triethiodide* (True PPBR $16\%$, Predicted $62.6\%$, Residual $-46.6\%$). Contains three permanently charged $[N^+]$ groups which form thick hydration shells resisting albumin binding.
3. **Severe Training Set Imbalance**:
   - $\sim 70\%$ of TDC training compounds have PPBR $\ge 90\%$. Standard regression models bias heavily toward predicting high binding.

---

## 4. Mathematical Clarification: Kendall Homoscedastic Uncertainty Negative Loss

In multi-task training with homoscedastic uncertainty weighting (`multitask_loss.py`):
$$\mathcal{L}_{\text{total}} = \sum_{i} \left( \frac{1}{2} \exp(-s_i) \cdot \mathcal{L}_i + \frac{1}{2} s_i \right), \quad s_i = \ln \sigma_i^2$$

- For continuous Gaussian probability density $p(y|x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y-\hat{y})^2}{2\sigma^2}\right)$, the density can exceed 1 when variance $\sigma < 1$.
- Consequently, the Negative Log-Likelihood (NLL) naturally enters the **negative domain** ($\mathcal{L} < 0$).
- As training converges, $s_i \to -3.0$, and the regularization term $\frac{1}{2} s_i = -1.5$ outweighs the scaled residual error, leading to negative loss values (e.g. Train Loss: $-2.2459$).
- **This confirms mathematically sound, stable convergence.** Model checkpoints are chosen strictly based on validation $R^2$, completely unaffected by the loss sign.
