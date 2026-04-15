**RESEARCH PROPOSAL**

**BayesPFN: A Bayesian Prior-Fitted Network with Imbalance-Stratified Pretraining and Class-Conditional Conformal Inference for Reliable Tabular Classification**

**Author:** Bayu Satya Adhitama | NRP 6025252023
**Affiliation:** S2 Teknik Informatika (Research Track) | Institut Teknologi Sepuluh Nopember (ITS)
**Date:** April 2026

---

### **Abstract**
Tabular data is the most common data type in real-world machine learning applications. While **TabPFN** established **Prior-data Fitted Networks (PFNs)** as a breakthrough for small tabular datasets—achieving state-of-the-art performance in seconds without hyperparameter tuning—it exhibits **statistically significant bias toward majority classes** and fragility under **out-of-distribution (OOD) conditions**. These limitations are structural, stemming from a synthetic pretraining prior that does not systematically enforce imbalance diversity. 

This proposal introduces **BayesPFN**, a novel foundation model designed to address these failure modes through three core architectural contributions: (1) an **imbalance-stratified synthetic pretraining prior**; (2) a **dual-head transformer architecture** for joint class and epistemic uncertainty prediction; and (3) an **in-context class-conditional conformal calibration layer**. BayesPFN aims to deliver improved reliability and per-class coverage guarantees while maintaining single-GPU tractability.

---

### **1. Background and Motivation**

#### **1.1 The Rise of Prior-Fitted Networks**
PFNs represent a paradigm shift: instead of fitting a model to a specific dataset, they are trained offline once to approximate **Bayesian inference** on millions of synthetic datasets. At inference time, the model performs **in-context learning (ICL)**, treating the training set as a sequence of labeled tokens to yield predictions for test samples in a single forward pass. **TabPFN** is the leading instantiation, utilizing a transformer trained on **Structural Causal Models (SCMs)** and **Bayesian Neural Networks (BNNs)** to outperform gradient-boosted trees on small tabular data.

#### **1.2 Documented Limitations of TabPFN**
Recent evaluations identify three critical weaknesses in TabPFN:
*   **Majority Class Bias:** Degraded performance on imbalanced datasets because the SCM prior does not enforce imbalance diversity during pretraining.
*   **OOD Fragility:** Performance drops under distributional shifts as the in-context mechanism assumes exchangeable data.
*   **Lack of Uncertainty Guarantees:** Standard softmax outputs are uncalibrated, providing no formal coverage guarantees for minority classes.

#### **1.3 Gaps in the Existing Literature**
Current extensions like **BoostPFN** (scalability) and **EquiTabPFN** (equivariance) do not tackle imbalance or uncertainty. While post-hoc methods like **TACP** exist, they require separate calibration sets. **BayesPFN** is the first to integrate imbalance-aware pretraining and uncertainty estimation into a single, end-to-end architecture.

---

### **2. Proposed Method: BayesPFN**

#### **2.1 Innovation 1: Imbalance-Stratified Synthetic Pretraining Prior**
BayesPFN replaces TabPFN's uniform random prior with a **stratified prior** that explicitly controls imbalance ratios using a **Beta distribution**. This forces the meta-learner to internalize minority-class patterns by ensuring 40% of pretraining datasets have an imbalance ratio > 5:1 and 30% have > 10:1.

#### **2.2 Innovation 2: Dual-Head Transformer Architecture**
While standard PFNs use one classification head, BayesPFN adds a **second head for epistemic uncertainty**. 
*   **Head 1 (Classification):** Produces class logits via softmax.
*   **Head 2 (Uncertainty):** Estimates **scalar variance ($\sigma^2$)**, trained via an auxiliary **negative log-likelihood (NLL) loss** on held-out synthetic validation sets.
This allows the model to signal higher uncertainty for minority classes or OOD inputs where its predictions are likely to err.

#### **2.3 Innovation 3: In-Context Class-Conditional Conformal Calibration**
BayesPFN eliminates the need for external calibration data by partitioning the in-context window into a **D_train and D_cal (80/20 split)**. It computes per-class nonconformity scores and class-conditional quantiles during the forward pass. This provides a **guaranteed class-conditional coverage** $P(y \in C(x) | y = k) \geq 1-\alpha$ for every class, including the minority.

---

### **3. Infrastructure and Implementation**

#### **3.1 Hardware Resources**
| Resource | Provider | Specification | Est. Cost |
|----------|----------|---------------|-----------|
| GPU | Vast.AI | RTX 5090 (32GB VRAM) | ~$0.60/hr |
| Storage | Cloudflare R2 | Checkpoints + logs | Free tier (10GB) |
| Development | Google Colab | Free tier for inference testing | Free |

#### **3.2 Software Stack**
*   **Base Model:** TabPFN v2 (Apache 2.0, open weights)
*   **Framework:** PyTorch 2.x + CUDA 12.x
*   **Cloud Storage:** boto3 for R2 integration
*   **Logging:** Weights & Biases (wandb)
*   **Repository:** GitHub (`automl/TabPFN` base + forked bayespfn repo)

#### **3.3 Pretraining Dataset Scale**
Following iterative sanity-check approach:
*   **500 datasets** - Quick sanity check (< 10 min)
*   **1,000 datasets** - Validate training loop
*   **10,000 datasets** - Confirm scaling behavior
*   **50,000 datasets** - Final pretraining scale

This approach reduces risk and total GPU time while ensuring meaningful pretraining scale.

---

### **4. Experimental Design**

#### **4.1 Datasets and Baselines**
Evaluation will focus on high-imbalance benchmarks:
| Dataset | OpenML ID | Imbalance Ratio |
|---------|-----------|-----------------|
| Creditcard Fraud | 1597 | 577:1 |
| Mammography | 43893 | 42:1 |
| Yeast | 181 | 31:1 (majority:minority) |

Baselines include **XGBoost + SMOTE**, **TabPFN v2** (open weights), and **TACP**.

#### **4.2 Evaluation Metrics**
*   **Balanced Accuracy** - Arithmetic mean of per-class recall
*   **F1-Macro** - Arithmetic mean of per-class F1 scores
*   **Coverage Gap** - Novel metric: $\max |actual\_coverage - target\_coverage|$ across classes

#### **4.3 Ablation Study**
| Model | Innovations Included |
|-------|---------------------|
| TabPFN v2 (baseline) | None |
| BayesPFN-v1 | Innovation 1 only (imbalance-stratified prior) |
| BayesPFN-v2 | Innovations 1+2 (added dual-head uncertainty) |
| BayesPFN-v3 | All innovations (full BayesPFN) |

---

### **5. Session-Based Workflow**

Given budget constraints (~$6 USD total GPU budget), implementation is organized into **2 main Vast.AI sessions**:

#### **Session 1: Baseline + Imbalance-Stratified Prior (~4-5 hours)**
1. Environment setup (PyTorch, TabPFN v2 dependencies)
2. Baseline verification on OpenML datasets
3. Sanity check (500 → 1K → 10K datasets)
4. Scale to 50K datasets with stratified Beta prior
5. Quick evaluation on imbalanced benchmark
6. Upload checkpoint to R2

#### **Session 2: Dual-Head + Conformal + Evaluation (~4-5 hours)**
1. Resume from R2 checkpoint
2. Implement dual-head architecture (classification + uncertainty)
3. Implement in-context conformal calibration (80/20 split)
4. Full benchmark on Creditcard Fraud, Mammography, Yeast
5. Ablation study across all model variants
6. Upload final checkpoint + evaluation results

---

### **6. Architecture Comparison**

| Aspect | TabPFN v2 | BayesPFN |
|--------|-----------|----------|
| **Pretraining Prior** | Uniform random over SCMs | Stratified Beta prior (imbalance-aware) |
| **Imbalance Control** | None | P(ratio > 5:1) = 0.4, P(ratio > 10:1) = 0.3 |
| **Architecture** | Single classification head | Dual-head (classification + epistemic $\sigma^2$) |
| **Uncertainty** | Uncalibrated softmax | In-context conformal calibration (80/20 split) |
| **Pretraining Scale** | 10M synthetic datasets | 50K (scalable) |
| **Coverage Guarantee** | None | Per-class: $P(y \in C(x) \| y=k) \geq 1-\alpha$ |

---

### **7. Timeline**

| Phase | Task | Duration |
|-------|------|----------|
| **Phase 0** | Infrastructure setup (Vast.AI, R2, GitHub) | Week 1 |
| **Session 1** | Baseline + Innovation 1 | Week 1-2 |
| **Session 2** | Innovations 2+3 + Evaluation | Week 2-3 |
| **Analysis** | Ablation study, results compilation | Week 4 |
| **Writing** | Paper methodology + results | Week 5-6 |
| **Submission** | Expert Systems with Applications or EAAI | May 2026 |

---

### **8. Expected Outcomes**

1. **BayesPFN-v1**: Imbalance-stratified TabPFN (Innovation 1) - handles imbalance ratios up to 100:1
2. **BayesPFN-v2**: + Dual-head uncertainty estimation
3. **BayesPFN-v3**: Full model with conformal calibration guarantees
4. **Evaluation**: Benchmark tables showing improvement over TabPFN v2 baseline
5. **Paper**: Submission-ready manuscript for Expert Systems with Applications or EAAI

---

### **9. References**

1. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. *ICLR 2023*.
2. Hollmann, N., Müller, S., Purucker, L., et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature 637*.
3. Müller, S., Hollmann, N., Arango, S.P., Grabocka, J., & Hutter, F. (2022). Transformers can do Bayesian inference. *ICLR 2022*.
4. TabPFN GitHub: https://github.com/automl/TabPFN

---

### **10. Acknowledgments**

Created by Bayu Satya Adhitama based on TabPFN source material and original research proposal.
