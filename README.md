# Knowledge Distillation for Lightweight Diabetic Retinopathy Detection  
### A Multi-Scale Feature Fusion Approach

Authors: Akshita Shukla, Bhavya Jain, Sahil Dhiman, Shubham Goel  
Video Link: https://youtu.be/pqGsWe9Xv_o

---

## Executive Summary

Diabetic Retinopathy (DR) is a leading cause of preventable blindness among working-age adults, with India being severely affected due to its high diabetes burden (estimated 90+ million cases). While state-of-the-art deep learning models achieve high accuracy for DR detection and severity grading, they require substantial computational resources unavailable in rural healthcare settings. This report presents a novel lightweight architecture that combines Knowledge Distillation (KD), Convolutional Block Attention Module (CBAM), and Topological Data Analysis (TDA) to achieve 85% test accuracy on the APTOS 2019 dataset while maintaining CPU compatibility (49 ms inference on Intel i7, 21.2 GFLOPs). By addressing the rural healthcare gap through model compression and attention mechanisms, this work demonstrates a practical pathway toward democratizing AI-driven DR screening in resource-constrained clinical environments.

---

# 1. Introduction

## 1.1 Clinical Context and Motivation

Diabetic Retinopathy represents one of the most serious complications of diabetes mellitus, affecting approximately 12–17% of the global diabetic population[1]. In India, the diabetes capital of the world with ~90 million affected individuals, the prevalence of DR is alarming: 12–17% of diabetics have already developed DR, and 4% have vision-threatening forms[2]. Early-stage DR is typically asymptomatic, making automated screening critical for early intervention and vision preservation.

Traditional DR screening relies on ophthalmologist-based funduscopy and grading, which is:

- Expensive: Requires specialized medical professionals  
- Geographically limited: Rural regions lack adequate screening infrastructure  
- Scalability-limited: Cannot accommodate India's growing patient volume  

---

## 1.2 The Technical Gap: Resource Constraints in Rural Healthcare

State-of-the-art deep learning models for DR classification (e.g., ResNet152, InceptionV3) consistently achieve high accuracy (90%+) but require significant computational resources:
ResNet-101: 7.8 GFLOPs, ~600 ms inference on CPU
ConvNeXt-Small: 8.7 GFLOPs, ~670 ms inference on CPU
EfficientNet-B5: 9.9 GFLOPs, ~760 ms inference on CPU

In contrast, rural clinics operate on legacy CPUs or handheld devices with limited memory and power, creating a deployment barrier. This motivated our investigation into model compression and efficient architectures.

---

## 1.3 Research Objectives

This project addresses the following objectives:

- **Efficient Architecture Selection:** Identify CNN backbones offering optimal accuracy-to-compute trade-offs for DR classification  
- **Knowledge Distillation Framework:** Implement teacher-student distillation to compress high-performing models (EfficientNet-B5) into lightweight student architectures (EfficientNet-B2) while preserving accuracy  
- **Multi-Modal Feature Fusion:** Integrate CNN features with topological features via CBAM attention and TDA for improved lesion detection and explainability  
- **Class Imbalance Handling:** Apply stratified splitting and weighted sampling to address severe class skew in the APTOS 2019 dataset  
- **Real-World Deployment Feasibility:** Demonstrate CPU-compatible inference (<<1 second per image) for practical clinical deployment  

---

# 2. Literature Review

## 2.1 Diabetic Retinopathy Detection: Deep Learning Approaches

Recent literature demonstrates the efficacy of transfer learning and ensemble methods for DR detection. Jabbar et al. (2024) proposed a lesion-based DR detection framework using GoogleNet and ResNet models with adaptive filtering, achieving competitive accuracy across severity levels[3]. Similarly, Akhtar et al. (2025) developed RSG-Net, a deep neural network optimized for 4-class and 2-class DR grading with clinical-grade performance[4].

However, these approaches typically prioritize accuracy over efficiency, resulting in deployment challenges in resource-constrained settings. The gap between research-grade models and clinically-deployable systems remains substantial[5].

---

## 2.2 Knowledge Distillation in Medical Imaging

Knowledge Distillation (KD) has emerged as a powerful model compression technique that transfers knowledge from a complex teacher network to a lightweight student network via soft probability distributions. Zhao et al. (2023) demonstrated that structured knowledge distillation effectively reduces model size and computational requirements while maintaining segmentation accuracy in medical imaging tasks[6]. Recent advances in explainable knowledge distillation (Yu et al., 2025) have further enhanced interpretability in medical image classification[7].

For DR detection specifically, KD offers multiple benefits:

- Class Imbalance Mitigation: Soft labels from the teacher provide smoother decision boundaries for minority classes  
- Lesion Detection: Teacher's rich feature representations aid in detecting subtle lesions (microaneurysms, hemorrhages)  
- Generalization: Reduced overfitting on small, imbalanced medical datasets  
- Deployment Feasibility: Smaller student models suitable for clinical devices  

---

## 2.3 Attention Mechanisms in Medical Image Analysis

Convolutional Block Attention Module (CBAM), proposed by Woo et al. (2018), is a lightweight, plug-and-play attention mechanism that learns channel and spatial attention without significant computational overhead[8]. CBAM has been successfully applied across classification and detection tasks, producing clearer GradCAM visualizations for explainability—critical for clinical AI adoption.

For DR detection, CBAM is particularly valuable because:

- Spatial Attention: Highlights lesion-rich retinal regions (avoiding feature confusion with healthy tissue)  
- Channel Attention: Emphasizes discriminative feature channels for distinguishing adjacent DR severity levels  
- Explainability: Provides interpretable attention maps for clinician validation  
- Efficiency: Negligible computational overhead  

---

## 2.4 Topological Data Analysis for Feature Extraction

Topological Data Analysis (TDA) and persistence homology offer topology-based feature extraction complementary to CNN features. Persistence diagrams capture the "birth" and "death" scales of topological features, providing rotation-, translation-, and lighting-invariant descriptors. Recent work has demonstrated TDA's effectiveness in feature extraction for medical imaging with limited training data.

For DR detection, TDA offers:

- Invariance: Lesion morphology preserved across varying imaging conditions (critical for rural settings with suboptimal lighting)  
- Complementary Features: Topological features capture different information than spatial CNN features  
- Robustness: Enhanced performance on small, imbalanced datasets via topological structure preservation  

# 3. Dataset and Preprocessing

## 3.1 APTOS 2019 Dataset

The APTOS 2019 Blindness Detection dataset is a publicly available, clinically annotated fundus image dataset comprising:

- **Sample Size:** 3,662 labeled images  
- **Severity Classes:** 5 levels (0–4 per international DR grading standards)

| Class | Count |
|-------|--------|
| Class 0 (No DR) | 1,805 images |
| Class 1 (Mild DR) | 370 images |
| Class 2 (Moderate DR) | 999 images |
| Class 3 (Severe DR) | 193 images |
| Class 4 (Proliferative DR) | 295 images |

- **Annotation Quality:** Clinically reviewed via standard DR severity scale  
- **Imbalance Ratio:** 9.7:1 (No DR to Proliferative DR), presenting significant class imbalance  

---

## 3.2 Data Preprocessing Pipeline

### 3.2.1 Class Imbalance Mitigation

**Stratified Splitting:**  
To preserve class distribution across train/validation/test splits, we employed StratifiedShuffleSplit with 70/15/15 allocation. This ensures minority classes are adequately represented in all splits.

**Weighted Random Sampling:**  
Applied during training to enforce balanced mini-batch composition. For each class \( i \), the sampling weight is: w_i = N / n_i

where:  
- \( N \) = total training samples  
- \( n_i \) = samples in class \( i \)

This approach ensures the model observes minority classes as frequently as majority classes during training, preventing bias toward the No DR class.

**Effective Number of Samples:**  
To prevent overfitting on minority classes as dataset volume grows, we adopted the Effective Number of Samples principle: E(n_i) = (1 - β^(n_i)) / (1 - β)


where:  
- β is a hyperparameter  
- \( n_i \) is the class sample count  

This provides principled class weighting that diminishes information gain from redundant samples.

---

### 3.2.2 Data Augmentation

**Geometric Transformations:**

- RandomHorizontalFlip  
- RandomVerticalFlip  
- RandomRotation(±20°)

**Photometric Transformations:**

- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3)  

**Input Standardization:**

- Resize to **224×224**  
- Normalize using ImageNet statistics  

---

## 3.3 Train/Validation/Test Splits

| Split | Count | Class Distribution |
|--------|--------|----------------------|
| Training | 2,564 | Stratified |
| Validation | 549 | Stratified |
| Test | 549 | Stratified |

---

# 4. Methodology

## 4.1 Architecture Exploration Phase

### 4.1.1 Baseline Model Evaluation

We systematically evaluated leading CNN architectures to identify efficient backbones:

**EfficientNet-B0:**  
- Baseline efficiency-focused model  
- Architecture: Compound scaling  
- Result: 81–83% accuracy (~1.9 GFLOPs)  

**ResNet-50:**  
- Strong on majority class  
- Weak on rare classes  
- Result: 72–83% accuracy  

**ResNeXt-50:**  
- Group convolutions  
- Heavy undersampling caused low performance  
- Result: 60–72% accuracy  

**ConvNeXt (ModernNet):**  
- Kernel sweep (k=3 → k=9)  
- k=9 extracted larger lesions better  
- Result: 76–80% accuracy  

### 4.1.2 Efficiency-Accuracy Trade-off Analysis

A key insight emerged: **Scale is not static**.  
DR features vary significantly in size:

- Microaneurysms: <10 pixels  
- Hemorrhages: >100 pixels  

Architectures with fixed receptive fields struggle.  
EfficientNet models adapt well due to **compound scaling**.

**Final Choice:**

- **Teacher:** EfficientNet-B5  
- **Student:** EfficientNet-B2  
- Distillation bridges accuracy + efficiency  

---

## 4.2 Proposed Architecture: Knowledge Distillation Framework

### 4.2.1 Knowledge Distillation Design

**Teacher-Student Setup:**

- Teacher → EfficientNet-B5  
- Student → EfficientNet-B2  

**Loss Function:**

L_total = α * KL(T_s || S_s) + (1 - α) * CE(S, y)

**Distillation Loss:**

KL(softmax(T/τ), softmax(S/τ))


Where:

- τ = temperature  
- α = interpolation hyperparameter  

**Hyperparameters:**

- Temperature: τ = 4  
- Alpha schedule: 0.2 → 0.8 over epochs  
- Optimizer: AdamW  
- Scheduler: OneCycleLR  

---

## 4.2.2 Convolutional Block Attention Module (CBAM)

CBAM refines feature maps using sequential:

- **Channel Attention**  
- **Spatial Attention**

Equations:

Channel Attention: M_c = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
Spatial Attention: M_s = σ(f7×7([AvgPool(F_c); MaxPool(F_c)]))
Final Output: F' = M_s ⊗ M_c ⊗ F


Advantages for DR detection:

- Highlights lesion-rich areas  
- Emphasizes discriminative channels  
- Provides interpretable attention maps  
- Lightweight overhead  

---

## 4.2.3 Topological Data Analysis (TDA) Branch

Pipeline:

1. Convert image to grayscale  
2. Build cubical complex  
3. Extract H0 and H1 persistence diagrams  
4. Convert to **persistence image** (32×32)  
5. Feed to 3-layer MLP (100 → 128 → 1024)

TDA benefits:

- Invariance to rotation/translation  
- Captures lesion morphology  
- Complements CNN spatial features  

---

## 4.2.4 Fusion Architecture


---

## 4.3 Training Protocol

- Optimizer: AdamW (weight decay = 0.01)  
- Scheduler: OneCycleLR (max_lr = 1e-3)  
- Batch Size: 32  
- Early Stopping: Patience = 15  
- Label Smoothing: 0.1  
- Dropout: 0.5  
- Class Weighting: Effective Number of Samples  
- Hardware:  
  - Train: NVIDIA RTX 4060
  - Inference: Intel i7/i5 CPUs  





