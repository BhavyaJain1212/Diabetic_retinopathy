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


