# Alzheimer's Disease Detection using Self-Supervised Learning (SSL)

This repository contains the implementation of three state-of-the-art Self-Supervised Learning (SSL) frameworks to classify Alzheimer's Disease from MRI scans. The project compares **SimCLR**, **BYOL**, and **MAE** against a supervised baseline, focusing on label efficiency and representation learning.

##  Project Overview
- **Course:** CSE 475 (Machine Learning)
- **Domain:** Medical Imaging / Healthcare AI
- **Objective:** Evaluate SSL models on Alzheimer's MRI data to reduce reliance on large labeled datasets.

##  Models Implemented
1. **Supervised Baseline:** DenseNet121 and Vision Transformer (ViT) trained from scratch.
2. **SimCLR:** Contrastive learning with aggressive data augmentations.
3. **BYOL:** A non-contrastive approach using online and target networks.
4. **MAE (Masked Autoencoders):** A reconstruction-based approach utilizing a Vision Transformer (ViT) backbone with a high masking ratio (75%).

##  Project Structure
- `supervised-baseline.ipynb`: Standard supervised training.
- `simclr-pretraining.ipynb`: SimCLR pre-training and evaluation.
- `byol-pretraining.ipynb`: BYOL pre-training and evaluation.
- `mae-pretraining.ipynb`: Masked Autoencoder implementation.
- `task7-statistics.ipynb`: Statistical analysis (Friedman & Nemenyi tests).
- `Reports/`: Full technical paper-style report.

##  Key Features
- **Label Efficiency:** Models evaluated on 1%, 5%, 10%, 25%, and 50% labeled data.
- **Ablation Studies:** Analysis of hyperparameters like ColorJitter, Momentum (tau), and Masking Ratio.
- **Statistical Significance:** Performance validated using Friedman and Nemenyi Post-hoc tests.
- **Checkpointing:** Resumable training loops with saved state dictionaries.
