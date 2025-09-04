# Machine Learning Lab 5: Dimensionality Reduction Techniques

## Overview

This lab focuses on implementing and understanding various dimensionality reduction techniques in machine learning. The primary methods explored include:

- **Principal Component Analysis (PCA):** An unsupervised technique that transforms data into a set of orthogonal components, capturing the maximum variance.
- **Linear Discriminant Analysis (LDA):** A supervised method that finds the linear combinations of features that best separate different classes.
- **Kernel Principal Component Analysis (KPCA):** An extension of PCA that uses kernel methods to handle non-linear data structures.

The lab uses the Wine dataset for PCA and LDA experiments, and synthetic datasets (half-moons and circles) for KPCA experiments.

## Objectives

By the end of this lab, you should be able to:

- Implement PCA from scratch and using scikit-learn.
- Apply LDA for supervised dimensionality reduction.
- Use KPCA to handle nonlinear data.
- Visualize results and analyze the impact of dimensionality reduction on classification tasks.

## Requirements

Ensure you have the following Python packages installed:

```bash
pip install numpy scipy matplotlib scikit-learn```


## Analysis Questions and Answers


1. How does explained variance change with components in PCA?
 The first principal components explain the most of the variance in the dataset.  For example, in the Wine dataset, the first 2-3 components account for the majority of the variance, with subsequent components contributing minimally.

 2. Why could LDA outperform PCA in classification?
 LDA is supervised and uses class labels to maximize class separability, whereas PCA is unsupervised and just maximizes variance.  LDA typically improves classification performance.

 3. What effect does the gamma parameter in KPCA have on nonlinear data separation?
 Gamma determines the width of the RBF kernel.  Higher gamma values result in tighter, more localized mappings, whilst lower gamma values give smoother mappings.  Gamma adjustment influences the capacity to separate nonlinear patterns.

4. Evaluate classifier performance (accuracy, runtime) using raw, PCA-transformed, and LDA-transformed data.

 Raw data includes baseline accuracy and runtime.

 Reduced dimensionality in PCA can reduce runtime while marginally reducing accuracy.

 LDA: Increases accuracy by maximizing class separation; runtime is often marginally lowered.

 5. When may PCA fail, and how does KPCA handle nonlinearity?
 PCA fails on non-linearly separable data because it can only detect linear patterns.  KPCA uses kernel functions to map data into a higher-dimensional space, allowing for linear separation.

 6. Observations on visuals

 PCA plots depict variance-maximizing directions; class separation may overlap.

 LDA plots show clearer class separation.

 KPCA plots effectively differentiate non-linear patterns, as proven with half-moon and circle datasets.
