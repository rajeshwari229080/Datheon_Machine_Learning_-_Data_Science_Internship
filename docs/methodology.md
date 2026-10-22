# Scientific & Engineering Methodology

## Overview
This document delineates the algorithmic foundations, mathematical rationale, data hygiene protocols, and evaluation paradigms implemented across the **Datheon Machine Learning & Data Science Internship Suite**.

---

## 1. Principles of Data Hygiene & Leakage Prevention
Data leakage occurs when information from outside the training dataset is utilized to create the model. In production machine learning, leakage leads to unrealistically optimistic validation metrics followed by catastrophic degradation in live deployment.

### Safeguards Implemented:
1. **Pre-Split Partitioning**: Data is split into training (80%) and testing (20%) partitions *before* computing imputers, standardizers, encoders, or vocabulary indices.
2. **Encapsulation via Scikit-learn Pipelines**: All transformers are tied directly to estimators in `Pipeline` and `ColumnTransformer` constructs. During cross-validation, transformers are re-fitted strictly on training folds.
3. **Stratification**: All classification tasks (Iris, Telco Churn, Review Sentiment) employ stratified sampling to ensure target class ratios are identical between training and test sets.

---

## 2. Mathematical Foundations by Task

### Task 1: Multi-Class Benchmark Classification
- **Logistic Regression (Multinomial Softmax)**:
  $$\hat{P}(Y=k \mid X) = \frac{\exp(\beta_k^T X)}{\sum_{j=1}^K \exp(\beta_j^T X)}$$
  Standardizes features to zero mean and unit variance so L2 penalty applies equitably across disparate biometric scales.
- **Decision Tree (CART)**:
  Recursively optimizes the Gini impurity:
  $$I_G(t) = 1 - \sum_{i=1}^C p(i \mid t)^2$$
  Depth regularization (`max_depth=4`) prevents single-sample leaf overfitting.
- **2D Boundary Projection Principle**:
  A common anti-pattern is plotting arbitrary 2D slices while holding other features constant and falsely claiming it represents the model's 4D surface. We train a dedicated 2-feature model specifically for geometric visualization and clearly document its subspace limitations.

### Task 2: California Housing Price Regression
- **Ordinary Least Squares (OLS)**:
  $$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$
- **Ridge Regression (L2 Shrinkage)**:
  $$\hat{\beta}_{Ridge} = \arg\min_{\beta} \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 = (X^T X + \alpha I)^{-1} X^T y$$
  Shrinks collinear coefficients smoothly toward zero without eliminating them, stabilizing predictive variance.
- **Lasso Regression (L1 Sparsity)**:
  $$\hat{\beta}_{Lasso} = \arg\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1$$
  Drives uninformative or redundant parameters to exactly zero, performing continuous feature selection.
- **Multicollinearity & VIF Analysis**:
  $$VIF_j = \frac{1}{1 - R_j^2}$$
  In geospatial census data, latitude and longitude exhibit high correlation ($r \approx -0.92$). Rather than deleting geographical location, Ridge regularization stabilizes coefficient variance while preserving spatial prediction power.

### Task 3: Customer Churn with Class Imbalance
- **Imbalance Mitigation via Loss Reweighting**:
  The empirical churn rate is ~26.5% positive vs 73.5% negative ($N_{neg}/N_{pos} \approx 2.77$).
  - Random Forest: `class_weight='balanced'` assigns sample weight $w_i = \frac{N}{2 \cdot N_{y_i}}$.
  - XGBoost: `scale_pos_weight = 2.77` scales minority gradient updates.
- **Evaluation Metric Selection**:
  Accuracy is misleading for imbalanced data (predicting all zeros achieves 73.5% accuracy). Models are selected and evaluated by **ROC-AUC (Area Under Receiver Operating Characteristic Curve)**, assessing discrimination across all operating thresholds.
- **Asymmetric Cost Decision Thresholding**:
  Standard $0.50$ thresholding balances false positives and false negatives equally. Because customer defection loss ($500-$1000) vastly exceeds proactive incentive costs ($15), lowering the operating threshold to $0.35$ raises churn recall from 77% to nearly 90%.

### Task 4: NLP Sentiment Classification via TF-IDF
- **TF-IDF Weighting**:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **Stopwords Rationale**:
  Generic stopword removal removes critical sentiment modifiers ("not good", "hardly works", "never buy"). We utilize bigram features (`ngram_range=(1,2)`) to retain localized context while preserving negation semantics.
- **Probability Calibration**:
  Naive Bayes assumes token independence:
  $$P(C \mid X) \propto P(C) \prod_{i=1}^n P(x_i \mid C)$$
  Because tokens in natural language are correlated, Naive Bayes yields probabilities that bunch close to 0 and 1. We include explicit calibration caveats in the inference engine.
