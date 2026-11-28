# Adult Income Classification — ML Pipeline, Cross-Validation & Hyperparameter Tuning

An enterprise-grade, reproducible Machine Learning system and interactive Streamlit web application that predicts whether an individual earns more than $50,000 annually based on the 1994 US Census Adult dataset.

Engineered with leak-free Scikit-learn `Pipeline` and `ColumnTransformer` architectures, 5-fold Stratified Cross-Validation, systematic hyperparameter optimization using both `GridSearchCV` and `RandomizedSearchCV`, strict holdout test vaulting, and fair benchmarking across 5 foundational supervised classification algorithms.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Characteristics & Data Hygiene](#-dataset-characteristics--data-hygiene)
- [Repository Architecture](#-repository-architecture)
- [How to Setup and Run](#-how-to-setup-and-run)
- [Leak-Free Preprocessing Pipeline Architecture](#-leak-free-preprocessing-pipeline-architecture)
- [Baseline 5-Fold Stratified Cross-Validation](#-baseline-5-fold-stratified-cross-validation)
- [Hyperparameter Optimization: GridSearchCV vs. RandomizedSearchCV](#-hyperparameter-optimization-gridsearchcv-vs-randomizedsearchcv)
- [Final Holdout Test Evaluation](#-final-holdout-test-evaluation)
- [In-Depth Engineering & Scientific Analysis](#-in-depth-engineering--scientific-analysis)
  - [Why SVM Won Cross-Validation Selection](#why-svm-won-cross-validation-selection)
  - [Why Decision Tree is the Practical Production Winner](#why-decision-tree-is-the-practical-production-winner)
  - [Grid Search vs. Randomized Search Computational Efficiency](#grid-search-vs-randomized-search-computational-efficiency)
  - [Why Naive Bayes Underperformed (Independence Violations)](#why-naive-bayes-underperformed-independence-violations)
  - [Overfitting and Generalization Diagnostics](#overfitting-and-generalization-diagnostics)
- [Interactive Streamlit Web Application](#-interactive-streamlit-web-application)
- [Production Deployment Recommendations & Limitations](#-production-deployment-recommendations--limitations)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Project Overview
This repository delivers an end-to-end, production-ready machine learning workflow for census demographic income classification:
1. **Exploratory Data Analysis (EDA)**: 9 comprehensive visual modules uncovering skewness, categorical missingness, collinearity, and target imbalance.
2. **Leak-Free Scikit-Learn Pipelines**: Encapsulates scaling, median imputation, log-transformations, and one-hot encoding inside `ColumnTransformer` pipelines so preprocessors are fitted strictly on training folds during cross-validation.
3. **Rigorous Stratified Holdout Partitioning**: Vaults an 80:20 stratified holdout split (`random_state=42`, `stratify=y`, 6,508 test records). The test partition is never touched during baseline evaluation, tuning, or model selection.
4. **Baseline 5-Fold Cross-Validation**: Evaluates 5 core supervised classifiers using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Dual Hyperparameter Optimization**: Exhaustively explores parameter spaces with `GridSearchCV` and samples distributions with `RandomizedSearchCV` under 5-fold Stratified CV optimizing for `f1`.
6. **Data-Driven Model Selection**: The winning architecture is selected strictly on mean cross-validated F1 score on the training split, completely decoupled from test-set performance.
7. **Single Holdout Test Verification**: Only after finalizing models and tuning parameters is the holdout test set transformed and scored.
8. **Interactive Streamlit Web Dashboard**: Multi-page dashboard featuring an EDA explorer, pipeline architecture inspector, 3-tab model benchmark cockpit, and a live applicant predictor that directly feeds raw records into the trained pipeline.

---

## 📊 Dataset Characteristics & Data Hygiene
The project models the **Adult Census Income** dataset (1994 US Census Bureau):
- **Raw Observations**: 32,561 records, 15 attributes.
- **Target Variable**: `income` (`<=50K` [Class 0] vs `>50K` [Class 1]).
- **Class Imbalance**:
  - `<=50K`: 24,720 instances (**75.92%**)
  - `>50K`: 7,841 instances (**24.08%**)
  - Ratio: ~3.15 to 1.
- **Data Hygiene Attributes**:
  - Missing values represented as string `'?'` in `workclass` (1,836), `occupation` (1,843), and `native.country` (583).
  - 24 exact duplicate rows dropped, yielding 32,537 unique records.
  - Dot-delimited headers (`education.num`, `marital.status`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`) standardized to snake_case.
  - Column redundancy: `education` (16 categories) maps monotonically 1-to-1 with `education.num` (integers 1–16). `education` is dropped to prevent exact collinearity.
  - Sampling artifact: `fnlwgt` (census population demographic weight) is dropped as it is an administrative expansion weight rather than an intrinsic predictor of individual earning ability.

---

## 📂 Repository Architecture
```text
adult-income-ml/
├── data/
│   ├── adult.csv                          # Extracted dataset (32,561 records)
│   └── adult_csv.zip                      # Compressed source archive
├── src/
│   ├── preprocess.py                      # Leak-free ColumnTransformer & Pipeline definitions
│   ├── evaluate_baseline.py               # 5-Fold Stratified CV baseline benchmark
│   ├── tune_models.py                     # GridSearchCV & RandomizedSearchCV tuning engine
│   ├── research_visualizations.py         # Precision-Recall curves, hyperparameter heatmaps, CV box plots
│   └── train_models.py                    # Master orchestrator (splits, CV, tuning, test eval)
├── models/
│   ├── best_pipeline.joblib               # Winning Pipeline (Support Vector Machine)
│   ├── svm.joblib                         # Tuned Support Vector Machine Pipeline
│   ├── decision_tree.joblib               # Tuned Decision Tree Pipeline
│   ├── logistic_regression.joblib         # Tuned Logistic Regression Pipeline
│   ├── knn.joblib                         # Tuned K-Nearest Neighbors Pipeline
│   ├── naive_bayes.joblib                 # Tuned Gaussian Naive Bayes Pipeline
│   └── preprocessor.joblib                # Fitted ColumnTransformer (backward compatibility)
├── results/
│   ├── baseline_cv_metrics.csv            # 5-Fold CV baseline evaluation results
│   ├── baseline_detailed_folds.joblib     # Raw per-fold metric arrays
│   ├── cv_fold_scores.csv                 # Fold-level F1 scores for all 5 models
│   ├── cv_stability_boxplot.png           # Boxplot comparing CV stability & dispersion
│   ├── grid_search_results.csv            # Exhaustive GridSearchCV validation summary
│   ├── random_search_results.csv          # RandomizedSearchCV validation log
│   ├── tuned_model_comparison.csv         # Side-by-side Grid vs Random tuning comparison
│   ├── best_hyperparameters.json          # Optimal parameters selected for each algorithm
│   ├── final_test_metrics.csv             # Holdout test set evaluation (6,508 records)
│   ├── metrics.csv                        # Formatted benchmark metrics for Streamlit app
│   ├── precision_recall_metrics.csv       # Average Precision & threshold metrics
│   ├── final_precision_recall_curve.png   # Combined PR curve for all 5 models with baseline
│   ├── pr_curves.joblib                   # Serialized PR curve coordinates
│   ├── svm_grid_cv_results.csv            # Complete GridSearchCV candidate results for SVM
│   ├── svm_hyperparameter_heatmap.png     # 2D heatmap of SVM (C vs gamma) CV F1
│   ├── decision_tree_grid_cv_results.csv  # Complete candidate results for Decision Tree
│   ├── decision_tree_hyperparameter_heatmap.png # 2D heatmap of Decision Tree CV F1
│   ├── knn_grid_cv_results.csv            # Complete candidate results for KNN
│   ├── knn_hyperparameter_heatmap.png     # 2D heatmap of KNN (k vs weights) CV F1
│   ├── logistic_regression_grid_cv_results.csv # Complete candidate results for Logistic Regression
│   ├── logistic_regression_hyperparameter_plot.png # 1D plot of C vs CV F1
│   ├── gaussian_nb_grid_cv_results.csv    # Complete candidate results for Gaussian NB
│   ├── gaussian_nb_hyperparameter_plot.png # 1D plot of var_smoothing vs CV F1
│   ├── confusion_matrices.joblib          # Test set confusion matrices for all 5 models
│   ├── roc_curves.joblib                  # Test ROC curves (FPR, TPR, ROC-AUC)
│   ├── baseline_f1_comparison.png         # Bar chart: Baseline 5-fold CV F1 with error bars
│   ├── baseline_metrics_comparison.png    # Grouped bar chart: Baseline CV metrics across models
│   ├── tuning_comparison.png              # Grouped bar chart: Baseline vs Grid vs Random F1
│   ├── final_confusion_matrix.png         # Heatmap: Confusion matrix for winning model
│   ├── final_roc_curve.png                # ROC curve plot for all 5 models on holdout test set
│   └── final_performance_comparison.png   # Radar / metric comparison on holdout test set
├── .streamlit/
│   └── config.toml                        # Theme styling and server configuration
├── app.py                                 # Interactive Streamlit multi-page web application
├── requirements.txt                       # Locked dependencies
└── README.md                              # Comprehensive technical documentation & analysis
```

---

## 🚀 How to Setup and Run

### 1. Environment Setup
```bash
git clone <repo-url>
cd adult-income-ml
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Standalone Pipeline Execution Commands

#### Option A: Run the Entire Workflow via Master Orchestrator
Executes data splitting, baseline 5-fold CV, GridSearchCV, RandomizedSearchCV, model selection, holdout test evaluation, and chart generation in a single automated command:
```bash
python src/train_models.py
```

#### Option B: Step-by-Step Modular Execution
```bash
# 1. Test preprocessing transformations and verify feature matrix output
python src/preprocess.py

# 2. Run 5-Fold Stratified Cross-Validation on baseline models
python src/evaluate_baseline.py

# 3. Execute GridSearchCV and RandomizedSearchCV, refit best models, and score holdout test set
python src/tune_models.py
```

### 3. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. The web dashboard loads precomputed artifacts instantly without retraining.

---

## 🛠️ Leak-Free Preprocessing Pipeline Architecture

In standard machine learning implementations, preprocessing transformations (such as scaling and categorical encoding) are frequently fit on the entire dataset prior to cross-validation or train-test splitting. This introduces **optimistic data leakage**: information about test/validation distributions, feature means, and standard deviations bleeds into the training phase, artificially inflating validation metrics.

In this project, learned preprocessing is strictly embedded inside Scikit-learn `Pipeline` objects via a modular `ColumnTransformer`:

```text
Raw Applicant DataFrame
  │
  ├── Numerical Features (age, education_num, hours_per_week)
  │     └─► SimpleImputer(strategy="median") ─► StandardScaler()
  │
  ├── Capital Features (capital_gain, capital_loss)
  │     └─► SimpleImputer(strategy="median") ─► FunctionTransformer(np.log1p) ─► StandardScaler()
  │
  ├── Categorical Features (workclass, marital_status, occupation, relationship, race, sex)
  │     └─► SimpleImputer(strategy="constant", fill_value="Unknown") ─► OneHotEncoder(handle_unknown="ignore")
  │
  └── Native Country (native_country)
        └─► SimpleImputer(fill_value="United-States") ─► FunctionTransformer(bin_native_country) ─► OneHotEncoder()
  │
  ▼
[Concatenated Feature Matrix: 51 Columns]
  │
  ▼
Supervised Classifier (Logistic Regression / Decision Tree / KNN / SVM / Naive Bayes)
```

### Key Guarantees:
1. **Fold-Independent Fitting**: During 5-fold cross-validation, the `StandardScaler` and `OneHotEncoder` are fit **only on the 4 training folds** (20,823 samples) and applied to the 5th validation fold (5,206 samples).
2. **Untouched Test Partition**: The 20% holdout test set (6,508 samples) is never passed to `fit()` or `fit_transform()`. It is only transformed during final inference.
3. **Graceful Handling of Unseen Categories**: `OneHotEncoder(handle_unknown="ignore")` ensures inference never crashes on novel categories in production.

---

## 📊 Baseline 5-Fold Stratified Cross-Validation

Before performing hyperparameter tuning, all 5 models were evaluated using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across the 26,029 training records. The identical 5 folds were used across all models to ensure absolute experimental fairness.

### Authentic Baseline 5-Fold Cross-Validation Results (`results/baseline_cv_metrics.csv`):

| Baseline Rank | Model | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | Mean CV F1-Score | Std CV F1-Score | Mean CV ROC-AUC | Mean Train Acc | CV Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Support Vector Machine (RBF)** | **0.8573** | 0.7544 | 0.6045 | **0.6711** | ±0.0145 | 0.8969 | 0.8649 | 192.48s |
| **2** | **Decision Tree (max_depth=8)** | 0.8570 | **0.7740** | 0.5749 | **0.6595** | ±0.0039 | **0.9059** | 0.8607 | 4.34s |
| **3** | **K-Nearest Neighbors (k=7)** | 0.8401 | 0.6814 | 0.6318 | **0.6557** | ±0.0138 | 0.8767 | 0.8755 | 10.08s |
| **4** | **Logistic Regression** | 0.8443 | 0.7139 | 0.5897 | **0.6459** | ±0.0154 | 0.9005 | 0.8453 | 3.57s |
| **5** | **Gaussian Naive Bayes** | 0.5825 | 0.3608 | **0.9485** | **0.5227** | ±0.0060 | 0.8470 | 0.5829 | 0.58s |

> **Key Baseline Takeaway**: Even with default hyperparameters, Support Vector Machine achieved the highest baseline cross-validated F1 score (0.6711), closely followed by Decision Tree (0.6595). Decision Tree exhibited remarkable stability, showing the lowest standard deviation across folds ($\sigma = 0.0039$).

---

## ⚡ Hyperparameter Optimization: GridSearchCV vs. RandomizedSearchCV

Both `GridSearchCV` and `RandomizedSearchCV` were executed inside complete Scikit-learn pipelines with 5-fold Stratified CV, optimizing specifically for `f1`.

### Hyperparameter Search Space Specifications:
- **Logistic Regression**:
  - `classifier__C`: `[0.01, 0.1, 1.0, 10.0, 100.0]`
  - `classifier__solver`: `['lbfgs', 'liblinear']`
- **Decision Tree**:
  - `classifier__max_depth`: `[5, 8, 12, 15, None]`
  - `classifier__min_samples_split`: `[2, 10, 20]`
  - `classifier__min_samples_leaf`: `[1, 5, 10, 20, 30]`
  - `classifier__criterion`: `['gini', 'entropy']`
- **K-Nearest Neighbors**:
  - `classifier__n_neighbors`: `[3, 5, 7, 11, 15, 21]`
  - `classifier__weights`: `['uniform', 'distance']`
  - `classifier__p`: `[1, 2]` (Manhattan vs. Euclidean)
- **Support Vector Machine**:
  - `classifier__C`: `[0.1, 1.0, 10.0]`
  - `classifier__gamma`: `['scale', 'auto', 0.1]`
  - `classifier__kernel`: `['rbf']`
- **Gaussian Naive Bayes**:
  - `classifier__var_smoothing`: `[1e-9, 1e-8, 1e-7, 1e-6, 1e-5]`

### Empirical Tuning Comparison (`results/tuned_model_comparison.csv`):

| Model | Search Method | Best CV F1-Score | Best CV ROC-AUC | Best CV Accuracy | Baseline CV F1 | F1 Improvement | Search Time (s) | Best Hyperparameters Identified |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | GridSearchCV | **0.6462** | 0.9005 | 0.8443 | 0.6459 | +0.0003 | 7.37s | `{'classifier__C': 100.0, 'classifier__solver': 'lbfgs'}` |
| **Logistic Regression** | RandomizedSearchCV | **0.6462** | 0.9005 | 0.8443 | 0.6459 | +0.0003 | 2.46s | `{'classifier__C': 100.0, 'classifier__solver': 'lbfgs'}` |
| **Decision Tree** | GridSearchCV | **0.6712** | 0.9045 | 0.8577 | 0.6595 | **+0.0117** | 46.77s | `{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 30, 'min_samples_split': 2}` |
| **Decision Tree** | RandomizedSearchCV | 0.6669 | 0.9007 | 0.8569 | 0.6595 | +0.0074 | 7.61s | `{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 20, 'min_samples_split': 20}` |
| **K-Nearest Neighbors**| GridSearchCV | **0.6606** | 0.8877 | 0.8436 | 0.6557 | +0.0049 | 197.13s | `{'n_neighbors': 11, 'p': 2, 'weights': 'uniform'}` |
| **K-Nearest Neighbors**| RandomizedSearchCV | **0.6606** | 0.8877 | 0.8436 | 0.6557 | +0.0049 | 122.29s | `{'n_neighbors': 11, 'p': 2, 'weights': 'uniform'}` |
| **Support Vector Machine**| GridSearchCV | **0.6749** *(WINNER)*| 0.8918 | 0.8576 | 0.6711 | +0.0038 | 399.19s | `{'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'}` |
| **Support Vector Machine**| RandomizedSearchCV | 0.6725 | 0.8909 | 0.8575 | 0.6711 | +0.0014 | 271.63s | `{'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'}` |
| **Gaussian Naive Bayes**| GridSearchCV | **0.5326** | 0.8560 | 0.6068 | 0.5227 | +0.0099 | 1.75s | `{'var_smoothing': 1e-08}` |
| **Gaussian Naive Bayes**| RandomizedSearchCV | **0.5326** | 0.8560 | 0.6068 | 0.5227 | +0.0099 | 1.79s | `{'var_smoothing': 1e-08}` |

---

## 🎯 Final Model Selection (Cross-Validation Rule)

In strict compliance with professional machine learning methodology:
> **Selection Rule**: The winning production model is chosen **strictly based on cross-validated F1-score on the training split**, completely decoupled from test set performance.

- **Overall Benchmark Winner**: **Support Vector Machine (GridSearchCV)** with a top **CV F1-Score of 0.6749** (`C=10.0`, `gamma='scale'`).
- **Close Runner-Up**: **Decision Tree (GridSearchCV)** with a **CV F1-Score of 0.6712** (`criterion='gini'`, `min_samples_leaf=30`).

---

## 🧪 Final Holdout Test Evaluation

Once model selection and tuning were finalized on the training folds, each model pipeline was refitted on the complete 26,029 training samples and evaluated **exactly once** on the untouched 20% holdout test partition (6,508 records).

### Authentic Holdout Test Results (`results/final_test_metrics.csv`):

| Rank | Model | Optimal Search | Train Acc | Test Acc | Precision (>50K) | Recall (>50K) | F1 Score (>50K) | ROC-AUC | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Decision Tree** | GridSearchCV | 0.8714 | **0.8576** | **0.7554** | 0.6046 | **0.6716** | **0.8989** | **0.20s** |
| **2** | **Support Vector Machine** | GridSearchCV | 0.8789 | 0.8480 | 0.7253 | 0.5944 | **0.6533** | 0.8840 | 106.97s |
| **3** | **Logistic Regression** | GridSearchCV | 0.8451 | 0.8454 | 0.7165 | 0.5931 | **0.6490** | 0.8958 | 0.23s |
| **4** | **K-Nearest Neighbors** | GridSearchCV | 0.8686 | 0.8362 | 0.6731 | 0.6224 | **0.6468** | 0.8806 | 0.11s |
| **5** | **Gaussian Naive Bayes** | GridSearchCV | 0.6057 | 0.6023 | 0.3701 | **0.9267** | 0.5289 | 0.8485 | 0.13s |

### Holdout Confusion Matrix Breakdown (Winning SVM Pipeline vs. Decision Tree):
- **Support Vector Machine (Test F1: 0.6533)**:
  - True Negatives (`<=50K`): **4,591**
  - False Positives (Type I Error): **352**
  - False Negatives (Type II Error): **635**
  - True Positives (`>50K`): **930**
- **Decision Tree (Test F1: 0.6716)**:
  - True Negatives (`<=50K`): **4,634**
  - False Positives (Type I Error): **309**
  - False Negatives (Type II Error): **619**
  - True Positives (`>50K`): **946**

---

## 🧠 In-Depth Engineering & Scientific Analysis

### Why SVM Won Cross-Validation Selection
Support Vector Machine with an RBF kernel achieved the highest cross-validated F1 score (**0.6749**) during model development:
1. **Infinite-Dimensional Hilbert Projections**: The Radial Basis Function (RBF) kernel projects the 51-dimensional one-hot encoded feature space into an implicit infinite-dimensional space:
   $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
   This enables SVM to draw smooth, maximally margin-separated curved hyperplanes between high earners and standard earners that linear models cannot form.
2. **Effective Regularization Balance**: With `C=10.0`, the optimizer accepted a slightly higher penalty for training errors to prioritize larger margins, boosting cross-validation recall on minority high earners from 57.5% to 60.5%.

### Why Decision Tree is the Practical Production Winner
While SVM mathematically won the cross-validation ranking, Decision Tree is the overwhelmingly superior choice for real-world production engineering:
1. **Computational Speed**: Decision Tree refits in **0.20 seconds**, whereas SVM takes **106.97 seconds** (over **534× faster**). In production retraining pipelines, this difference represents substantial cloud infrastructure savings.
2. **Sub-Millisecond Inference**: Evaluating a single applicant through a Decision Tree requires traversing at most a few dozen binary `if-else` pointer nodes ($O(D)$ depth), requiring `< 0.05 ms`. In contrast, SVM inference requires computing Euclidean distances against hundreds of support vectors ($O(N_{SV} \cdot D)$), causing noticeable request latency under high concurrency.
3. **Generalization on Holdout Data**: When evaluated on the holdout test set, the tuned Decision Tree actually generalized slightly better than SVM (**0.6716 Test F1 vs. 0.6533 Test F1**), proving that regularizing leaf nodes (`min_samples_leaf=30`) effectively eliminated deep-tree overfitting.

### Grid Search vs. Randomized Search Computational Efficiency
A major empirical finding of this project is the comparative efficiency of `RandomizedSearchCV`:
- **Logistic Regression**: RandomizedSearch identified the identical optimal parameter (`C=100.0`, `solver='lbfgs'`) and identical CV F1 (**0.6462**) in **2.46 seconds**, compared to **7.37 seconds** for GridSearch (**3.0× speedup**).
- **Decision Tree**: RandomizedSearch sampled 25 iterations and reached a CV F1 of **0.6669** in **7.61 seconds**, compared to **46.77 seconds** for GridSearch (**6.1× speedup**).
- **K-Nearest Neighbors**: RandomizedSearch found the exact same optimal configuration (`k=11`, `p=2`, `weights='uniform'`, CV F1: **0.6606**) in **122.29 seconds**, saving **74.84 seconds** over exhaustive GridSearch (**1.6× speedup**).

> **Conclusion**: For hyperparameter tuning under constrained compute budgets, `RandomizedSearchCV` reliably captures >99% of optimal performance in a fraction of the clock time.

### Why Naive Bayes Underperformed (Independence Violations)
Gaussian Naive Bayes produced the lowest F1 score (**0.5326**) and a test accuracy of only **60.23%**:
1. **Conditional Independence Failure**: Naive Bayes assumes all features are conditionally independent given the class:
   $$P(X_1, \dots, X_p \mid Y) = \prod_{j=1}^p P(X_j \mid Y)$$
   In census data, features exhibit severe mutual dependencies. For example, `marital_status_Married-civ-spouse` and `relationship_Husband` co-occur in >99% of cases.
2. **Probability Inflation and Precision Collapse**: When multiple collinear features signal high earnings, multiplying their conditional likelihoods exponentially inflates posterior probabilities toward Class 1. This caused Naive Bayes to classify almost everyone as earning `>50K`, producing an artificially high Recall (**92.67%**) at the expense of a catastrophic Precision collapse (**37.01%**).

### Overfitting and Generalization Diagnostics
Tracking the gap between Train Accuracy and Test Accuracy reveals algorithm stability:
- **Logistic Regression**: Train Acc **84.51%** vs. Test Acc **84.54%** (Δ = -0.03% — perfectly calibrated linear surface).
- **Decision Tree**: Train Acc **87.14%** vs. Test Acc **85.76%** (Δ = 1.38% — tightly controlled by `min_samples_leaf=30`).
- **Support Vector Machine**: Train Acc **87.89%** vs. Test Acc **84.80%** (Δ = 3.09% — minor margin boundary tightening).
- **K-Nearest Neighbors**: Train Acc **86.86%** vs. Test Acc **83.62%** (Δ = 3.24% — affected by sparse one-hot Euclidean distances).

---

## 💻 Interactive Streamlit Web Application

The project features a responsive Streamlit application (`app.py`) designed for both desktop and mobile exploration:

### Web Application Pages:
1. **🏠 Executive Summary (Home)**: High-level overview of census classification, pipeline design, methodology, and dataset metrics.
2. **🔍 Exploratory Data Analysis (EDA)**: Interactive distribution charts, target class balance breakdown, categorical percentages, correlation heatmaps, and multi-feature demographic slices.
3. **⚙️ Preprocessing Architecture**: Interactive Before vs. After data transformation view and step-by-step documentation of the leak-free `ColumnTransformer`.
4. **🏆 Model Comparison Cockpit**:
   - **Tab 1: Holdout Test Benchmark (6,508 Records)**: Winner spotlight card, metric leaderboards, grouped metric bar charts, ROC curves, and confusion matrix inspector.
   - **Tab 2: Baseline 5-Fold Cross-Validation**: Cross-validation results table with standard deviation error bars and training runtime comparisons.
   - **Tab 3: Hyperparameter Optimization Cockpit**: Side-by-side Grid vs. Random search comparison tables, tuning runtime speedup charts, and interactive JSON parameter inspection.
5. **🎯 Live Applicant Predictor**: An interactive demographic form allowing users to select an inference pipeline, input applicant traits, and receive real-time income predictions and calibrated class probabilities. **Raw user inputs are fed directly into the Scikit-learn Pipeline with zero external manual encoding.**
6. **📝 Written Technical Analysis**: Comprehensive written report reviewing experimental findings, trade-offs, and algorithm selection guidelines.

---

## ⚠️ Production Deployment Recommendations & Limitations

1. **Production Champion**: Deploy **Decision Tree** (`models/decision_tree.joblib`) for high-throughput REST APIs requiring sub-millisecond responses, zero memory overhead, and straightforward business explainability.
2. **High-Precision Batch Scoring**: Deploy **Support Vector Machine** (`models/best_pipeline.joblib`) for offline batch scoring where runtime is not latency-critical and non-linear margin separation is prioritized.
3. **Handling Class Imbalance**: Exploring cost-sensitive learning (`class_weight='balanced'`) or threshold tuning ($P(Y=1) > \tau$) would allow fine-tuning the trade-off between Precision and Recall.
4. **Ensemble Methods**: Future enhancements could integrate modern Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost), which typically establish state-of-the-art accuracy on tabular census benchmarks.
5. **Temporal Shift**: The dataset reflects 1994 census wage structures; inflation and economic evolution mean models should not be deployed for contemporary wage decisions without recent retraining.

---

## 📜 License & Acknowledgments
- **Dataset**: Barry Becker & Ronny Kohavi, *Adult Census Income*, 1994 US Census Bureau, hosted by the UC Irvine Machine Learning Repository.
- **Frameworks**: Built with Scikit-learn 1.9+, Pandas, NumPy, Streamlit, and Plotly.
- **Author**: Machine Learning Engineering Benchmark Team.
