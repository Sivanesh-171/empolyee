# Adult Income Classification (Census Machine Learning Benchmark)

An end-to-end, reproducible Machine Learning project and interactive Streamlit web application that predicts whether an individual earns more than $50,000 annually based on the 1994 US Census Adult dataset.

Built with simple, modular Python code, leak-free feature engineering, and a fair benchmark of 5 supervised classification algorithms.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Architecture](#project-architecture)
- [How to Setup and Run](#how-to-setup-and-run)
- [Key EDA Findings](#key-eda-findings)
- [Preprocessing Pipeline Decisions](#preprocessing-pipeline-decisions)
- [Authentic Experimental Benchmark Results](#authentic-experimental-benchmark-results)
- [In-Depth Written Analysis](#in-depth-written-analysis)
  - [Why the Best Model Won (Decision Tree)](#why-the-best-model-won-decision-tree)
  - [Why the Worst Model Struggled (Naive Bayes)](#why-the-worst-model-struggled-naive-bayes)
  - [Analysis of Intermediate Models (SVM, Logistic Regression, KNN)](#analysis-of-intermediate-models)
  - [Algorithm Selection Guide](#algorithm-selection-guide)
- [Limitations & Next Steps](#limitations--next-steps)

---

## 🌟 Project Overview
This repository provides a complete machine learning lifecycle solution for binary income classification:
- **Full Exploratory Data Analysis (EDA)**: 9 structured visual sections with explicit analytical insights.
- **Unified Leak-Free Preprocessing**: Scalers and encoders fitted exclusively on the 80% training split.
- **Rigorous 5-Model Comparison**: Evaluated on identical test sets across Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Training Time.
- **Interactive Streamlit Web Dashboard**: Responsive UI across desktop and mobile devices featuring an interactive EDA explorer, live demographic predictor, and model comparison cockpit.

---

## 📊 Dataset Description
The project utilizes the classic **Adult Census Income** dataset (1994 United States Census):
- **Raw Observations**: 32,561 rows, 15 columns.
- **Target Variable**: `income` (`<=50K` vs `>50K`).
- **Class Imbalance**:
  - `<=50K`: 24,720 individuals (**75.92%**)
  - `>50K`: 7,841 individuals (**24.08%**)
  - Ratio: Approximately 3.15 to 1.
- **Data Hygiene Attributes**:
  - Missing values represented as string `'?'` in `workclass` (1,836), `occupation` (1,843), and `native.country` (583).
  - 24 identical duplicate rows.
  - Column headers containing periods (`education.num`, `marital.status`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`).
  - Redundant column: `education` (16 categories) maps monotonically 1-to-1 with `education.num` (integers 1–16).
  - Uninformative feature: `fnlwgt` is an administrative demographic expansion weight, not an individual human trait.

---

## 📂 Project Architecture
```text
adult-income-ml/
├── data/
│   └── adult.csv                      # Extracted dataset (32,561 records)
├── src/
│   ├── preprocess.py                  # Cleaning, feature engineering, leak-free pipeline
│   └── train_models.py                # 5-model training, benchmarking, and artifact export
├── models/
│   ├── decision_tree.joblib           # Saved Decision Tree model (Rank 1)
│   ├── svm.joblib                     # Saved Support Vector Machine model (Rank 2)
│   ├── logistic_regression.joblib     # Saved Logistic Regression model (Rank 3)
│   ├── knn.joblib                     # Saved K-Nearest Neighbors model (Rank 4)
│   ├── naive_bayes.joblib             # Saved Gaussian Naive Bayes model (Rank 5)
│   └── preprocessor.joblib            # Fitted ColumnTransformer bundle (scaler + encoder)
├── results/
│   ├── metrics.csv                    # Authentic benchmark metrics table
│   ├── confusion_matrices.joblib      # Confusion matrices for all 5 models
│   └── roc_curves.joblib              # Test ROC curves (FPR, TPR, AUC)
├── .streamlit/
│   └── config.toml                    # Theme and responsiveness settings
├── app.py                             # Interactive Streamlit multi-page web application
├── requirements.txt                   # Dependency manifest
└── README.md                          # Detailed documentation and empirical analysis
```

---

## 🚀 How to Setup and Run

### 1. Clone or Open Workspace
```bash
cd adult-income-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing & Model Training (Runs Once, Saves Artifacts)
```bash
python src/train_models.py
```
*This executes leak-free preprocessing, trains all 5 supervised models on the identical split, and exports `results/metrics.csv` along with model binaries.*

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. The app loads precomputed artifacts instantly without retraining.

---

## 🔍 Key EDA Findings
1. **The Accuracy Paradox**: A naive dummy model predicting `<=50K` for everyone would achieve **75.92% accuracy** while capturing zero high-earning individuals (Recall = 0.0, F1 = 0.0). Model ranking must therefore rely on **F1-Score** and **ROC-AUC**.
2. **Missingness is Categorical**: Missing values (`'?'`) appear exclusively in `workclass` (5.64%), `occupation` (5.66%), and `native.country` (1.79%). Imputing these as a distinct category (`'Unknown'`) retains all samples without fabricating data.
3. **Severe Skewness & Zero-Inflation**: `capital.gain` (91.7% zeros, max $99,999) and `capital.loss` (95.3% zeros) exhibit extreme positive skewness, demanding `log1p` transformation.
4. **1-to-1 Redundancy**: `education` and `education.num` possess a strict 1-to-1 mapping (e.g., Preschool = 1, HS-grad = 9, Bachelors = 13, Doctorate = 16). Dropping the string column eliminates perfect collinearity.
5. **High Mutual Information**: `marital.status` and `relationship` overlap heavily (e.g., `Married-civ-spouse` coincides with `Husband` or `Wife` in 99.8% of cases), violating independence assumptions in linear/Bayesian models.
6. **Key Demographic Drivers**: Individuals earning `>50K` are systematically older (median 44 vs 34), hold higher education levels (Prof-school & Doctorate exceed 70% high-income rate), and work 40–50+ hours weekly.

---

## 🛠️ Preprocessing Pipeline Decisions
To ensure absolute fairness and zero data leakage:
1. **Deduplication**: 24 duplicate records dropped, yielding 32,537 unique records.
2. **Feature Pruning**: `fnlwgt` (census administrative weight) and `education` (string duplicate of `education_num`) dropped.
3. **Country Binning**: Over 89.5% of samples originate from the US. `native_country` was binned into `'United-States'` and `'Other'` to prevent extreme one-hot column explosion on single-instance foreign countries.
4. **Logarithmic Scaling**: Applied $x' = \ln(1 + x)$ (`np.log1p`) to `capital_gain` and `capital_loss`.
5. **Stratified Partitioning**: 80% train (26,029 samples) and 20% test (6,508 samples) with `random_state=42` and `stratify=y`.
6. **Leak-Free Transformation**:
   - `StandardScaler` fitted **only on the training split** for numeric features (`age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`).
   - `OneHotEncoder(handle_unknown='ignore')` fitted **only on the training split** for categorical features.
   - Resulting in a unified feature matrix of **51 columns** applied identically across all 5 models.
   *Note: Feature scaling is mathematically required for distance and gradient-based algorithms (KNN, SVM, Logistic Regression) and completely harmless for axis-aligned split algorithms (Decision Trees).*

---

## 📈 Authentic Experimental Benchmark Results

The table below reports the **real, authentic results** produced by `src/train_models.py` on the exact 20% test partition (6,508 samples):

| Rank | Model | Train Accuracy | Test Accuracy | Precision (>50K) | Recall (>50K) | F1 Score (>50K) | ROC-AUC | Training Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Decision Tree** | **0.8617** | **0.8545** | **0.7656** | 0.5708 | **0.6540** | **0.8983** | **0.05s** |
| **2** | **Support Vector Machine (RBF)** | 0.8651 | 0.8506 | 0.7427 | 0.5816 | 0.6524 | 0.8892 | 65.74s |
| **3** | **Logistic Regression** | 0.8448 | 0.8456 | 0.7180 | 0.5912 | 0.6485 | 0.8957 | 0.08s |
| **4** | **K-Nearest Neighbors** | 0.8766 | 0.8317 | 0.6623 | 0.6154 | 0.6380 | 0.8661 | 0.00s* |
| **5** | **Gaussian Naive Bayes** | 0.5880 | 0.5853 | 0.3610 | **0.9362** | 0.5210 | 0.8415 | 0.01s |

*\*Note: KNN is a lazy learner ($O(1)$ fit time), but incurs significant test-time inference latency ($O(N \cdot D)$).*

---

## 🧠 In-Depth Written Analysis

### Why the Best Model Won: Decision Tree (Rank 1)
1. **Non-Linear Step-Function Thresholds**: Human socioeconomic dynamics do not scale linearly. Earning potential jumps abruptly at specific critical thresholds (e.g., reaching 13 years of education for a Bachelor's degree, or crossing age 30 and 40 weekly work hours). Axis-aligned recursive partitioning in a Decision Tree mirrors these thresholds naturally without requiring artificial polynomial feature generation.
2. **Robustness to High-Dimensional One-Hot Sparsity**: With 51 features created after one-hot encoding, tree algorithms isolate primary discriminators (such as `marital_status_Married-civ-spouse` and `education_num`) at high-level splits, while ignoring uninformative sparse columns.
3. **Controlled Generalization**: By setting `max_depth=8` and `min_samples_leaf=20`, the Decision Tree avoided memorizing the training noise, achieving near-identical **Train Accuracy (86.17%) and Test Accuracy (85.45%)**. Furthermore, it trained in only **0.05 seconds**, making it over **1,300× faster** than the RBF SVM.

### Why the Worst Model Struggled: Naive Bayes (Rank 5)
1. **Severe Violation of the Conditional Independence Assumption**: Naive Bayes operates on the foundational assumption that all features are mutually independent given the class:
   $$P(X_1, X_2, \dots, X_p \mid Y) = \prod_{j=1}^p P(X_j \mid Y)$$
   In the adult census dataset, this assumption is radically violated. `marital_status_Married-civ-spouse` and `relationship_Husband` convey virtually the same demographic information.
2. **Probability Inflation and Precision Collapse**: When multiple correlated features point toward the high-income class, Naive Bayes multiplies their probabilities, dramatically inflating posterior odds. As proven by the empirical results, Naive Bayes predicted `>50K` indiscriminately:
   - While Recall was high (**93.62%**), Precision collapsed to a dismal **36.10%**.
   - Overall Test Accuracy plunged to **58.53%** (worse than random majority-class guessing).

### Analysis of Intermediate Models

#### Support Vector Machine (RBF Kernel) (Rank 2)
- Achieved **85.06% Test Accuracy, 0.6524 F1, and 0.8892 ROC-AUC**, closely trailing the Decision Tree.
- Projecting the 51 features into an infinite-dimensional Hilbert space allows the RBF kernel to form smooth, non-linear margin boundaries.
- **The Bottleneck**: Required **65.74 seconds** to train. Because kernel SVM computation scales between $O(N^2)$ and $O(N^3)$, training time is 1,300× longer than Decision Trees, rendering it less practical for larger datasets.

#### Logistic Regression (Rank 3)
- Achieved **84.56% Test Accuracy, 0.6485 F1, and 0.8957 ROC-AUC**, training in just **0.08 seconds**.
- Demonstrates that linear models with standardized features and log-transformed capital gains provide a remarkably competitive, fast, and stable baseline.
- Its main limitation is the inability to detect complex interactive relationships (e.g. sex $\times$ marital status $\times$ education) without manual interaction feature engineering.

#### K-Nearest Neighbors (Rank 4)
- Achieved **83.17% Test Accuracy and 0.6380 F1-Score**.
- Showed the highest degree of overfitting among all competitive models (**Train Accuracy: 87.66% vs. Test Accuracy: 83.17%**).
- Suffers from the **Curse of Dimensionality**: in a 51-dimensional one-hot space, pairwise Euclidean distances become uniform, degrading the nearest-neighbor discriminatory signal.

---

### 🗺️ Algorithm Selection Guide
| Algorithm | Key Strengths | Core Vulnerabilities | Best Applied To |
| :--- | :--- | :--- | :--- |
| **Decision Trees / Ensembles** | Captures non-linear thresholds, invariant to scale, lightning-fast inference | Prone to deep tree overfitting if unpruned | Tabular business data, risk scoring, demographic classification |
| **Support Vector Machine (RBF)** | Strong non-linear margin separation in complex feature spaces | $O(N^2 \text{ to } N^3)$ time complexity, poor scaling on large datasets | Small-to-medium clean datasets with complex boundaries |
| **Logistic Regression** | Fast training, transparent coefficient interpretation, calibrated probabilities | Strictly linear decision surface | High-throughput production services, credit risk, regulated environments |
| **K-Nearest Neighbors** | Non-parametric, zero training phase | Distance metric degrades in high dimensions ($D \ge 50$), high test latency | Low-dimensional continuous spatial/metric data |
| **Naive Bayes** | Ultra-fast, simple parameter estimation | Completely collapses when features are correlated or collinear | Text analysis, spam detection, independent sparse bag-of-words |

---

## 🔬 Limitations & Next Steps
1. **Hyperparameter Optimization**: Model hyperparameters in this benchmark were fixed to transparent baseline values (`max_depth=8`, `n_neighbors=7`). Tuning via Bayesian Optimization (`Optuna`) could further refine F1 scores.
2. **Cost-Sensitive Learning**: Implementing `class_weight='balanced'` in Logistic Regression and Decision Trees to explicitly penalize false negatives on the 24% minority class.
3. **Advanced Ensemble Benchmarks**: Evaluating modern gradient boosting frameworks (XGBoost, LightGBM, CatBoost), which typically represent the industry standard for tabular data.
4. **Historical Temporal Shift**: The dataset reflects 1994 census wages; models trained on this data should not be utilized for contemporary automated wage decisions without recalibration.

---

## 📜 License & Acknowledgments
Dataset source: Barry Becker & Ronny Kohavi, *Adult Census Income*, 1994 US Census Bureau, hosted by UC Irvine Machine Learning Repository.
Developed for educational and machine learning benchmarking purposes.

