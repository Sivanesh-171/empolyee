"""
Master Machine Learning Workflow Orchestrator for Adult Census Income Classification.
Executes:
1. Leak-Free Preprocessing with Scikit-learn Pipelines and ColumnTransformer.
2. Baseline 5-Fold Stratified Cross-Validation for all 5 models.
3. Hyperparameter Optimization using both GridSearchCV and RandomizedSearchCV.
4. Model Comparison, Winner Selection based strictly on CV F1-Score, and Refitting on full training data.
5. Final Single Evaluation on the completely untouched holdout test dataset.
"""

import os
import sys
import time

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

<<<<<<< HEAD
from preprocess import load_raw_data, clean_data, split_data
from evaluate_baseline import run_baseline_cross_validation
from tune_models import run_grid_and_random_search, select_and_evaluate_final_models
=======
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from preprocess import load_raw_data, clean_data, split_and_preprocess


def get_models():
    """Define dictionary of 5 supervised classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        # scikit-learn 1.9 deprecated SVC(probability=True): fit() emits a FutureWarning
        # and the parameter is removed in 1.11. The recommended replacement performs the
        # same Platt calibration and still exposes predict_proba() for ROC-AUC / the live
        # predictor, so no metric or downstream code has to change.
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=42), ensemble=False
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train a single model, measure wall-clock training time, and compute all evaluation metrics."""
    print(f"--> Training {name}...")
    start_time = time.time()

    # Step 1: Fit model on preprocessed training partition
    model.fit(X_train, y_train)
    fit_time = round(time.time() - start_time, 2)

    # Step 2: Make predictions on train set (for overfitting check) and test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Step 3: Obtain predicted probabilities for class 1 (>50K) for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = model.decision_function(X_test)

    # Step 4: Calculate evaluation metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Step 5: Compute confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    # Step 6: Compute ROC curve coordinates
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    metrics = {
        "Model": name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Training Time (s)": fit_time,
    }

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": round(roc_auc, 4),
    }

    print(
        f"    Done in {fit_time}s | Test Acc: {test_acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
    )
    return metrics, cm, roc_data, model
>>>>>>> f0da5b80b83eb98a774dbf2614854809d53e89b2


def main():
    print("=" * 80)
    print("ADULT CENSUS INCOME PREDICTION: COMPLETE ML WORKFLOW")
    print("=" * 80)
    start_total_time = time.time()

    # Step 1: Load and clean raw dataset
    print("\n[PHASE 1] Loading raw census dataset and executing data hygiene...")
    raw_df = load_raw_data("data/adult.csv")
    cleaned_df = clean_data(raw_df)

    # Step 2: Perform strict 80:20 stratified train-test split
    # Test dataset is vaulted and untouched during all training and selection phases
    X_train, X_test, y_train, y_test = split_data(cleaned_df, test_size=0.20, random_state=42)
    print(f"--> Training partition: {X_train.shape[0]} records | Holdout test partition: {X_test.shape[0]} records (UNTOUCHED)")

    # Step 3: Run Baseline 5-Fold Stratified Cross-Validation
    print("\n[PHASE 2] Executing Baseline 5-Fold Stratified Cross-Validation on training folds...")
    baseline_df, _ = run_baseline_cross_validation(X_train, y_train)

    # Step 4: Run Hyperparameter Optimization (GridSearchCV & RandomizedSearchCV)
    print("\n[PHASE 3] Executing Hyperparameter Optimization (GridSearchCV & RandomizedSearchCV)...")
    comp_df, best_grid_estimators, best_random_estimators = run_grid_and_random_search(X_train, y_train)

    # Step 5: Select winner strictly by CV F1-Score, refit, and evaluate once on untouched test set
    print("\n[PHASE 4] Selecting best model via CV F1-score and evaluating holdout test set...")
    final_metrics_df = select_and_evaluate_final_models(
        comp_df, best_grid_estimators, best_random_estimators, X_train, y_train, X_test, y_test
    )

    total_time = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 80)
    print(f"COMPLETE WORKFLOW FINISHED IN {total_time}s!")
    print("Saved Artifacts:")
    print("  - models/best_pipeline.joblib")
    print("  - models/*.joblib (all 5 tuned pipelines)")
    print("  - results/baseline_cv_metrics.csv")
    print("  - results/grid_search_results.csv")
    print("  - results/random_search_results.csv")
    print("  - results/tuned_model_comparison.csv")
    print("  - results/final_test_metrics.csv")
    print("  - results/metrics.csv")
    print("  - results/*.png (all comparison and evaluation charts)")
    print("=" * 80)


if __name__ == "__main__":
    main()
