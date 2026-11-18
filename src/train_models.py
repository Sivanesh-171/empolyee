"""
Train and Evaluate 5 Supervised Machine Learning Models for Adult Income Classification.
Models:
1. Logistic Regression
2. Decision Tree (max_depth constrained)
3. K-Nearest Neighbors
4. Support Vector Machine (RBF)
5. Gaussian Naive Bayes

All models are trained on the exact same leak-free preprocessed dataset split.
Results (metrics, ROC curves, confusion matrices) are exported to results/ and models/.
"""

import time
import os
import sys

# Ensure src directory is in path whether script is called from root or inside src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
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


def main():
    """Coordinate the full model training, evaluation, ranking, and artifact saving process."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Step 1: Loading raw data and executing leak-free preprocessing...")
    raw_df = load_raw_data("data/adult.csv")
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test, preprocessor, _, _ = split_and_preprocess(cleaned_df)

    # Save fitted preprocessor
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print(f"Preprocessed train matrix: {X_train.shape}, test matrix: {X_test.shape}")

    models = get_models()
    all_metrics = []
    all_cms = {}
    all_rocs = {}

    # File slug mapping for saving trained models
    model_slugs = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Support Vector Machine": "svm",
        "Naive Bayes": "naive_bayes",
    }

    print("\nStep 2: Training and benchmarking all 5 models on the identical split...")
    for name, model in models.items():
        metrics, cm, roc_data, trained_model = train_and_evaluate_model(
            name, model, X_train, y_train, X_test, y_test
        )
        all_metrics.append(metrics)
        all_cms[name] = cm
        all_rocs[name] = roc_data

        # Save individual model with joblib
        slug = model_slugs[name]
        joblib.dump(trained_model, f"models/{slug}.joblib")

    # Step 3: Create DataFrame and rank models by F1 Score (primary) and ROC-AUC (secondary)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(
        by=["F1 Score", "ROC-AUC"], ascending=[False, False]
    ).reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1
    metrics_df.index.name = "Rank"

    # Step 4: Save metrics, confusion matrices, and ROC curves to results/
    metrics_path = "results/metrics.csv"
    metrics_df.to_csv(metrics_path)
    joblib.dump(all_cms, "results/confusion_matrices.joblib")
    joblib.dump(all_rocs, "results/roc_curves.joblib")

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RANKING TABLE (Sorted by F1 Score, then ROC-AUC):")
    print("=" * 80)
    print(metrics_df.to_string())
    print("=" * 80)
    print(f"Results successfully saved to {metrics_path}!")


if __name__ == "__main__":
    main()
