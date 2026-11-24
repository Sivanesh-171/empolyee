"""
Baseline Model Evaluation using 5-Fold Stratified Cross-Validation.
All 5 models are evaluated inside leak-free Scikit-learn Pipelines.
The test partition remains completely untouched.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

from preprocess import load_raw_data, clean_data, split_data, create_pipeline


def get_baseline_models():
    """Define dictionary of 5 baseline classifiers with preserved default configurations."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=42), ensemble=False
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }


def run_baseline_cross_validation(X_train, y_train):
    """
    Evaluate all 5 baseline models using 5-fold Stratified Cross-Validation.
    Each model is embedded inside a complete pipeline with preprocessing,
    guaranteeing that scalers and encoders are fitted strictly on the 4 training folds.
    """
    os.makedirs("results", exist_ok=True)
    models = get_baseline_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    baseline_records = []
    detailed_fold_results = {}

    print("=" * 80)
    print("STEP 1: 5-FOLD STRATIFIED CROSS-VALIDATION BASELINE EVALUATION")
    print("=" * 80)

    for name, clf in models.items():
        print(f"--> Evaluating Baseline Pipeline: {name} (5 folds in parallel)...")
        pipeline = create_pipeline(clf)

        start_time = time.time()
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )
        elapsed_time = round(time.time() - start_time, 2)

        mean_acc = np.mean(cv_res["test_accuracy"])
        mean_prec = np.mean(cv_res["test_precision"])
        mean_rec = np.mean(cv_res["test_recall"])
        mean_f1 = np.mean(cv_res["test_f1"])
        std_f1 = np.std(cv_res["test_f1"])
        mean_auc = np.mean(cv_res["test_roc_auc"])
        mean_train_acc = np.mean(cv_res["train_accuracy"])

        record = {
            "Model": name,
            "Mean CV Accuracy": round(mean_acc, 4),
            "Mean CV Precision": round(mean_prec, 4),
            "Mean CV Recall": round(mean_rec, 4),
            "Mean CV F1-Score": round(mean_f1, 4),
            "Std CV F1-Score": round(std_f1, 4),
            "Mean CV ROC-AUC": round(mean_auc, 4),
            "Mean Train Accuracy": round(mean_train_acc, 4),
            "CV Execution Time (s)": elapsed_time,
        }
        baseline_records.append(record)
        detailed_fold_results[name] = cv_res

        print(
            f"    Done in {elapsed_time}s | Mean F1: {mean_f1:.4f} (+/- {std_f1:.4f}) | "
            f"Mean ROC-AUC: {mean_auc:.4f} | Mean Acc: {mean_acc:.4f}"
        )

    # Convert to DataFrame and sort by Mean CV F1-Score descending
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df = baseline_df.sort_values(
        by=["Mean CV F1-Score", "Mean CV ROC-AUC"], ascending=[False, False]
    ).reset_index(drop=True)
    baseline_df.index = baseline_df.index + 1
    baseline_df.index.name = "Baseline Rank"

    # Save baseline report
    baseline_csv_path = "results/baseline_cv_metrics.csv"
    baseline_df.to_csv(baseline_csv_path)
    joblib.dump(detailed_fold_results, "results/baseline_detailed_folds.joblib")

    print("\n" + "=" * 80)
    print("BASELINE 5-FOLD CROSS-VALIDATION SUMMARY REPORT (Sorted by Mean F1):")
    print("=" * 80)
    print(baseline_df.to_string())
    print("=" * 80)
    print(f"Baseline report successfully saved to {baseline_csv_path}\n")

    # Generate Baseline Visualizations
    generate_baseline_visualizations(baseline_df, detailed_fold_results)

    return baseline_df, detailed_fold_results


def generate_baseline_visualizations(baseline_df, detailed_fold_results):
    """Generate static comparison plots for baseline cross-validation performance."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Baseline F1-score comparison with standard deviation error bars
    plt.figure(figsize=(10, 5))
    models = baseline_df["Model"].tolist()
    means = baseline_df["Mean CV F1-Score"].tolist()
    stds = baseline_df["Std CV F1-Score"].tolist()
    bars = plt.bar(models, means, yerr=stds, capsize=6, color="#2563eb", alpha=0.85, edgecolor="#1e3a8a")
    plt.title("5-Fold Cross-Validation Baseline F1-Score Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Mean F1-Score (Positive Class: >50K)", fontsize=11)
    plt.ylim(0, 0.8)
    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{mean:.4f}\n±{std:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/baseline_f1_comparison.png", dpi=200)
    plt.close()

    # 2. Comprehensive Metric Comparison Grouped Bar Chart
    plt.figure(figsize=(12, 6))
    plot_df = baseline_df.set_index("Model")[["Mean CV Accuracy", "Mean CV Precision", "Mean CV Recall", "Mean CV F1-Score", "Mean CV ROC-AUC"]]
    plot_df.columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df.plot(kind="bar", figsize=(12, 6), colormap="Set2", edgecolor="black", width=0.8)
    plt.title("Baseline 5-Fold Cross-Validation Metrics by Algorithm", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/baseline_metrics_comparison.png", dpi=200)
    plt.close()

    print("Baseline visualization charts saved to results/baseline_f1_comparison.png and results/baseline_metrics_comparison.png")


if __name__ == "__main__":
    raw = load_raw_data("data/adult.csv")
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned, test_size=0.20, random_state=42)
    print(f"Training dataset size: {X_train.shape[0]} rows | Holdout test size: {X_test.shape[0]} rows (UNTOUCHED)")
    run_baseline_cross_validation(X_train, y_train)
