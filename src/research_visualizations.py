"""
Research Visualizations Module for Adult Census Income Classification.

Implements publication-quality ML research visualizations:
1. Precision-Recall (PR) Curves with Average Precision (AP) and baseline prevalence.
2. Hyperparameter Heatmaps (SVM, Decision Tree, KNN) and 1D parameter curves (Logistic Regression, Gaussian NB).
3. Cross-Validation Stability Box Plots comparing fold-wise F1 distributions.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src directory is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score
from preprocess import load_raw_data, clean_data, split_data


# ---------------------------------------------------------------------------
# Plotting Style & Palette
# ---------------------------------------------------------------------------
def set_research_style():
    """Apply consistent scientific publication styling."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8


MODEL_COLORS = {
    "Decision Tree": "#10b981",          # Emerald Green
    "Support Vector Machine": "#3b82f6",  # Blue
    "Logistic Regression": "#f59e0b",    # Amber
    "K-Nearest Neighbors": "#8b5cf6",    # Purple
    "Naive Bayes": "#ef4444",            # Red
}


# ---------------------------------------------------------------------------
# TASK 1: Precision-Recall Curves
# ---------------------------------------------------------------------------
def generate_precision_recall_curves(
    models_dir="models",
    data_path="data/adult.csv",
    output_png="results/final_precision_recall_curve.png",
    output_csv="results/precision_recall_metrics.csv",
    output_joblib="results/pr_curves.joblib",
):
    """
    Generate combined Precision-Recall curves on the untouched holdout test partition.
    Computes Average Precision (AP), evaluates precision/recall at default threshold (0.5),
    and exports high-resolution figure and metrics CSV.
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    # Load cleaned data and get identical holdout test split
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data(raw_df)
    _, X_test, _, y_test = split_data(cleaned_df, test_size=0.20, random_state=42)

    # Model slug mapping
    models_map = {
        "Decision Tree": "decision_tree.joblib",
        "Support Vector Machine": "svm.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
    }

    # Baseline prevalence: P(Y = 1) in test set
    prevalence = float(np.mean(y_test))
    print(f"--> Holdout Test Positive-Class Prevalence (>50K): {prevalence:.4f} ({prevalence*100:.2f}%)")

    pr_records = []
    pr_data = {"prevalence": prevalence, "models": {}}

    plt.figure(figsize=(9, 6.5), dpi=300)

    # Plot baseline reference line
    plt.axhline(
        y=prevalence,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline Prevalence (AP={prevalence:.4f})",
        alpha=0.85,
    )

    for name, slug in models_map.items():
        model_path = os.path.join(models_dir, slug)
        if not os.path.exists(model_path):
            print(f"    [Warning] Missing model file: {model_path}")
            continue

        pipe = joblib.load(model_path)

        # Get continuous prediction probabilities
        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_test)
        else:
            scores = pipe.predict(X_test)

        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)

        # Discrete predictions at default threshold (0.5)
        hard_preds = pipe.predict(X_test)
        prec_05 = precision_score(y_test, hard_preds, pos_label=1, zero_division=0)
        rec_05 = recall_score(y_test, hard_preds, pos_label=1, zero_division=0)

        pr_records.append({
            "Model": name,
            "Average Precision": round(float(ap), 4),
            "Precision (Threshold=0.5)": round(float(prec_05), 4),
            "Recall (Threshold=0.5)": round(float(rec_05), 4),
        })

        color = MODEL_COLORS.get(name, "#1f2937")
        plt.plot(
            recall,
            precision,
            label=f"{name} (AP = {ap:.4f})",
            color=color,
            linewidth=2.2,
        )

        # Save curve points for Streamlit interactive plotting
        pr_data["models"][name] = {
            "recall": recall.tolist(),
            "precision": precision.tolist(),
            "average_precision": round(float(ap), 4),
            "prec_threshold_05": round(float(prec_05), 4),
            "rec_threshold_05": round(float(rec_05), 4),
        }

    plt.title("Precision-Recall Curves for Adult Income Classification (>50K Class)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Recall (True Positive Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10, shadow=False)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved PR Curve Figure: {output_png}")

    df_pr = pd.DataFrame(pr_records).sort_values(by="Average Precision", ascending=False).reset_index(drop=True)
    df_pr.to_csv(output_csv, index=False)
    joblib.dump(pr_data, output_joblib)
    print(f"--> Saved PR Metrics Table: {output_csv}")
    print(df_pr.to_string(index=False))

    return df_pr, pr_data


# ---------------------------------------------------------------------------
# TASK 3: Cross-Validation Stability Box Plots
# ---------------------------------------------------------------------------
def generate_cv_stability_boxplot(
    folds_path="results/baseline_detailed_folds.joblib",
    output_png="results/cv_stability_boxplot.png",
    output_csv="results/cv_fold_scores.csv",
):
    """
    Extract fold-wise F1 scores from existing 5-fold Stratified Cross-Validation results.
    Generates structured CSV and combined box plot displaying median, IQR, and fold observations.
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Missing detailed folds artifact: {folds_path}")

    folds = joblib.load(folds_path)

    # Extract fold-wise scores
    records = []
    for model_name, metrics in folds.items():
        f1_scores = metrics.get("test_f1", [])
        for fold_idx, score in enumerate(f1_scores, 1):
            records.append({
                "Model": model_name,
                "Fold": f"Fold {fold_idx}",
                "Fold Number": fold_idx,
                "F1 Score": round(float(score), 4),
            })

    df_folds = pd.DataFrame(records)
    df_folds.to_csv(output_csv, index=False)
    print(f"--> Saved Fold-Wise F1 Scores: {output_csv}")

    # Compute order by mean F1 score
    model_order = (
        df_folds.groupby("Model")["F1 Score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    palette = [MODEL_COLORS.get(m, "#4b5563") for m in model_order]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Box plot displaying median, IQR, whiskers
    sns.boxplot(
        data=df_folds,
        x="Model",
        y="F1 Score",
        hue="Model",
        legend=False,
        order=model_order,
        palette=palette,
        width=0.45,
        fliersize=0,
        ax=ax,
        boxprops=dict(alpha=0.75, edgecolor="#1f2937", linewidth=1.2),
        medianprops=dict(color="#111827", linewidth=2.0),
        whiskerprops=dict(color="#374151", linewidth=1.2),
        capprops=dict(color="#374151", linewidth=1.2),
    )

    # Overlay stripplot showing all 5 individual fold observations with jitter
    sns.stripplot(
        data=df_folds,
        x="Model",
        y="F1 Score",
        order=model_order,
        color="#111827",
        size=7,
        jitter=0.15,
        alpha=0.85,
        ax=ax,
    )

    # Annotate mean markers (diamond)
    means = df_folds.groupby("Model")["F1 Score"].mean().loc[model_order]
    for idx, (model_name, mean_val) in enumerate(means.items()):
        ax.scatter(
            idx,
            mean_val,
            marker="D",
            color="#dc2626",
            s=50,
            zorder=5,
            label="Mean F1" if idx == 0 else "",
        )
        ax.text(
            idx,
            mean_val + 0.006,
            f"μ={mean_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1f2937",
        )

    ax.set_title("5-Fold Cross-Validation F1-Score Stability & Dispersion Across Algorithms", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Supervised Classification Model", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation F1-Score (>50K Class)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.set_ylim([0.48, 0.72])

    # Custom legend for symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#111827", lw=2, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#dc2626", markersize=8, label="Mean F1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#111827", markersize=7, label="Fold Observation (N=5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved CV Stability Boxplot: {output_png}")

    return df_folds


# ---------------------------------------------------------------------------
# TASK 2: Hyperparameter Heatmaps & 1D Plots
# ---------------------------------------------------------------------------
def plot_svm_heatmap(cv_results_df, output_png="results/svm_hyperparameter_heatmap.png"):
    """
    Generate 2D heatmap of SVM hyperparameters (C vs. gamma) colored by Mean CV F1.
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    df = cv_results_df.copy()
    c_col = "param_classifier__C" if "param_classifier__C" in df.columns else "param_C"
    gamma_col = "param_classifier__gamma" if "param_classifier__gamma" in df.columns else "param_gamma"

    df[c_col] = df[c_col].astype(str)
    df[gamma_col] = df[gamma_col].astype(str)

    pivot_table = df.pivot_table(
        index=gamma_col,
        columns=c_col,
        values="mean_test_f1",
        aggfunc="mean",
    )

    # Sort columns numerically
    sorted_c = sorted(pivot_table.columns, key=lambda x: float(x))
    pivot_table = pivot_table[sorted_c]

    plt.figure(figsize=(7.5, 5.5), dpi=300)
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar_kws={"label": "Mean CV F1-Score"},
        linewidths=1.0,
        linecolor="#ffffff",
    )

    plt.title("Support Vector Machine (RBF Kernel)\nHyperparameter Optimization Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Regularization Parameter C", fontsize=11, fontweight="bold")
    plt.ylabel("Kernel Coefficient gamma", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved SVM Heatmap: {output_png}")


def plot_decision_tree_heatmap(cv_results_df, output_png="results/decision_tree_hyperparameter_heatmap.png"):
    """
    Generate 2D heatmap of Decision Tree hyperparameters (max_depth vs. min_samples_leaf).
    Conditioned on optimal criterion ('gini') and min_samples_split=2.
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    df = cv_results_df.copy()
    depth_col = "param_classifier__max_depth" if "param_classifier__max_depth" in df.columns else "param_max_depth"
    leaf_col = "param_classifier__min_samples_leaf" if "param_classifier__min_samples_leaf" in df.columns else "param_min_samples_leaf"
    crit_col = "param_classifier__criterion" if "param_classifier__criterion" in df.columns else "param_criterion"
    split_col = "param_classifier__min_samples_split" if "param_classifier__min_samples_split" in df.columns else "param_min_samples_split"

    # Filter to optimal criterion and split for a clean 2D slice
    if crit_col in df.columns and "gini" in df[crit_col].values:
        df = df[df[crit_col] == "gini"]
    if split_col in df.columns and 2 in df[split_col].values:
        df = df[df[split_col] == 2]

    # Represent None as string 'None' and integers cleanly
    df[depth_col] = df[depth_col].apply(lambda x: "None" if pd.isna(x) or str(x) in ["None", "nan"] else str(int(float(x))))
    df[leaf_col] = df[leaf_col].apply(lambda x: int(float(x)))

    pivot_table = df.pivot_table(
        index=leaf_col,
        columns=depth_col,
        values="mean_test_f1",
        aggfunc="mean",
    )

    # Order columns: numeric first, then 'None'
    def sort_depth(d):
        return (1, 9999) if d == "None" else (0, int(float(d)))

    sorted_cols = sorted(pivot_table.columns, key=sort_depth)
    pivot_table = pivot_table[sorted_cols]
    pivot_table = pivot_table.sort_index(ascending=True)

    plt.figure(figsize=(8.5, 6), dpi=300)
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={"label": "Mean CV F1-Score"},
        linewidths=1.0,
        linecolor="#ffffff",
    )

    plt.title("Decision Tree: max_depth vs. min_samples_leaf\n(Criterion='gini', min_samples_split=2)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Maximum Tree Depth (max_depth)", fontsize=11, fontweight="bold")
    plt.ylabel("Minimum Leaf Samples (min_samples_leaf)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved Decision Tree Heatmap: {output_png}")


def plot_knn_heatmap(cv_results_df, output_png="results/knn_hyperparameter_heatmap.png"):
    """
    Generate 2D heatmap of KNN hyperparameters (n_neighbors vs. weights).
    Conditioned on optimal distance metric p=2 (Euclidean).
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    df = cv_results_df.copy()
    k_col = "param_classifier__n_neighbors" if "param_classifier__n_neighbors" in df.columns else "param_n_neighbors"
    w_col = "param_classifier__weights" if "param_classifier__weights" in df.columns else "param_weights"
    p_col = "param_classifier__p" if "param_classifier__p" in df.columns else "param_p"

    # Filter to Euclidean distance (p=2)
    if p_col in df.columns and 2 in df[p_col].values:
        df = df[df[p_col] == 2]

    df[k_col] = df[k_col].apply(lambda x: int(float(x)))
    df[w_col] = df[w_col].astype(str)

    pivot_table = df.pivot_table(
        index=w_col,
        columns=k_col,
        values="mean_test_f1",
        aggfunc="mean",
    )
    pivot_table = pivot_table.sort_index(axis=1)

    plt.figure(figsize=(8, 4.5), dpi=300)
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="Purples",
        cbar_kws={"label": "Mean CV F1-Score"},
        linewidths=1.0,
        linecolor="#ffffff",
    )

    plt.title("K-Nearest Neighbors: n_neighbors vs. Weighting Function\n(Distance Metric p=2: Euclidean)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Neighbors (k)", fontsize=11, fontweight="bold")
    plt.ylabel("Weighting Strategy", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved KNN Heatmap: {output_png}")


def plot_logistic_regression(cv_results_df, output_png="results/logistic_regression_hyperparameter_plot.png"):
    """
    Generate 1D hyperparameter plot for Logistic Regression (C vs. Mean CV F1).
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    df = cv_results_df.copy()
    c_col = "param_classifier__C" if "param_classifier__C" in df.columns else "param_C"
    solver_col = "param_classifier__solver" if "param_classifier__solver" in df.columns else "param_solver"

    df[c_col] = df[c_col].astype(float)
    df = df.sort_values(by=c_col)

    plt.figure(figsize=(7.5, 5), dpi=300)

    if solver_col in df.columns:
        for solver_name, group in df.groupby(solver_col):
            plt.plot(
                group[c_col],
                group["mean_test_f1"],
                marker="o",
                linewidth=2.2,
                label=f"Solver: {solver_name}",
            )
            # Add error band if std available
            if "std_test_f1" in group.columns:
                plt.fill_between(
                    group[c_col],
                    group["mean_test_f1"] - group["std_test_f1"],
                    group["mean_test_f1"] + group["std_test_f1"],
                    alpha=0.15,
                )
    else:
        plt.plot(df[c_col], df["mean_test_f1"], marker="o", color="#f59e0b", linewidth=2.2, label="Logistic Regression")
        if "std_test_f1" in df.columns:
            plt.fill_between(
                df[c_col],
                df["mean_test_f1"] - df["std_test_f1"],
                df["mean_test_f1"] + df["std_test_f1"],
                color="#f59e0b",
                alpha=0.15,
            )

    plt.xscale("log")
    plt.title("Logistic Regression: Regularization Parameter C vs. Cross-Validation F1", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Inverse Regularization Strength C (log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean CV F1-Score (>50K Class)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved Logistic Regression Plot: {output_png}")


def plot_gaussian_nb(cv_results_df, output_png="results/gaussian_nb_hyperparameter_plot.png"):
    """
    Generate 1D hyperparameter plot for Gaussian Naive Bayes (var_smoothing vs. Mean CV F1).
    """
    set_research_style()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    df = cv_results_df.copy()
    vs_col = "param_classifier__var_smoothing" if "param_classifier__var_smoothing" in df.columns else "param_var_smoothing"

    df[vs_col] = df[vs_col].astype(float)
    df = df.sort_values(by=vs_col)

    plt.figure(figsize=(7.5, 5), dpi=300)
    plt.plot(df[vs_col], df["mean_test_f1"], marker="s", color="#ef4444", linewidth=2.2, label="Gaussian Naive Bayes")

    if "std_test_f1" in df.columns:
        plt.fill_between(
            df[vs_col],
            df["mean_test_f1"] - df["std_test_f1"],
            df["mean_test_f1"] + df["std_test_f1"],
            color="#ef4444",
            alpha=0.15,
        )

    plt.xscale("log")
    plt.title("Gaussian Naive Bayes: Variance Smoothing vs. Cross-Validation F1", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Variance Smoothing Parameter (var_smoothing, log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean CV F1-Score (>50K Class)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved Gaussian NB Plot: {output_png}")
