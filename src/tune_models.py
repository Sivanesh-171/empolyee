"""
Hyperparameter Optimization and Final Model Selection Workflow.
Executes GridSearchCV and RandomizedSearchCV inside leak-free Scikit-learn Pipelines,
compares tuned models using 5-fold cross-validated F1-score, selects the winning configuration,
refits on the entire training dataset, and evaluates once on the untouched test set.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from preprocess import load_raw_data, clean_data, split_data, create_pipeline
from research_visualizations import (
    plot_svm_heatmap,
    plot_decision_tree_heatmap,
    plot_knn_heatmap,
    plot_logistic_regression,
    plot_gaussian_nb,
    generate_precision_recall_curves,
)


# ---------------------------------------------------------------------------
# Search Spaces Definition
# ---------------------------------------------------------------------------
def get_tuning_configurations():
    """
    Define base estimators, parameter search spaces, and iteration budgets.
    All parameter names correctly reference the classifier inside the Pipeline.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "random_n_iter": 5,
            "has_proba": True,
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "classifier__max_depth": [5, 8, 10, 12, None],
                "classifier__min_samples_leaf": [5, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__criterion": ["gini", "entropy"],
            },
            "random_n_iter": 25,
            "has_proba": True,
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
            "random_n_iter": 10,
            "has_proba": True,
        },
        "Support Vector Machine": {
            # Tuned with base SVC for computational efficiency during CV;
            # CalibratedClassifierCV(..., ensemble=False) is applied on final refit
            # to provide calibrated probabilities without deprecation warnings.
            "estimator": SVC(kernel="rbf", random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.01, 0.1],
                "classifier__kernel": ["rbf"],
            },
            "random_n_iter": 6,
            "has_proba": True,
        },
        "Naive Bayes": {
            "estimator": GaussianNB(),
            "param_grid": {
                "classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8],
            },
            "random_n_iter": 4,
            "has_proba": True,
        },
    }


# ---------------------------------------------------------------------------
# Search Execution Functions
# ---------------------------------------------------------------------------
def run_grid_and_random_search(X_train, y_train):
    """
    Execute both GridSearchCV and RandomizedSearchCV across all 5 models inside
    complete pipelines with 5-fold Stratified cross-validation optimizing F1-score.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    configs = get_tuning_configurations()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    grid_results = []
    random_results = []
    comparison_records = []

    best_grid_estimators = {}
    best_random_estimators = {}

    print("=" * 80)
    print("STEP 2: HYPERPARAMETER OPTIMIZATION (GRIDSEARCHCV & RANDOMIZEDSEARCHCV)")
    print("=" * 80)

    for name, config in configs.items():
        base_clf = config["estimator"]
        param_grid = config["param_grid"]
        n_iter = config["random_n_iter"]
        pipeline = create_pipeline(base_clf)

        # ---------------------------------------------------------
        # 1. GridSearchCV
        # ---------------------------------------------------------
        total_grid_candidates = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"\n--> [GridSearchCV] {name} ({total_grid_candidates} candidates x 5 folds = {total_grid_candidates*5} fits)...")
        grid_start = time.time()
        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        gs.fit(X_train, y_train)
        grid_time = round(time.time() - grid_start, 2)

        best_g_idx = gs.best_index_
        g_f1 = gs.cv_results_["mean_test_f1"][best_g_idx]
        g_acc = gs.cv_results_["mean_test_accuracy"][best_g_idx]
        g_prec = gs.cv_results_["mean_test_precision"][best_g_idx]
        g_rec = gs.cv_results_["mean_test_recall"][best_g_idx]
        g_auc = gs.cv_results_["mean_test_roc_auc"][best_g_idx]
        best_g_params = gs.best_params_
        best_grid_estimators[name] = gs.best_estimator_

        print(f"    GridSearch Best F1: {g_f1:.4f} | Acc: {g_acc:.4f} | ROC-AUC: {g_auc:.4f} | Time: {grid_time}s")
        print(f"    Best Params: {best_g_params}")

        grid_rec = {
            "Model": name,
            "Search Method": "GridSearchCV",
            "Best Hyperparameters": json.dumps(best_g_params),
            "Best CV F1-Score": round(g_f1, 4),
            "CV Accuracy": round(g_acc, 4),
            "CV Precision": round(g_prec, 4),
            "CV Recall": round(g_rec, 4),
            "CV ROC-AUC": round(g_auc, 4),
            "Tuning Time (s)": grid_time,
        }
        grid_results.append(grid_rec)
        comparison_records.append(grid_rec)

        # Preserve complete candidate GridSearchCV results (Task 2.A)
        grid_slugs = {
            "Support Vector Machine": "svm",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "knn",
            "Logistic Regression": "logistic_regression",
            "Naive Bayes": "gaussian_nb",
        }
        cand_slug = grid_slugs.get(name, name.lower().replace(" ", "_").replace("-", "_"))
        df_candidates = pd.DataFrame(gs.cv_results_)
        df_candidates.to_csv(f"results/{cand_slug}_grid_cv_results.csv", index=False)
        print(f"    Saved full candidate CV results: results/{cand_slug}_grid_cv_results.csv ({len(df_candidates)} candidates)")

        # Generate hyperparameter heatmap / 1D plot (Task 2.B)
        if name == "Support Vector Machine":
            plot_svm_heatmap(df_candidates)
        elif name == "Decision Tree":
            plot_decision_tree_heatmap(df_candidates)
        elif name == "K-Nearest Neighbors":
            plot_knn_heatmap(df_candidates)
        elif name == "Logistic Regression":
            plot_logistic_regression(df_candidates)
        elif name == "Naive Bayes":
            plot_gaussian_nb(df_candidates)

        # ---------------------------------------------------------
        # 2. RandomizedSearchCV
        # ---------------------------------------------------------
        print(f"--> [RandomizedSearchCV] {name} ({n_iter} iterations x 5 folds = {n_iter*5} fits)...")
        rand_start = time.time()
        rs = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
        rs.fit(X_train, y_train)
        rand_time = round(time.time() - rand_start, 2)

        best_r_idx = rs.best_index_
        r_f1 = rs.cv_results_["mean_test_f1"][best_r_idx]
        r_acc = rs.cv_results_["mean_test_accuracy"][best_r_idx]
        r_prec = rs.cv_results_["mean_test_precision"][best_r_idx]
        r_rec = rs.cv_results_["mean_test_recall"][best_r_idx]
        r_auc = rs.cv_results_["mean_test_roc_auc"][best_r_idx]
        best_r_params = rs.best_params_
        best_random_estimators[name] = rs.best_estimator_

        print(f"    RandomSearch Best F1: {r_f1:.4f} | Acc: {r_acc:.4f} | ROC-AUC: {r_auc:.4f} | Time: {rand_time}s")
        print(f"    Best Params: {best_r_params}")

        rand_rec = {
            "Model": name,
            "Search Method": "RandomizedSearchCV",
            "Best Hyperparameters": json.dumps(best_r_params),
            "Best CV F1-Score": round(r_f1, 4),
            "CV Accuracy": round(r_acc, 4),
            "CV Precision": round(r_prec, 4),
            "CV Recall": round(r_rec, 4),
            "CV ROC-AUC": round(r_auc, 4),
            "Tuning Time (s)": rand_time,
        }
        random_results.append(rand_rec)
        comparison_records.append(rand_rec)

    # Save detailed search reports
    grid_df = pd.DataFrame(grid_results)
    rand_df = pd.DataFrame(random_results)
    comp_df = pd.DataFrame(comparison_records)

    grid_df.to_csv("results/grid_search_results.csv", index=False)
    rand_df.to_csv("results/random_search_results.csv", index=False)
    comp_df.to_csv("results/tuned_model_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("TUNING COMPARISON TABLE (All Models - Grid vs Randomized Search):")
    print("=" * 80)
    print(comp_df[["Model", "Search Method", "Best CV F1-Score", "CV ROC-AUC", "CV Accuracy", "Tuning Time (s)"]].to_string(index=False))
    print("=" * 80)

    return comp_df, best_grid_estimators, best_random_estimators


# ---------------------------------------------------------------------------
# Final Model Selection and Test Evaluation
# ---------------------------------------------------------------------------
def select_and_evaluate_final_models(comp_df, best_grid_estimators, best_random_estimators, X_train, y_train, X_test, y_test):
    """
    Select the single winning model strictly by highest Cross-Validation F1-score.
    Refit all 5 tuned models on full X_train.
    Evaluate once on untouched X_test, y_test.
    """
    print("\n" + "=" * 80)
    print("STEP 3: FINAL MODEL SELECTION (BASED STRICTLY ON CROSS-VALIDATION F1-SCORE)")
    print("=" * 80)

    # Find winning configuration across all models and search methods
    sorted_comp = comp_df.sort_values(by=["Best CV F1-Score", "CV ROC-AUC"], ascending=[False, False]).reset_index(drop=True)
    winner_row = sorted_comp.iloc[0]
    winner_name = winner_row["Model"]
    winner_method = winner_row["Search Method"]
    winner_f1 = winner_row["Best CV F1-Score"]
    winner_params = json.loads(winner_row["Best Hyperparameters"])

    print(f"🥇 WINNING MODEL CONFIGURATION: {winner_name} via {winner_method}")
    print(f"   Validation CV F1-Score: {winner_f1:.4f}")
    print(f"   Optimal Hyperparameters: {winner_params}")
    print("   (Selection made strictly from training fold cross-validation without test set access)")

    # Dictionary mapping model name to its best estimator across Grid/Random search
    best_tuned_pipelines = {}
    best_hyperparameters_summary = {}

    model_slugs = {
        "Decision Tree": "decision_tree",
        "Support Vector Machine": "svm",
        "Logistic Regression": "logistic_regression",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
    }

    # Extract the best tuned pipeline for each model
    for model_name in model_slugs.keys():
        sub_df = comp_df[comp_df["Model"] == model_name].sort_values("Best CV F1-Score", ascending=False)
        best_method = sub_df.iloc[0]["Search Method"]
        best_params = json.loads(sub_df.iloc[0]["Best Hyperparameters"])
        best_hyperparameters_summary[model_name] = {
            "search_method": best_method,
            "best_cv_f1": sub_df.iloc[0]["Best CV F1-Score"],
            "best_params": best_params,
        }

        if best_method == "GridSearchCV":
            pipe = best_grid_estimators[model_name]
        else:
            pipe = best_random_estimators[model_name]

        # For SVM, wrap with CalibratedClassifierCV(..., ensemble=False) on the final estimator
        # before refitting on full training set so the app has calibrated probabilities without deprecation warnings.
        if model_name == "Support Vector Machine":
            clf_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
            clf_params["random_state"] = 42
            base_svc = SVC(**clf_params)
            calibrated_svc = CalibratedClassifierCV(base_svc, ensemble=False)
            pipe.set_params(classifier=calibrated_svc)

        # Refit complete pipeline on the ENTIRE training dataset (26,029 samples)
        print(f"--> Refitting optimal tuned pipeline on full training set: {model_name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)
        pipe.fit_time_ = fit_time

        best_tuned_pipelines[model_name] = pipe

        # Save individual tuned pipeline
        slug = model_slugs[model_name]
        joblib.dump(pipe, f"models/{slug}.joblib")

    # Save the overall winning pipeline
    winning_pipeline = best_tuned_pipelines[winner_name]
    joblib.dump(winning_pipeline, "models/best_pipeline.joblib")
    with open("results/best_hyperparameters.json", "w") as f:
        json.dump(best_hyperparameters_summary, f, indent=2)

    # ---------------------------------------------------------
    # Untouched Holdout Test Set Evaluation (EXACTLY ONCE)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: UNTOUCHED TEST SET EVALUATION (FINAL GENERALIZATION BENCHMARK)")
    print("=" * 80)

    test_records = []
    all_cms = {}
    all_rocs = {}

    for model_name, pipe in best_tuned_pipelines.items():
        y_test_pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            y_test_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = pipe.decision_function(X_test)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Train accuracy for overfitting check
        y_train_pred = pipe.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)

        all_cms[model_name] = cm
        all_rocs[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": round(test_auc, 4),
        }

        test_records.append({
            "Model": model_name,
            "Optimal Search": best_hyperparameters_summary[model_name]["search_method"],
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "F1 Score": round(test_f1, 4),
            "ROC-AUC": round(test_auc, 4),
            "Training Time (s)": getattr(pipe, "fit_time_", 0.0),
        })

    test_metrics_df = pd.DataFrame(test_records)
    test_metrics_df = test_metrics_df.sort_values(by=["F1 Score", "ROC-AUC"], ascending=[False, False]).reset_index(drop=True)
    test_metrics_df.index = test_metrics_df.index + 1
    test_metrics_df.index.name = "Rank"

    # Save to metrics.csv and final_test_metrics.csv
    test_metrics_df.to_csv("results/final_test_metrics.csv")
    test_metrics_df.to_csv("results/metrics.csv")  # Preserves Streamlit app compatibility
    joblib.dump(all_cms, "results/confusion_matrices.joblib")
    joblib.dump(all_rocs, "results/roc_curves.joblib")

    print("\n" + "=" * 80)
    print("FINAL HOLDOUT TEST PERFORMANCE BENCHMARK (UNTOUCHED TEST SET - 6,508 SAMPLES):")
    print("=" * 80)
    print(test_metrics_df.to_string())
    print("=" * 80)

    # Detailed Classification Report for the Winner
    print(f"\nDetailed Classification Report for Winner ({winner_name}):")
    y_win_pred = winning_pipeline.predict(X_test)
    print(classification_report(y_test, y_win_pred, target_names=["<=50K (0)", ">50K (1)"]))

    # Generate Final Visualizations
    generate_tuning_visualizations(comp_df, test_metrics_df, all_cms, all_rocs, winner_name)

    # Generate Precision-Recall Curves (Task 1)
    generate_precision_recall_curves()

    return test_metrics_df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def generate_tuning_visualizations(comp_df, test_metrics_df, all_cms, all_rocs, winner_name):
    """Generate static publication-quality visualization figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. GridSearchCV vs RandomizedSearchCV F1-score comparison
    plt.figure(figsize=(10, 5))
    pivot_f1 = comp_df.pivot(index="Model", columns="Search Method", values="Best CV F1-Score")
    ax = pivot_f1.plot(kind="bar", figsize=(10, 5), color=["#3b82f6", "#10b981"], edgecolor="black", width=0.7)
    plt.title("Hyperparameter Optimization: GridSearchCV vs. RandomizedSearchCV (CV F1-Score)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Validation F1-Score", fontsize=11)
    plt.ylim(0.45, 0.75)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(title="Search Method", frameon=True)
    plt.tight_layout()
    plt.savefig("results/tuning_comparison.png", dpi=200)
    plt.close()

    # 2. Final Confusion Matrix for Winner
    plt.figure(figsize=(6, 5))
    win_cm = all_cms[winner_name]
    sns.heatmap(win_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["<=50K (0)", ">50K (1)"],
                yticklabels=["<=50K (0)", ">50K (1)"])
    plt.title(f"Final Test Confusion Matrix: {winner_name}", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/final_confusion_matrix.png", dpi=200)
    plt.close()

    # 3. Final Overlaid ROC Curves for All 5 Tuned Models
    plt.figure(figsize=(8, 6))
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    for idx, (m_name, rdata) in enumerate(all_rocs.items()):
        plt.plot(rdata["fpr"], rdata["tpr"], label=f"{m_name} (AUC = {rdata['roc_auc']:.4f})", color=colors[idx % len(colors)], lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.50)")
    plt.title("Final Holdout Test ROC Curves: All 5 Tuned Models", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/final_roc_curve.png", dpi=200)
    plt.close()

    # 4. Final Performance Comparison Grouped Bar Chart
    plt.figure(figsize=(11, 5))
    metrics_plot = test_metrics_df.set_index("Model")[["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    metrics_plot.plot(kind="bar", figsize=(11, 5), colormap="Set2", edgecolor="black", width=0.8)
    plt.title("Final Holdout Test Performance Comparison Across All 5 Tuned Models", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/final_performance_comparison.png", dpi=200)
    plt.close()

    print("Visualizations successfully saved to results/*.png")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    raw_df = load_raw_data("data/adult.csv")
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df, test_size=0.20, random_state=42)

    comp_df, best_g_estimators, best_r_estimators = run_grid_and_random_search(X_train, y_train)
    select_and_evaluate_final_models(comp_df, best_g_estimators, best_r_estimators, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
