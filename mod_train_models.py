"""Modify src/train_models.py in the project dir: add the new CV/tuning/selection workflow."""
import sys
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir

D = project_dir()
P = lambda rel: f"{D}/{rel}"

src = open(P("src/train_models.py"), encoding="utf-8").read()

# --- 1. Extend imports ---
old_imp = '''import time
import os
import sys
'''
new_imp = '''import time
import os
import sys
import json
'''
assert old_imp in src
src = src.replace(old_imp, new_imp)

old_imp2 = '''from sklearn.calibration import CalibratedClassifierCV
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
'''
new_imp2 = '''from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
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

from preprocess import (
    build_model_pipeline,
    load_raw_data,
    clean_data,
    split_and_preprocess,
)
from model_tuning import (
    get_param_grids,
    get_param_distributions,
    get_random_iter_counts,
    grid_size,
    run_grid_search,
    run_randomized_search,
)
'''
assert old_imp2 in src
src = src.replace(old_imp2, new_imp2)

# --- 2. Add workflow functions after get_models() ---
anchor = '''def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):'''
new_funcs = '''# ---------------------------------------------------------------------------
# Cross-Validation / Hyperparameter Optimization Workflow (leakage-safe)
# ---------------------------------------------------------------------------

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine": "svm",
    "Naive Bayes": "naive_bayes",
}

SCORING_LIST = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# StandardScaler is only required for distance/gradient-based models; scaling is
# harmless for trees and GaussianNB, so we disable it there ("appropriate
# preprocessing for each model").
SCALE_NUMERIC_MAP = {
    "Logistic Regression": True,
    "Decision Tree": False,
    "K-Nearest Neighbors": True,
    "Support Vector Machine": True,
    "Naive Bayes": False,
}

N_OUTER_FOLDS = 5  # StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def get_tunable_models(calib_cv):
    """
    Return the 5 models as complete Scikit-learn Pipelines (preprocessing + model).

    The SVM entry reuses the existing CalibratedClassifierCV(SVC) implementation,
    but with an explicit internal CV splitter. Preprocessing happens upstream in
    the Pipeline, so the calibration folds operate on data already transformed by
    the pipeline fitted on the outer training fold (no leakage, no double
    preprocessing of the same records).
    """
    models = get_models()
    models["Support Vector Machine"] = CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=42), ensemble=False, cv=calib_cv
    )
    return models


def get_shared_cv():
    """The single shared splitter used for all CV comparisons (same folds for every model)."""
    return StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)


def run_baseline_cv(models, X_train_raw, y_train, cv):
    """
    Baseline: 5-fold stratified cross_validate for Accuracy/Precision/Recall/F1/ROC-AUC.

    All folds are fit independently per pipeline, so transformers are fitted on
    each training fold only. Returns a DataFrame sorted by Mean F1-Score.
    """
    rows = []
    for name, model in models.items():
        pipe = build_model_pipeline(model, scale_numeric=SCALE_NUMERIC_MAP[name])
        res = cross_validate(
            pipe, X_train_raw, y_train, cv=cv, scoring=SCORING_LIST, n_jobs=-1
        )
        rows.append(
            {
                "Model": name,
                "Mean Accuracy": round(res["test_accuracy"].mean(), 4),
                "Mean Precision": round(res["test_precision"].mean(), 4),
                "Mean Recall": round(res["test_recall"].mean(), 4),
                "Mean F1-Score": round(res["test_f1"].mean(), 4),
                "Std F1-Score": round(res["test_f1"].std(), 4),
                "Mean ROC-AUC": round(res["test_roc_auc"].mean(), 4),
            }
        )
    df = pd.DataFrame(rows).sort_values(by="Mean F1-Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


def tune_model(name, model, X_train_raw, y_train, cv, method):
    """Run one hyperparameter search (grid or random) on a complete pipeline."""
    pipe = build_model_pipeline(model, scale_numeric=SCALE_NUMERIC_MAP[name])

    if method == "grid":
        grid = get_param_grids()[name]
        search = run_grid_search(pipe, grid, cv, scoring="f1", n_jobs=-1)
    else:
        dist = get_param_distributions()[name]
        n_iter = get_random_iter_counts()[name]
        search = run_randomized_search(
            pipe, dist, n_iter, cv, scoring="f1", n_jobs=-1, random_state=42
        )

    t0 = time.time()
    search.fit(X_train_raw, y_train)
    fit_time = time.time() - t0

    # Fresh 5-fold CV metrics of the tuned best estimator (same folds).
    res = cross_validate(
        search.best_estimator_, X_train_raw, y_train, cv=cv, scoring=SCORING_LIST, n_jobs=-1
    )
    return search, fit_time, res


def run_tuning(models, X_train_raw, y_train, cv):
    """Run GridSearchCV and RandomizedSearchCV for all models. Returns (df, searches)."""
    rows = []
    searches = {}
    for name, model in models.items():
        for method in ("grid", "random"):
            print(f"  Tuning {name} [{method} search]...")
            search, fit_time, res = tune_model(name, model, X_train_raw, y_train, cv, method)
            rows.append(
                {
                    "Model": name,
                    "Search Method": "GridSearchCV" if method == "grid" else "RandomizedSearchCV",
                    "Best Hyperparameters": str(search.best_params_),
                    "Best CV F1": round(search.best_score_, 4),
                    "CV Accuracy": round(res["test_accuracy"].mean(), 4),
                    "CV Precision": round(res["test_precision"].mean(), 4),
                    "CV Recall": round(res["test_recall"].mean(), 4),
                    "CV ROC-AUC": round(res["test_roc_auc"].mean(), 4),
                    "Training Time (s)": round(fit_time, 2),
                }
            )
            searches[(name, method)] = search

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Best CV F1", ascending=False).reset_index(drop=True)
    return df, searches


def select_best_model(tuned_df):
    """Pick the overall best configuration by cross-validated F1 (test set NOT used)."""
    best = tuned_df.sort_values(
        by=["Best CV F1", "CV ROC-AUC", "CV Accuracy"], ascending=False
    ).iloc[0]
    method = "grid" if best["Search Method"] == "GridSearchCV" else "random"
    return best, method


def evaluate_final(pipe, X_test_raw, y_test):
    """
    Evaluate the selected pipeline exactly once on the untouched test set.

    Returns (metrics dict, confusion_matrix, (fpr, tpr, thresholds)).
    """
    y_pred = pipe.predict(X_test_raw)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test_raw)[:, 1]
    else:
        y_proba = pipe.decision_function(X_test_raw)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return metrics, cm, (fpr, tpr, thresholds)


def save_plots(baseline_df, tuned_df, final_metrics, cm, roc_data):
    """Generate the professional static visualizations in results/plots/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results/plots", exist_ok=True)
    fpr, tpr, auc = roc_data

    # 1. Baseline F1 comparison (sorted)
    b = baseline_df.sort_values("Mean F1-Score", ascending=False)
    plt.figure(figsize=(8, 4.2))
    plt.barh(b["Model"], b["Mean F1-Score"], color="#2563eb")
    plt.xlabel("Mean F1-Score (5-fold CV)")
    plt.title("Baseline Model F1-Score Comparison")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/01_baseline_f1.png", dpi=110)
    plt.close()

    # 2. Baseline CV metrics comparison
    b2 = baseline_df.set_index("Model")
    plt.figure(figsize=(9, 5))
    for col, marker in [
        ("Mean Accuracy", "o"),
        ("Mean Precision", "s"),
        ("Mean Recall", "^"),
        ("Mean F1-Score", "D"),
        ("Mean ROC-AUC", "*"),
    ]:
        plt.plot(b2.index, b2[col], marker=marker, label=col)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title("Baseline Cross-Validation Metrics per Model")
    plt.tight_layout()
    plt.savefig("results/plots/02_baseline_cv_metrics.png", dpi=110)
    plt.close()

    # 3. GridSearchCV vs RandomizedSearchCV best CV F1 per model
    t = tuned_df.pivot_table(index="Model", columns="Search Method", values="Best CV F1")
    t = t.reindex(index=b["Model"])
    plt.figure(figsize=(8, 4.2))
    ax = t.plot(kind="bar", color=["#2563eb", "#f97316"])
    plt.ylabel("Best CV F1-Score")
    plt.title("GridSearchCV vs RandomizedSearchCV - Best CV F1 per Model")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/plots/03_tuned_f1_comparison.png", dpi=110)
    plt.close()

    # 4. Best hyperparameters per model/method (text table figure)
    plt.figure(figsize=(11, 4.5))
    plt.axis("off")
    cols = ["Model", "Search Method", "Best Hyperparameters", "Best CV F1"]
    rows = [tuple(r) for r in tuned_df[cols].itertuples(index=False)]
    plt.table(cellText=rows, colLabels=cols, loc="center")
    plt.title("Best Hyperparameters per Model (Cross-Validated F1)")
    plt.savefig("results/plots/04_best_hyperparameters.png", dpi=110)
    plt.close()

    # 5. Final confusion matrix (untouched test set)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)
    tick = ["<=50K", ">50K"]
    plt.xticks([0, 1], tick)
    plt.yticks([0, 1], tick)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.title("Final Confusion Matrix (Untouched Test Set)")
    plt.tight_layout()
    plt.savefig("results/plots/05_final_confusion_matrix.png", dpi=110)
    plt.close()

    # 6. Final ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC=0.50)")
    plt.plot(fpr, tpr, "-", linewidth=2, label=f"Best Model (AUC={final_metrics['ROC-AUC']:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/06_final_roc_curve.png", dpi=110)
    plt.close()

    # 7. Final performance on the untouched test set
    keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    plt.figure(figsize=(8, 4.2))
    plt.bar(keys, [final_metrics[k] for k in keys], color="#2563eb")
    plt.ylim(0, 1.05)
    plt.title("Final Holdout Test Performance (Untouched Test Set)")
    plt.tight_layout()
    plt.savefig("results/plots/07_final_performance.png", dpi=110)
    plt.close()


def print_expected_fits(models):
    """Explain the compute budget before the searches run."""
    grids = get_param_grids()
    total_grid = sum(grid_size(grids[n]) for n in models) * N_OUTER_FOLDS
    total_rand = sum(get_random_iter_counts()[n] for n in models) * N_OUTER_FOLDS
    print(f"Expected pipeline fits:")
    print(f"  - Baseline cross_validate: 5 folds x 5 models = 25 fits")
    print(f"  - GridSearchCV: {total_grid} fits (SVM includes calibration inner CV)")
    print(f"  - RandomizedSearchCV: {total_rand} fits")


def run_full_ml_workflow():
    """
    Full leakage-safe ML workflow: stratified split -> baseline CV -> grid + random
    hyperparameter searches -> best-model selection by CV F1 -> final eval on the
    untouched test set -> artifact + report saving.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    print("Loading raw data and running legacy preprocessing (leak-free fit on train)...")
    raw_df = load_raw_data("data/adult.csv")
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test, _, X_train_raw, X_test_raw = split_and_preprocess(cleaned_df)

    cv = get_shared_cv()
    calib_cv = get_shared_cv()
    models = get_tunable_models(calib_cv)

    print_expected_fits(models)

    print("\\nStep 1: Baseline cross-validation (5-fold stratified, Accuracy/Precision/Recall/F1/ROC-AUC)...")
    baseline_df = run_baseline_cv(models, X_train_raw, y_train, cv)
    baseline_df.to_csv("results/baseline_cv.csv", index=False)
    print(baseline_df.to_string())

    print("\\nStep 2: Hyperparameter optimization (GridSearchCV + RandomizedSearchCV)...")
    tuned_df, searches = run_tuning(models, X_train_raw, y_train, cv)
    tuned_df.to_csv("results/tuned_models_comparison.csv", index=False)
    print(tuned_df.to_string())

    # Persist every tuned pipeline (grid + random winners) as artifacts.
    for name in models:
        for method in ("grid", "random"):
            slug = MODEL_SLUGS[name]
            joblib.dump(
                searches[(name, method)].best_estimator_,
                f"models/best_pipeline_{method}_{slug}.joblib",
            )

    best_row, best_method = select_best_model(tuned_df)
    winner = searches[(best_row["Model"], best_method)]
    best_pipe = winner.best_estimator_
    print(f"\\nStep 3: Best model selected by cross-validated F1: {best_row['Model']} "
          f"({best_row['Search Method']}), Best CV F1 = {best_row['Best CV F1']}")

    joblib.dump(best_pipe, "models/best_pipeline.joblib")
    json_report = {
        "selected_model": best_row["Model"],
        "search_method": best_row["Search Method"],
        "best_params": winner.best_params_,
        "best_cv_f1": best_row["Best CV F1"],
        "cv_metrics": {
            "accuracy": best_row["CV Accuracy"],
            "precision": best_row["CV Precision"],
            "recall": best_row["CV Recall"],
            "roc_auc": best_row["CV ROC-AUC"],
        },
    }
    with open("results/best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    print("\\nStep 4: Final evaluation on the UNTOUCHED test set (no further tuning)...")
    final_metrics, cm, roc_data = evaluate_final(best_pipe, X_test_raw, y_test)
    final_df = pd.DataFrame([{"Model": best_row["Model"], **final_metrics}])
    final_df.to_csv("results/final_test_metrics.csv", index=False)
    print(final_df.to_string())

    with open("results/final_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected model: {best_row['Model']} ({best_row['Search Method']})\\n")
        f.write(classification_report(y_test, best_pipe.predict(X_test_raw), pos_label=1, zero_division=0))

    joblib.dump(cm, "results/final_confusion_matrix.joblib")
    joblib.dump({"fpr": roc_data[0], "tpr": roc_data[1], "auc": final_metrics["ROC-AUC"]}, "results/final_roc_curve.joblib")

    print("\\nStep 5: Saving visualizations...")
    save_plots(baseline_df, tuned_df, final_metrics, cm, roc_data)

    print(f"\\nDone. Best pipeline saved to models/best_pipeline.joblib")
    return best_row, final_metrics


''' + anchor
assert anchor in src
src = src.replace(anchor, new_funcs)

# --- 3. Update main() to call the new workflow ---
old_main_tail = '''    print("\\n" + "=" * 80)
    print("FINAL BENCHMARK RANKING TABLE (Sorted by F1 Score, then ROC-AUC):")
    print("=" * 80)
    print(metrics_df.to_string())
    print("=" * 80)
    print(f"Results successfully saved to {metrics_path}!")


if __name__ == "__main__":
    main()
'''
new_main_tail = '''    print("\\n" + "=" * 80)
    print("FINAL BENCHMARK RANKING TABLE (Sorted by F1 Score, then ROC-AUC):")
    print("=" * 80)
    print(metrics_df.to_string())
    print("=" * 80)
    print(f"Results successfully saved to {metrics_path}!")

    print("\\n" + "#" * 80)
    print("# Running the leakage-safe ML workflow (CV / hyperparameter search)...")
    print("#" * 80)
    run_full_ml_workflow()


if __name__ == "__main__":
    main()
'''
assert old_main_tail in src
src = src.replace(old_main_tail, new_main_tail)

open(P("src/train_models.py"), "w", encoding="utf-8").write(src)
print("train_models.py updated OK, new length:", len(src))
