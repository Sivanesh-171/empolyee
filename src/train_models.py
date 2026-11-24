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

from preprocess import load_raw_data, clean_data, split_data
from evaluate_baseline import run_baseline_cross_validation
from tune_models import run_grid_and_random_search, select_and_evaluate_final_models


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
