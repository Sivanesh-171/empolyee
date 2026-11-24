"""Create src/model_tuning.py in the project dir."""
import sys
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir

D = project_dir()

CONTENT = '''"""
Hyperparameter search configuration and runners for the Adult Income ML workflow.

All searches operate on complete Scikit-learn Pipelines (preprocessing + model)
and use the shared StratifiedKFold splitter, scoring="f1", n_jobs=-1 and
refit=True so the fitted pipelines are ready for direct deployment.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def get_param_grids():
    """Exhaustive parameter grids for GridSearchCV, keyed by model name."""
    return {
        "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        "Decision Tree": {
            "classifier__max_depth": [5, 8, 10, 12, None],
            "classifier__min_samples_leaf": [5, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__criterion": ["gini", "entropy"],
        },
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "Support Vector Machine": {
            "classifier__estimator__C": [0.1, 1, 10, 100],
            "classifier__estimator__gamma": ["scale", 0.001, 0.01, 0.1],
            "classifier__estimator__kernel": ["rbf"],
        },
        "Naive Bayes": {"classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8]},
    }


def get_param_distributions():
    """Parameter distributions for RandomizedSearchCV, keyed by model name."""
    return {
        "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        "Decision Tree": {
            "classifier__max_depth": [5, 8, 10, 12, None, 16, 20],
            "classifier__min_samples_leaf": [5, 10, 20, 30, 50],
            "classifier__min_samples_split": [2, 5, 10, 20, 30],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
        },
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 13],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2, 3],
        },
        "Support Vector Machine": {
            "classifier__estimator__C": [0.1, 1, 10, 100],
            "classifier__estimator__gamma": ["scale", 0.001, 0.01, 0.1, 0.05, 0.5],
            "classifier__estimator__kernel": ["rbf"],
        },
        "Naive Bayes": {"classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
    }


def get_random_iter_counts():
    """n_iter budget for RandomizedSearchCV per model."""
    return {
        "Logistic Regression": 5,
        "Decision Tree": 20,
        "K-Nearest Neighbors": 10,
        "Support Vector Machine": 8,
        "Naive Bayes": 4,
    }


def grid_size(grid):
    """Total number of combinations in an exhaustive grid."""
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def run_grid_search(pipe, grid, cv, scoring="f1", n_jobs=-1):
    """Run a GridSearchCV on a complete pipeline; refits on the full training set."""
    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )


def run_randomized_search(pipe, dist, n_iter, cv, scoring="f1", n_jobs=-1, random_state=42):
    """Run a RandomizedSearchCV on a complete pipeline; refits on the full training set."""
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        random_state=random_state,
    )
'''

with open(f"{D}/src/model_tuning.py", "w", encoding="utf-8") as f:
    f.write(CONTENT)
print("wrote", f"{D}/src/model_tuning.py")
