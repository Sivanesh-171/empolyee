"""Modify src/preprocess.py in the project dir (see helpers.project_dir)."""
import sys
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir

D = project_dir()
P = lambda rel: f"{D}/{rel}"

src = open(P("src/preprocess.py"), encoding="utf-8").read()

old_imports = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
"""
new_imports = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
import joblib

# Canonical column groups shared by the legacy preprocessing and the ML pipelines.
NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def build_model_pipeline(model, scale_numeric=True):
    """
    Wrap a classifier with a leak-free preprocessing Pipeline (ColumnTransformer).

    Numeric features get a median imputer; StandardScaler is applied when the
    model requires scaled features (Logistic Regression, KNN, SVM).
    Categorical features get a constant imputer and one-hot encoding with
    ``handle_unknown="ignore"`` so unseen categories at predict time are safe.

    All transformers are fit inside the pipeline, so they are fit only on
    whatever data reaches the pipeline (e.g. the outer training fold during
    cross-validation) - no test data and no extra preprocessing pass is used.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLS),
            ("categorical", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
'''
assert old_imports in src
src = src.replace(old_imports, new_imports)

old_cols = """    # Define numeric and categorical columns
    numeric_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_cols = [
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]
"""
new_cols = """    # Define numeric and categorical columns (shared with the ML pipelines)
    numeric_cols = NUMERIC_COLS
    categorical_cols = CATEGORICAL_COLS
"""
assert old_cols in src
src = src.replace(old_cols, new_cols)

start = src.index("def transform_single_input")
end = src.index('if __name__ == "__main__":')
new_transform = '''def transform_single_input(raw_input_dict):
    """
    Build a single-row cleaned DataFrame ready for the fitted ML pipeline.

    Mirrors the cleaning applied in clean_data() so the pipeline receives
    exactly the same feature space it was fitted on (log1p on capital_gain /
    capital_loss, native_country grouped to 'United-States' vs 'Other').
    The pipeline's own transformers then handle imputation, scaling and
    one-hot encoding at predict time.
    """
    df_single = pd.DataFrame([raw_input_dict])

    # Apply the same variance-stabilizing transform used during training
    df_single["capital_gain"] = np.log1p(float(df_single["capital_gain"].iloc[0]))
    df_single["capital_loss"] = np.log1p(float(df_single["capital_loss"].iloc[0]))

    # Group native country the same way clean_data() does
    df_single["native_country"] = (
        "United-States" if df_single["native_country"].iloc[0] == "United-States" else "Other"
    )

    return df_single[NUMERIC_COLS + CATEGORICAL_COLS]


'''
src = src[:start] + new_transform + src[end:]

open(P("src/preprocess.py"), "w", encoding="utf-8").write(src)
print("preprocess.py updated OK, new length:", len(src))
