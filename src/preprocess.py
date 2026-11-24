"""
Data Preprocessing Pipeline for Adult Income Classification.
Handles loading, cleaning, feature engineering, and leak-free encoding/scaling
inside Scikit-learn Pipelines and ColumnTransformer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import joblib


# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = ["age", "education_num", "hours_per_week"]
CAPITAL_COLS = ["capital_gain", "capital_loss"]
CATEGORICAL_COLS = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
]
COUNTRY_COL = ["native_country"]
ALL_FEATURE_COLS = NUMERIC_COLS + CAPITAL_COLS + CATEGORICAL_COLS + COUNTRY_COL


# ---------------------------------------------------------------------------
# Custom Transformer Functions
# ---------------------------------------------------------------------------
def log1p_transform(X):
    """Apply log1p transformation to reduce positive skewness, clipping negative values."""
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def bin_native_country(X):
    """Bin native country into 'United-States' vs 'Other' to prevent sparse category noise."""
    if hasattr(X, "iloc"):
        return X.apply(
            lambda col: col.map(
                lambda v: "United-States" if str(v).strip() == "United-States" else "Other"
            )
        )
    arr = np.array(X, dtype=str)
    return np.where(arr == "United-States", "United-States", "Other")


# ---------------------------------------------------------------------------
# Data Ingestion & Cleaning
# ---------------------------------------------------------------------------
def load_raw_data(filepath="data/adult.csv"):
    """Step 1: Load raw adult dataset from CSV file and strip whitespace."""
    df = pd.read_csv(filepath)
    # Strip any whitespace around string entries
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


def clean_data(df):
    """
    Step 2: Clean column names, replace missing markers, drop duplicates and redundant features.
    Does NOT fit any transformers or perform leakage-prone operations.
    """
    df = df.copy()

    # Standardize column names (replace dots with underscores)
    df.columns = [col.replace(".", "_") for col in df.columns]

    # Replace "?" missing value markers with "Unknown"
    df = df.replace("?", "Unknown")

    # Drop duplicate records (24 identical rows)
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop fnlwgt (census population weight, not personal trait) and redundant education column
    df = df.drop(columns=["fnlwgt", "education"])

    # Encode target income if present: <=50K is 0, >50K is 1
    if "income" in df.columns:
        df["income"] = df["income"].apply(lambda val: 1 if ">50K" in str(val) else 0)

    return df


# ---------------------------------------------------------------------------
# Preprocessing Pipeline Construction
# ---------------------------------------------------------------------------
def get_preprocessor():
    """
    Build a leak-free ColumnTransformer that performs all feature scaling,
    logarithmic transformations, missing value imputation, and one-hot encoding.
    This preprocessor is designed to be embedded directly into a Scikit-learn Pipeline.
    """
    # Numerical pipeline for general numeric features (age, education_num, hours_per_week)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Capital pipeline for heavily skewed capital gains and losses
    cap_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_transform, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline for nominal features
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Native country pipeline with automatic grouping and one-hot encoding
    country_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("bin_country", FunctionTransformer(bin_native_country, feature_names_out="one-to-one")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_COLS),
            ("cap", cap_pipeline, CAPITAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
            ("country", country_pipeline, COUNTRY_COL),
        ],
        remainder="drop",
    )

    return preprocessor


def create_pipeline(classifier):
    """Combine the leak-free ColumnTransformer and a classifier into a single Scikit-learn Pipeline."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Train-Test Splitting (Leak-Free)
# ---------------------------------------------------------------------------
def split_data(df, test_size=0.20, random_state=42):
    """
    Perform a strict stratified train/test split.
    The test split must remain completely untouched during baseline evaluation,
    cross-validation, and hyperparameter optimization.
    """
    X = df.drop(columns=["income"])
    y = df["income"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Backward Compatibility Helpers
# ---------------------------------------------------------------------------
def split_and_preprocess(df, test_size=0.2, random_state=42):
    """Provided for backward compatibility with earlier scripts."""
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )
    preprocessor = get_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, preprocessor, X_train_raw, X_test_raw


def transform_single_input(raw_input_dict, preprocessor=None):
    """Transform a single user input dictionary using the preprocessor."""
    df_single = pd.DataFrame([raw_input_dict])
    if preprocessor is None:
        preprocessor = get_preprocessor()
    if hasattr(preprocessor, "transform"):
        return preprocessor.transform(df_single)
    return preprocessor.fit_transform(df_single)


if __name__ == "__main__":
    print("Testing pipeline preprocessing module...")
    raw = load_raw_data("data/adult.csv")
    cleaned = clean_data(raw)
    X_tr, X_te, y_tr, y_te = split_data(cleaned)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    pipe = create_pipeline(None)
    X_tr_proc = pipe.named_steps["preprocessor"].fit_transform(X_tr)
    X_te_proc = pipe.named_steps["preprocessor"].transform(X_te)
    print(f"Preprocessed train matrix: {X_tr_proc.shape}, test matrix: {X_te_proc.shape}")
    print("Pipeline preprocessing setup successfully verified!")
