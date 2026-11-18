"""
Data Preprocessing Pipeline for Adult Income Classification.
Handles loading, cleaning, feature engineering, and leak-free encoding/scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


def load_raw_data(filepath="data/adult.csv"):
    """Step 1: Load raw adult dataset from CSV file."""
    df = pd.read_csv(filepath)
    # Strip any whitespace around string entries
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


def clean_data(df):
    """Step 2: Clean column names, replace missing markers, drop duplicates and redundant features."""
    df = df.copy()

    # Standardize column names (replace dots with underscores)
    df.columns = [col.replace(".", "_") for col in df.columns]

    # Replace "?" missing value markers with "Unknown"
    df = df.replace("?", "Unknown")

    # Drop duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop fnlwgt (census weight, not personal attribute) and redundant education column
    df = df.drop(columns=["fnlwgt", "education"])

    # Group native_country into 'United-States' vs 'Other'
    df["native_country"] = df["native_country"].apply(
        lambda country: "United-States" if country == "United-States" else "Other"
    )

    # Apply log1p transformation to capital_gain and capital_loss to reduce skewness
    df["capital_gain"] = np.log1p(df["capital_gain"])
    df["capital_loss"] = np.log1p(df["capital_loss"])

    # Encode target income: <=50K is 0, >50K is 1
    if "income" in df.columns:
        df["income"] = df["income"].apply(lambda val: 1 if ">50K" in str(val) else 0)

    return df


def split_and_preprocess(df, test_size=0.2, random_state=42):
    """Step 3: Perform stratified train/test split and leak-free scaling + one-hot encoding."""
    # Separate features and target
    X = df.drop(columns=["income"])
    y = df["income"].values

    # Perform stratified train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Define numeric and categorical columns
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

    # Initialize transformers
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Fit transformers ONLY on training split
    scaler.fit(X_train_raw[numeric_cols])
    encoder.fit(X_train_raw[categorical_cols])

    # Transform both train and test splits
    X_train_num = scaler.transform(X_train_raw[numeric_cols])
    X_test_num = scaler.transform(X_test_raw[numeric_cols])

    X_train_cat = encoder.transform(X_train_raw[categorical_cols])
    X_test_cat = encoder.transform(X_test_raw[categorical_cols])

    # Combine scaled numeric features and one-hot categorical features
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    # Extract all feature names
    cat_feature_names = list(encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    # Bundle preprocessor components for later single-sample prediction
    preprocessor = {
        "scaler": scaler,
        "encoder": encoder,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_feature_names": all_feature_names,
    }

    return X_train, X_test, y_train, y_test, preprocessor, X_train_raw, X_test_raw


def transform_single_input(raw_input_dict, preprocessor):
    """Step 4: Transform a single user input dictionary using the fitted preprocessor."""
    scaler = preprocessor["scaler"]
    encoder = preprocessor["encoder"]
    numeric_cols = preprocessor["numeric_cols"]
    categorical_cols = preprocessor["categorical_cols"]

    # Convert dictionary to DataFrame
    df_single = pd.DataFrame([raw_input_dict])

    # Apply log1p on capital gain and loss
    df_single["capital_gain"] = np.log1p(float(df_single["capital_gain"].iloc[0]))
    df_single["capital_loss"] = np.log1p(float(df_single["capital_loss"].iloc[0]))

    # Group native country
    df_single["native_country"] = (
        "United-States" if df_single["native_country"].iloc[0] == "United-States" else "Other"
    )

    # Transform numeric and categorical
    num_vals = scaler.transform(df_single[numeric_cols])
    cat_vals = encoder.transform(df_single[categorical_cols])

    # Combine into single feature vector
    return np.hstack([num_vals, cat_vals])


if __name__ == "__main__":
    print("Executing preprocessing pipeline...")
    # Load and clean raw data
    raw_df = load_raw_data("data/adult.csv")
    cleaned_df = clean_data(raw_df)

    # Split and preprocess
    X_train, X_test, y_train, y_test, preprocessor, _, _ = split_and_preprocess(cleaned_df)

    # Save fitted preprocessor
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print(f"Preprocessed train shape: {X_train.shape}")
    print(f"Preprocessed test shape: {X_test.shape}")
    print(f"Total features created: {len(preprocessor['all_feature_names'])}")
    print("Preprocessor saved to models/preprocessor.joblib successfully!")

