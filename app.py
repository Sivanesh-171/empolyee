"""
Adult Income Classification - Interactive Machine Learning Web Application
Built with Streamlit, Scikit-Learn, and Plotly.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure current and src directory are accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# pyrefly: ignore[missing-import]
from preprocess import load_raw_data, clean_data, transform_single_input


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Adult Income ML Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Caching Functions for Data & Models
# ---------------------------------------------------------
@st.cache_data
def get_raw_and_cleaned_data():
    """Load raw dataset and produce cleaned version for EDA and Preprocessing."""
    raw = load_raw_data("data/adult.csv")
    cleaned = clean_data(raw)
    return raw, cleaned


@st.cache_data
def get_metrics():
    """Load precomputed metrics and ranking table."""
    metrics_df = pd.read_csv("results/metrics.csv", index_col="Rank")
    return metrics_df


@st.cache_data
def get_baseline_metrics():
    """Load baseline 5-fold cross-validation metrics table."""
    if os.path.exists("results/baseline_cv_metrics.csv"):
        return pd.read_csv("results/baseline_cv_metrics.csv", index_col="Baseline Rank")
    return None


@st.cache_data
def get_tuning_metrics():
    """Load GridSearchCV vs RandomizedSearchCV comparative metrics table."""
    if os.path.exists("results/tuned_model_comparison.csv"):
        return pd.read_csv("results/tuned_model_comparison.csv")
    return None


@st.cache_data
def get_best_hyperparameters():
    """Load JSON metadata of optimal hyperparameters for all models."""
    if os.path.exists("results/best_hyperparameters.json"):
        import json
        with open("results/best_hyperparameters.json", "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_confusion_matrices():
    """Load precomputed confusion matrices."""
    return joblib.load("results/confusion_matrices.joblib")


@st.cache_data
def get_roc_curves():
    """Load precomputed ROC curves for all models."""
    return joblib.load("results/roc_curves.joblib")


@st.cache_data
def get_pr_metrics():
    """Load precomputed Precision-Recall metrics table."""
    if os.path.exists("results/precision_recall_metrics.csv"):
        return pd.read_csv("results/precision_recall_metrics.csv")
    return None


@st.cache_data
def get_pr_curves():
    """Load precomputed Precision-Recall curves data."""
    if os.path.exists("results/pr_curves.joblib"):
        return joblib.load("results/pr_curves.joblib")
    return None


@st.cache_data
def get_cv_fold_scores():
    """Load fold-wise cross-validation scores."""
    if os.path.exists("results/cv_fold_scores.csv"):
        return pd.read_csv("results/cv_fold_scores.csv")
    return None


@st.cache_data
def get_grid_cv_candidates(model_slug):
    """Load full GridSearchCV candidate evaluations for a specific model."""
    p = f"results/{model_slug}_grid_cv_results.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


@st.cache_resource
def load_all_models():
    """Load all 5 pre-trained tuned model pipelines from models directory."""
    models = {
        "Support Vector Machine": joblib.load("models/svm.joblib"),
        "Decision Tree": joblib.load("models/decision_tree.joblib"),
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "K-Nearest Neighbors": joblib.load("models/knn.joblib"),
        "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    }
    return models


@st.cache_resource
def load_best_pipeline():
    """Load the overall winning trained Scikit-learn Pipeline."""
    if os.path.exists("models/best_pipeline.joblib"):
        return joblib.load("models/best_pipeline.joblib")
    return joblib.load("models/svm.joblib")


@st.cache_resource
def load_fitted_preprocessor():
    """Load fitted preprocessor bundle for backward compatibility."""
    if os.path.exists("models/preprocessor.joblib"):
        return joblib.load("models/preprocessor.joblib")
    return None


# ---------------------------------------------------------
# Custom Styling for Responsiveness & Cards
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2563eb;
        margin-bottom: 15px;
    }
    .insight-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #1e3a8a;
        margin-top: -10px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .best-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Page 1: Home
# ---------------------------------------------------------
def render_home(raw_df, cleaned_df, metrics_df):
    st.title("💼 Adult Income Classification Dashboard")
    st.subheader("Predicting Individual Income (> $50K vs ≤ $50K) Using 1994 US Census Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### Project Overview
            Welcome to the **Adult Income Classification** machine learning platform.
            This project provides an end-to-end data science lifecycle demonstration:
            - **Complete Exploratory Data Analysis (EDA)** with in-depth insights into income disparity factors.
            - **Leak-Free Preprocessing Pipeline** with outlier handling, logarithmic transformations, and encoding.
            - **5 Supervised Machine Learning Models** benchmarked strictly on the identical train/test split.
            - **Interactive Model Comparison** with ROC curves, confusion matrices, and overfitting metrics.
            - **Interactive Live Predictor** to test custom demographic profiles with pre-trained models.
            """
        )

        st.info(
            "📌 **Core ML Problem**: Binary classification on an imbalanced dataset (~76% earn ≤ $50K, ~24% earn > $50K). "
            "Because of class imbalance, models are evaluated primarily on **F1-Score** and **ROC-AUC** rather than raw accuracy alone."
        )

    with col2:
        st.markdown("### 📊 Dataset Fast Facts")
        st.metric("Total Records", f"{len(raw_df):,}")
        st.metric("Raw Features", f"{raw_df.shape[1]}")
        st.metric("Duplicate Rows Dropped", f"{raw_df.duplicated().sum()}")
        st.metric("Cleaned Preprocessed Features", "51")

    st.divider()

    st.markdown("### 🧭 How to Navigate This App")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**📊 EDA Explorer**")
        st.caption("Inspect distributions, data quality, correlation, and feature vs target patterns.")
    with c2:
        st.markdown("**⚙️ Preprocessing**")
        st.caption("Review step-by-step cleaning, log transforms, and before/after sample data.")
    with c3:
        st.markdown("**🏆 Model Comparison**")
        st.caption("Compare 5 algorithms, ROC curves, confusion matrices, and runtime costs.")
    with c4:
        st.markdown("**🔮 Live Predictor**")
        st.caption("Enter personal attributes and predict income tier with confidence scores.")
    with c5:
        st.markdown("**📝 Analysis**")
        st.caption("Detailed breakdown of why the best model won and the worst struggled.")


# ---------------------------------------------------------
# Page 2: EDA Explorer
# ---------------------------------------------------------
def render_eda(raw_df, cleaned_df):
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Examine data patterns, distributions, quality markers, and demographic drivers of income.")

    # Global Filter Toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 EDA Filter Controls")
    filter_sex = st.sidebar.multiselect("Filter by Sex:", options=["All", "Male", "Female"], default="All")
    filter_income = st.sidebar.multiselect("Filter by Income:", options=["All", "<=50K", ">50K"], default="All")

    # --- DataFrames with separate responsibilities ---------------------------------
    # (1) eda_df: VISUALISATION frame. Raw data with all selected filters (sex + income)
    #     applied. It represents the population the user asked to inspect and drives
    #     counts/distributions/target balance.
    eda_df = raw_df.copy()
    if "All" not in filter_sex and filter_sex:
        eda_df = eda_df[eda_df["sex"].isin(filter_sex)]
    if "All" not in filter_income and filter_income:
        eda_df = eda_df[eda_df["income"].isin(filter_income)]

    # (2) rate_calc_df: INCOME-RATE frame. Only demographic filters (sex) are applied,
    #     so both income classes stay present. The income filter is deliberately NOT
    #     applied here: selecting a single class would mathematically force every
    #     category's ">50K rate" to 100% (or 0%), which is an artefact, not a real rate.
    rate_calc_df = raw_df.copy()
    if "All" not in filter_sex and filter_sex:
        rate_calc_df = rate_calc_df[rate_calc_df["sex"].isin(filter_sex)]

    # (3) modeling_df: CLEANED (model-ready) frame for modelling-oriented views.
    #     clean_data() uses underscore column names and encodes income as 0/1.
    #     The same filters are applied so the view stays interactive.
    modeling_df = cleaned_df.copy()
    if "All" not in filter_sex and filter_sex:
        modeling_df = modeling_df[modeling_df["sex"].isin(filter_sex)]
    if "All" not in filter_income and filter_income:
        income_codes = {"<=50K": 0, ">50K": 1}
        selected_codes = [income_codes[val] for val in filter_income if val in income_codes]
        if selected_codes:
            modeling_df = modeling_df[modeling_df["income"].isin(selected_codes)]

    # Duplicate count is derived from the raw data, never hard-coded.
    duplicate_count = raw_df.duplicated().sum()

    def _warn_if_empty(df, section_label):
        """Show a warning and report True when a filtered frame has no rows."""
        if df.empty:
            st.warning(
                f"⚠️ No records match the current filters for **{section_label}**. "
                "Adjust the sidebar filters to display this view."
            )
            return True
        return False

    eda_tabs = st.tabs([
        "1. Overview",
        "2. Data Quality",
        "3. Target Balance",
        "4. Numeric Distributions",
        "5. Categorical Counts",
        "6. Features vs Target",
        "7. Feature Correlations",
        "8. Multi-Feature Views",
        "9. Key Findings",
    ])

    # Tab 1: Overview
    with eda_tabs[0]:
        st.subheader("1. Dataset Overview & Summary Statistics")
        st.caption("Source: **raw dataset** with the current sidebar filters applied.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(eda_df):,}")
        c2.metric("Columns", f"{eda_df.shape[1]}")
        c3.metric("Memory Usage", f"{eda_df.memory_usage().sum() / 1024:.1f} KB")

        if not _warn_if_empty(eda_df, "Dataset Overview"):
            st.markdown("#### Sample Records (First 5 Rows)")
            st.dataframe(eda_df.head(), use_container_width=True)

            st.markdown("#### Summary Statistics (Numeric Features)")
            st.dataframe(eda_df.describe().round(2), use_container_width=True)
            # Descriptive statistics are computed from the filtered frame so the
            # explanation stays correct when filters change.
            avg_age = eda_df["age"].mean()
            med_hours = eda_df["hours.per.week"].median()
            gain_zero = (eda_df["capital.gain"] == 0).mean() * 100
            loss_zero = (eda_df["capital.loss"] == 0).mean() * 100
            st.markdown(
                f'<div class="insight-box">💡 <b>Insight:</b> In the selected population the average age is {avg_age:.1f} years and median weekly hours are {med_hours:.0f}. Capital gain ({gain_zero:.1f}% zeros) and capital loss ({loss_zero:.1f}% zeros) are strongly zero-inflated. These are descriptive summaries of the filtered sample.</div>',
                unsafe_allow_html=True,
            )

    # Tab 2: Data Quality
    with eda_tabs[1]:
        st.subheader("2. Data Quality & Missing Value Assessment")
        st.caption("Source: **raw dataset** (unfiltered — quality checks intentionally ignore the sidebar filters).")
        col_q1, col_q2 = st.columns(2)

        with col_q1:
            # Count "?" markers per column (raw data only; cleaned data has no '?')
            q_counts = (raw_df == "?").sum()
            q_counts = q_counts[q_counts > 0].reset_index()
            q_counts.columns = ["Feature", "Missing ('?') Count"]
            q_counts["Missing %"] = (q_counts["Missing ('?') Count"] / len(raw_df) * 100).round(2)

            if not _warn_if_empty(q_counts, "missing-value assessment"):
                fig_q = px.bar(
                    q_counts,
                    x="Feature",
                    y="Missing ('?') Count",
                    text="Missing %",
                    title="Missing Values ('?') by Column",
                    color="Missing ('?') Count",
                    color_continuous_scale="Reds",
                )
                fig_q.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_q.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_q, use_container_width=True)
                q_summary = ", ".join(
                    f"{row['Feature']} ({row['Missing %']:.2f}%)"
                    for _, row in q_counts.sort_values("Missing %", ascending=False).iterrows()
                )
                st.markdown(
                    f'<div class="insight-box">💡 <b>Insight:</b> Missingness is concentrated in {len(q_counts)} categorical features: {q_summary}. The pipeline imputes these markers as "Unknown"; that is a modelling choice, not a claim about why the values are missing.</div>',
                    unsafe_allow_html=True,
                )

        with col_q2:
            st.markdown("#### Data Quality Findings")
            if raw_df.empty:
                st.warning("⚠️ No raw records are available for data-quality checks.")
            else:
                gain_zero_pct = (raw_df["capital.gain"] == 0).mean() * 100
                st.write(f"- **Duplicate Rows**: Found **{duplicate_count:,} identical duplicate rows** in the raw census records (computed dynamically).")
                st.write(f"- **Null Values (NaN)**: {int(raw_df.isna().sum().sum())} (missingness is coded as string `'?'`).")
                st.write("- **Redundant Feature**: `education` (16 categories) maps 1-to-1 with `education.num` (integers 1–16).")
                st.write("- **Uninformative Feature**: `fnlwgt` (census population sampling weight) is administrative, not a personal attribute, so it should not be read as a driver of income.")
                st.write(f"- **Potential Outliers**: `capital.gain` reaches {int(raw_df['capital.gain'].max()):,} while {gain_zero_pct:.1f}% of values are 0. Extremes are flagged for inspection, but extreme values are not automatically data errors.")

    # Tab 3: Target Balance
    with eda_tabs[2]:
        st.subheader("3. Target Distribution: Income Class Balance")
        st.caption("Source: **raw dataset** filtered by the sidebar selection — this is the population currently under inspection.")
        if not _warn_if_empty(eda_df, "Target Distribution"):
            income_counts = eda_df["income"].value_counts().reset_index()
            income_counts.columns = ["Income Class", "Count"]
            income_counts["Percentage"] = (income_counts["Count"] / len(eda_df) * 100).round(2)

            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                fig_target = px.pie(
                    income_counts,
                    names="Income Class",
                    values="Count",
                    hole=0.45,
                    title="Target Class Balance (<=50K vs >50K)",
                    color="Income Class",
                    color_discrete_map={"<=50K": "#64748b", ">50K": "#2563eb"},
                )
                st.plotly_chart(fig_target, use_container_width=True)
                high_pct = eda_df["income"].astype(str).str.contains(">50K").mean() * 100
                balance_note = (
                    " This reflects the active income filter, not the natural class balance of the full dataset."
                    if "All" not in filter_income and filter_income
                    else " The full dataset is imbalanced (~3:1)."
                )
                st.markdown(
                    f'<div class="insight-box">💡 <b>Insight:</b> In the selected population, {100 - high_pct:.1f}% earn ≤50K and {high_pct:.1f}% earn >50K.{balance_note}</div>',
                    unsafe_allow_html=True,
                )

            with col_t2:
                st.markdown("#### Why Accuracy Alone is Misleading")
                st.warning(
                    "⚠️ **The Accuracy Paradox**: On the full (unfiltered) dataset, a naive baseline model that predicts "
                    "every single applicant earns **'<=50K'** would immediately score **~75.9% accuracy**, "
                    "despite failing to identify a single high-earning individual (Recall = 0.0, F1 = 0.0)!\n\n"
                    "Therefore, in Phase 3 we prioritize **F1-Score (for >50K)** and **ROC-AUC** to fairly evaluate "
                    "genuine predictive capability on the minority class."
                )

    # Tab 4: Numeric Distributions
    with eda_tabs[3]:
        st.subheader("4. Numeric Feature Distributions & Skewness")
        st.caption("Charts: **raw dataset** with current filters. Notes below describe the full raw dataset.")
        num_options = ["age", "education.num", "hours.per.week", "capital.gain", "capital.loss", "fnlwgt"]
        selected_num = st.selectbox("Select Numeric Feature to Inspect:", num_options, index=0)

        if not _warn_if_empty(eda_df, "Numeric Distributions"):
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                fig_hist = px.histogram(
                    eda_df,
                    x=selected_num,
                    nbins=40,
                    marginal="box",
                    title=f"Histogram of {selected_num}",
                    color_discrete_sequence=["#3b82f6"],
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_n2:
                fig_box = px.box(
                    eda_df,
                    y=selected_num,
                    title=f"Boxplot of {selected_num}",
                    color_discrete_sequence=["#2563eb"],
                )
                st.plotly_chart(fig_box, use_container_width=True)

        insights_num = {
            "age": "Age spans 17 to 90 years with a right-skewed peak around 30–45 years, reflecting active career stages.",
            "education.num": "Education years cluster heavily around 9 (High School grad), 10 (Some college), and 13 (Bachelors).",
            "hours.per.week": "A pronounced spike exists at exactly 40 hours/week (standard full-time workweek) with long tails.",
            "capital.gain": "Capital gain is zero-inflated (91.7% are $0) with extremes up to $99,999. These may be genuine or capped values rather than errors; their skew motivates the log1p transform in preprocessing.",
            "capital.loss": "Capital loss is also zero-inflated (95.3% are $0); its skew motivates the log1p transform used to stabilise variance.",
            "fnlwgt": "This is a census sampling weight (administrative metadata), not a personal attribute. It has no clear conceptual link to an individual's earnings and is treated as metadata, not a cause of income.",
        }
        st.markdown(
            f'<div class="insight-box">💡 <b>Insight:</b> {insights_num.get(selected_num, "")}</div>',
            unsafe_allow_html=True,
        )

    # Tab 5: Categorical Counts
    with eda_tabs[4]:
        st.subheader("5. Categorical Feature Distributions")
        st.caption("Source: **raw dataset** with current filters. Notes describe the full raw dataset.")
        cat_options = [
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
        ]
        selected_cat = st.selectbox("Select Categorical Feature:", cat_options, index=0)

        cat_counts = eda_df[selected_cat].value_counts().reset_index()
        cat_counts.columns = [selected_cat, "Count"]

        if not _warn_if_empty(cat_counts, "Categorical Distributions"):
            fig_cat = px.bar(
                cat_counts,
                x=selected_cat,
                y="Count",
                title=f"Category Frequencies: {selected_cat}",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig_cat.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)

        insights_cat = {
            "workclass": "Private sector employment dominates the workforce (~74%), while without-pay and never-worked are negligible.",
            "education": "High school graduates (HS-grad) form the largest segment, followed by Some-college and Bachelors degrees.",
            "marital.status": "Married-civ-spouse and Never-married account for over 80% of all observed marital records.",
            "occupation": "Exec-managerial, Prof-specialty, and Craft-repair are the top three most frequent job roles.",
            "relationship": "Husband is the most common role (~40%), followed closely by Not-in-family and Own-child.",
            "race": "White individuals represent ~85% of sample observations, with Black (~10%) and Asian (~3%) following.",
            "sex": "Males comprise ~67% of the census dataset sample, reflecting 1994 labor force participation patterns.",
            "native.country": "Over 89% of participants are from the United-States; all foreign nations have sparse individual counts, justifying binary grouping.",
        }
        st.markdown(
            f'<div class="insight-box">💡 <b>Insight:</b> {insights_cat.get(selected_cat, "")}</div>',
            unsafe_allow_html=True,
        )

    # Tab 6: Features vs Target
    with eda_tabs[5]:
        st.subheader("6. Demographic Drivers vs Income Tier")
        col_vt1, col_vt2 = st.columns(2)

        with col_vt1:
            st.markdown("#### Numeric Features by Income Class")
            st.caption("Source: **raw dataset** with current filters applied.")
            num_v = st.selectbox("Select Numeric Variable:", ["age", "education.num", "hours.per.week"], index=0)
            if not _warn_if_empty(eda_df, "Numeric Features by Income Class"):
                fig_v = px.box(
                    eda_df,
                    x="income",
                    y=num_v,
                    color="income",
                    title=f"{num_v} Distribution by Income Class",
                    color_discrete_map={"<=50K": "#64748b", ">50K": "#2563eb"},
                )
                st.plotly_chart(fig_v, use_container_width=True)
                medians = eda_df.groupby("income")[num_v].median()
                if "<=50K" in medians.index and ">50K" in medians.index:
                    st.markdown(
                        f'<div class="insight-box">💡 <b>Insight:</b> In the selected population the median {num_v} is {medians[">50K"]:.1f} for >50K earners versus {medians["<=50K"]:.1f} for ≤50K earners. This is an observed association in this sample, not evidence that this feature causes higher income.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        "Both income classes are needed for a meaningful comparison — adjust the income filter in the sidebar."
                    )

        with col_vt2:
            st.markdown("#### Percentage Earning >50K by Categorical Feature")
            st.caption(
                "Rates use **both income classes** of the raw data with demographic (sex) filters preserved. "
                "The income filter is deliberately excluded here — applying it would force every rate to 100% or 0%."
            )
            cat_v = st.selectbox(
                "Select Categorical Variable:",
                ["education", "occupation", "marital.status", "relationship", "sex", "workclass"],
                index=0,
            )
            if not _warn_if_empty(rate_calc_df, "Categorical Income Rates"):
                rate_df = (
                    rate_calc_df.groupby(cat_v)["income"]
                    .apply(lambda s: s.astype(str).str.contains(">50K").mean() * 100)
                    .reset_index()
                )
                rate_df.columns = [cat_v, "High Income Rate (%)"]
                rate_df = rate_df.sort_values(by="High Income Rate (%)", ascending=False)

                fig_rate = px.bar(
                    rate_df,
                    x=cat_v,
                    y="High Income Rate (%)",
                    title=f"% Earning >50K by {cat_v}",
                    color="High Income Rate (%)",
                    color_continuous_scale="Viridis",
                )
                fig_rate.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_rate, use_container_width=True)
                st.markdown(
                    f'<div class="insight-box">💡 <b>Insight:</b> Categories at the top of this chart have a higher <i>observed</i> share of >50K earners within {cat_v}. This is an unadjusted association in the 1994 sample: it does not prove causation, and a group-level rate is not a prediction for any individual.</div>',
                    unsafe_allow_html=True,
                )

    # Tab 7: Feature Correlations & Overlap
    with eda_tabs[6]:
        st.subheader("7. Numeric Correlation Heatmap & Categorical Overlap")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### Pearson Correlation Heatmap")
            st.caption(
                "Source: **cleaned, model-ready dataset** (numeric features with log-transformed capital columns). "
                "Correlation measures linear association only — it does not imply causation."
            )
            # Modelling-oriented view: use the cleaned frame (underscore column names,
            # log-transformed capital features) rather than the raw visualisation frame.
            numeric_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
            if not _warn_if_empty(modeling_df, "Correlation Heatmap"):
                corr = modeling_df[numeric_cols].corr()

                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Matrix of Numeric Features (Cleaned Data)",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown(
                    '<div class="insight-box">💡 <b>Insight:</b> In the full cleaned dataset all pairwise linear correlations are weak (|r| ≤ 0.15); the strongest is between education_num and hours_per_week (r ≈ 0.15). Weak linear correlation does not rule out non-linear relationships, and correlation never establishes causation.</div>',
                    unsafe_allow_html=True,
                )

        with col_c2:
            st.markdown("#### Redundancy: Education vs Education.num")
            st.caption("Source: **raw dataset** with current filters applied.")
            if not _warn_if_empty(eda_df, "Education Mapping"):
                edu_map = eda_df[["education", "education.num"]].drop_duplicates().sort_values("education.num")
                st.dataframe(edu_map.reset_index(drop=True), use_container_width=True)
                st.markdown(
                    '<div class="insight-box">💡 <b>Insight:</b> There is a strict 1-to-1 monotonic mapping between "education" and "education.num". Keeping only the numeric version preserves the same ordering while removing a redundant duplicate feature.</div>',
                    unsafe_allow_html=True,
                )

    # Tab 8: Multi-Feature Views
    with eda_tabs[7]:
        st.subheader("8. Multi-Feature Interactions")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### Age vs. Hours Per Week (Colored by Income)")
            st.caption("Source: **raw dataset** with current filters applied (random sample for readability).")
            if not _warn_if_empty(eda_df, "Age vs Hours Scatter"):
                sample_df = eda_df.sample(n=min(2500, len(eda_df)), random_state=42)
                sample_size = len(sample_df)
                fig_scatter = px.scatter(
                    sample_df,
                    x="age",
                    y="hours.per.week",
                    color="income",
                    title=f"Age vs. Hours/Week (Sampled N={sample_size:,})",
                    opacity=0.6,
                    color_discrete_map={"<=50K": "#94a3b8", ">50K": "#2563eb"},
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown(
                    '<div class="insight-box">💡 <b>Insight:</b> In this sample, >50K earners are concentrated around ages 35–55 and 40–60 weekly hours, and are rare among workers under 25 or below 30 hours. This is a visual association in the sampled data, not a causal rule or an individual prediction.</div>',
                    unsafe_allow_html=True,
                )

        with col_m2:
            st.markdown("#### High Income Rate by Sex and Education Level")
            st.caption(
                "Rates use **both income classes** with demographic filters preserved; the income filter is excluded "
                "so the percentages remain the true observed rates rather than 100%/0% artefacts."
            )
            edu_order = [
                "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
                "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
                "Prof-school", "Doctorate",
            ]
            if not _warn_if_empty(rate_calc_df, "Income Rate by Sex & Education"):
                rate_sex_edu = (
                    rate_calc_df.groupby(["education", "sex"])["income"]
                    .apply(lambda s: s.astype(str).str.contains(">50K").mean() * 100)
                    .reset_index()
                )
                rate_sex_edu.columns = ["education", "sex", "High Income Rate (%)"]

                fig_sex_edu = px.bar(
                    rate_sex_edu,
                    x="education",
                    y="High Income Rate (%)",
                    color="sex",
                    barmode="group",
                    category_orders={"education": edu_order},
                    title="Income Rate (%) across Sex & Education",
                    color_discrete_map={"Male": "#2563eb", "Female": "#ec4899"},
                )
                fig_sex_edu.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_sex_edu, use_container_width=True)

                # Compare rates only when both sexes are present under the current filter.
                pivot = rate_sex_edu.pivot(index="education", columns="sex", values="High Income Rate (%)")
                if {"Male", "Female"}.issubset(pivot.columns):
                    higher_count = int((pivot["Male"] > pivot["Female"]).sum())
                    tier_count = len(pivot)
                    sex_note = (
                        f"Males show a higher observed rate in {higher_count} of {tier_count} education tiers."
                    )
                else:
                    sex_note = "Select both sexes to compare their observed rates by education."
                st.markdown(
                    f'<div class="insight-box">💡 <b>Insight:</b> In the full 1994 sample, higher degrees are associated with higher observed >50K rates for both sexes. {sex_note} These are <i>unadjusted</i> group rates reflecting historical labour-market and demographic patterns — not a causal effect of sex or education, and not a prediction for any individual.</div>',
                    unsafe_allow_html=True,
                )

    # Tab 9: Summary & Decisions
    with eda_tabs[8]:
        st.subheader("9. Summary of Key EDA Findings & Justified Preprocessing")
        st.markdown(
            f"""
            | # | EDA Observation | Preprocessing Decision Justified |
            |---|---|---|
            | **1** | Missing values are coded as `'?'` in 3 categorical features | Replaced with `'Unknown'` category so models retain all {len(raw_df):,} records without loss |
            | **2** | {duplicate_count:,} duplicate rows detected in census data | Dropped duplicates to prevent memorization and redundant sample weighting |
            | **3** | `education` has exact 1-to-1 mapping with `education.num` | Dropped string column `education` and kept `education_num` to remove redundancy |
            | **4** | `fnlwgt` is a census demographic sampling weight | Dropped `fnlwgt` because it is administrative metadata, not a personal attribute; it should not be used to explain or predict individual earnings |
            | **5** | Over 89% of participants are from the United States | Grouped `native_country` into `'United-States'` vs `'Other'` to avoid high-cardinality noise |
            | **6** | `capital.gain` and `capital.loss` are 90%+ zeros and heavily right-skewed | Applied `log1p` transformation to compress extreme ranges and stabilize variance |
            | **7** | Target ratio is ~76% / 24% (imbalanced) | Used stratified 80/20 train/test split; benchmarked models using F1-score & ROC-AUC |
            | **8** | Categorical features have diverse cardinalities | Applied One-Hot Encoding fitted only on the training set to prevent data leakage |
            | **9** | Numeric features have varied scales (age ~17-90, hours ~1-99) | Applied `StandardScaler` fitted only on training set for distance/linear models |
            """
        )


# ---------------------------------------------------------
# Page 3: Preprocessing
# ---------------------------------------------------------
def render_preprocessing(raw_df, cleaned_df):
    st.title("⚙️ Shared Preprocessing Pipeline")
    st.write("Inspect how the raw data was transformed before feeding all 5 models.")

    st.markdown(
        """
        ### Unified Leak-Free Design
        To guarantee a completely fair benchmark, all 5 models are trained on the **exact same preprocessed matrix**:
        - Transformers (`StandardScaler`, `OneHotEncoder`) were **fitted strictly on the training partition (80%)** and then applied to test data (20%).
        - Scaling numeric features is mandatory for **Logistic Regression**, **KNN**, and **SVM (RBF)**, and mathematically harmless for **Decision Trees**.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Sample (Before)")
        st.caption(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        st.dataframe(raw_df.head(6), use_container_width=True)

    with col2:
        st.subheader("Cleaned & Engineered Sample (After)")
        st.caption(f"Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
        st.dataframe(cleaned_df.head(6), use_container_width=True)

    st.divider()

    st.subheader("Detailed Pipeline Steps")
    # Counts are computed from the current data so the text stays correct if the dataset changes.
    missing_markers = int((raw_df == "?").sum().sum())
    duplicate_count = raw_df.duplicated().sum()
    steps = [
        ("1. Header Standardization", "Replaced dots with underscores (e.g., `education.num` ➔ `education_num`, `marital.status` ➔ `marital_status`)."),
        ("2. Missing Value Imputation", f"Identified {missing_markers:,} `'?'` text markers across workclass, occupation, and native-country and encoded as `'Unknown'`."),
        ("3. Deduplication", f"Detected and dropped {duplicate_count:,} duplicate records, preserving {len(cleaned_df):,} unique observations."),
        ("4. Feature Selection", "Dropped `fnlwgt` (census weighting) and redundant text `education` (retained `education_num`)."),
        ("5. Categorical Aggregation", "Consolidated `native_country` into `'United-States'` and `'Other'` to eliminate low-sample sparse categories."),
        ("6. Variance Stabilization (log1p)", "Applied `np.log1p(x) = ln(1 + x)` on `capital_gain` and `capital_loss` to dampen zero-inflation and severe skewness."),
        ("7. Target Binarization", "Mapped `<=50K` ➔ 0 and `>50K` ➔ 1."),
        ("8. Stratified Partitioning", "Split into 80% train (26,029 samples) and 20% test (6,508 samples) with identical target balance (`stratify=y`)."),
        ("9. Leak-Free Scaling & Encoding", "Standard scaled 5 numeric features and One-Hot encoded 7 categorical features, expanding to 51 model features."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.write(desc)


# ---------------------------------------------------------
# Page 4: Model Comparison
# ---------------------------------------------------------
def render_model_comparison(metrics_df, confusion_matrices, roc_curves):
    st.title("🏆 Supervised Model Benchmarking & Optimization")
    st.write(
        "Explore baseline cross-validation, hyperparameter tuning (GridSearch vs. RandomSearch), "
        "and final generalization performance on the untouched test partition."
    )

    baseline_df = get_baseline_metrics()
    tuning_df = get_tuning_metrics()
    best_params_dict = get_best_hyperparameters()

    comp_tabs = st.tabs([
        "🎯 1. Final Holdout Test Benchmark",
        "📊 2. Baseline 5-Fold Cross-Validation",
        "⚙️ 3. Hyperparameter Optimization Cockpit",
    ])

    # ---------------------------------------------------------
    # TAB 1: FINAL HOLDOUT TEST BENCHMARK
    # ---------------------------------------------------------
    with comp_tabs[0]:
        st.subheader("Final Holdout Evaluation (Untouched Test Set — 6,508 Records)")

        # Spotlight Winner Card
        st.markdown(
            """
            <div class="best-card">
                <h2>🥇 Overall Winning Model: Support Vector Machine (RBF)</h2>
                <p style="font-size: 1.1rem; margin-bottom: 8px;">
                    Selected strictly based on highest <b>5-Fold Cross-Validated F1-Score (0.6749)</b> and <b>CV ROC-AUC (0.8918)</b> during hyperparameter optimization.
                </p>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <span><b>Optimal Search:</b> GridSearchCV</span>
                    <span><b>CV F1-Score:</b> 0.6749</span>
                    <span><b>Holdout Test F1:</b> 0.6533</span>
                    <span><b>Holdout Test Accuracy:</b> 84.80%</span>
                    <span><b>Optimal Hyperparameters:</b> C=10.0, gamma='scale', kernel='rbf'</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "💡 **Model Selection & Trade-Off Analysis**: "
            "In strict machine learning practice, model selection is decided purely by **validation cross-validation performance** "
            "(where SVM achieved **0.6749** F1 vs Decision Tree's **0.6712** F1). "
            "On the holdout test set, Decision Tree generalized slightly higher (**0.6716** F1) while training **500× faster** (0.20s vs 106.97s), "
            "making Decision Tree an attractive production alternative where inference throughput or compute budget is prioritized."
        )

        st.subheader("Official Holdout Test Performance Ranking Table")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Train Accuracy": "{:.4f}",
                    "Test Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "Training Time (s)": "{:.2f}",
                }
            )
            .highlight_max(
                subset=["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                color="#bbf7d0",
            )
            .highlight_min(
                subset=["Training Time (s)"],
                color="#dbeafe",
            ),
            use_container_width=True,
        )

        st.divider()

        # Side-by-Side Holdout Bar Chart
        st.subheader("Holdout Test Metrics Across All 5 Tuned Models")
        metrics_to_plot = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        fig_grouped = px.bar(
            metrics_df,
            x="Model",
            y=metrics_to_plot,
            barmode="group",
            title="Holdout Test Evaluation Metrics (Tuned Pipelines)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_grouped.update_layout(yaxis_range=[0, 1.05], margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_grouped, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Overfitting Diagnostic: Train vs Test Accuracy")
            fig_overfit = px.bar(
                metrics_df,
                x="Model",
                y=["Train Accuracy", "Test Accuracy"],
                barmode="group",
                title="Generalization Gap (Trained vs. Holdout Test Accuracy)",
                color_discrete_map={"Train Accuracy": "#94a3b8", "Test Accuracy": "#2563eb"},
            )
            fig_overfit.update_layout(yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig_overfit, use_container_width=True)
            st.caption("A tight gap indicates healthy generalization; large gaps indicate memorization.")

        with col2:
            st.subheader("Overlaid Holdout Test ROC Curves")
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random Chance (AUC = 0.50)", line=dict(dash="dash", color="grey")
                )
            )
            colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
            for idx, (model_name, rdata) in enumerate(roc_curves.items()):
                fig_roc.add_trace(
                    go.Scatter(
                        x=rdata["fpr"],
                        y=rdata["tpr"],
                        mode="lines",
                        name=f"{model_name} (AUC = {rdata['roc_auc']:.4f})",
                        line=dict(color=colors[idx % len(colors)], width=2),
                    )
                )
            fig_roc.update_layout(
                title="ROC Curves on Test Partition",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.15),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        # Confusion Matrix Inspector
        st.subheader("Interactive Confusion Matrix Inspector")
        col_cm1, col_cm2 = st.columns([1, 2])
        with col_cm1:
            selected_cm_model = st.selectbox(
                "Select Model to Inspect Confusion Matrix:", list(confusion_matrices.keys())
            )
            cm_norm = st.checkbox("Show as Percentage of Actual Class (%)", value=False)

        with col_cm2:
            raw_cm = confusion_matrices[selected_cm_model]
            if cm_norm:
                display_cm = (raw_cm / raw_cm.sum(axis=1, keepdims=True) * 100).round(1)
                text_template = "%{z:.1f}%"
            else:
                display_cm = raw_cm
                text_template = "%{z:,}"

            fig_cm = px.imshow(
                display_cm,
                labels=dict(x="Predicted Income Class", y="Actual Income Class", color="Count"),
                x=["<=50K (0)", ">50K (1)"],
                y=["<=50K (0)", ">50K (1)"],
                title=f"Confusion Matrix: {selected_cm_model}",
                color_continuous_scale="Blues",
            )
            fig_cm.update_traces(text=display_cm, texttemplate=text_template)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()

        # Precision-Recall Curves & Minority Class Detection
        st.subheader("🎯 Precision-Recall (PR) Curves & Minority-Class Detection (>50K Class)")
        st.markdown(
            "Under class imbalance (76% ≤$50K vs 24% >$50K), ROC curves can be overly optimistic because the large volume "
            "of True Negatives keeps the False Positive Rate low. **Precision-Recall curves** focus strictly on the minority positive class (`>50K`), "
            "evaluating the direct trade-off between positive predictive value (Precision) and detection capture (Recall)."
        )

        pr_metrics_df = get_pr_metrics()
        pr_curves_data = get_pr_curves()

        col_pr1, col_pr2 = st.columns([1.3, 1])

        with col_pr1:
            if pr_curves_data is not None:
                fig_pr = go.Figure()
                prev = pr_curves_data.get("prevalence", 0.2409)
                fig_pr.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[prev, prev],
                        mode="lines",
                        name=f"Baseline Prevalence (AP = {prev:.4f})",
                        line=dict(dash="dash", color="grey", width=1.5),
                    )
                )

                pr_color_map = {
                    "Decision Tree": "#10b981",
                    "Support Vector Machine": "#3b82f6",
                    "Logistic Regression": "#f59e0b",
                    "K-Nearest Neighbors": "#8b5cf6",
                    "Naive Bayes": "#ef4444",
                }
                for m_name, m_data in pr_curves_data.get("models", {}).items():
                    fig_pr.add_trace(
                        go.Scatter(
                            x=m_data["recall"],
                            y=m_data["precision"],
                            mode="lines",
                            name=f"{m_name} (AP = {m_data['average_precision']:.4f})",
                            line=dict(color=pr_color_map.get(m_name, "#2563eb"), width=2),
                        )
                    )

                fig_pr.update_layout(
                    title="Holdout Test Precision-Recall Curves (All 5 Tuned Models)",
                    xaxis_title="Recall (True Positive Rate)",
                    yaxis_title="Precision (Positive Predictive Value)",
                    xaxis=dict(range=[0.0, 1.0]),
                    yaxis=dict(range=[0.0, 1.05]),
                    legend=dict(x=0.45, y=0.95),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            elif os.path.exists("results/final_precision_recall_curve.png"):
                st.image("results/final_precision_recall_curve.png", use_container_width=True)

        with col_pr2:
            st.markdown("#### Average Precision (AP) Benchmark")
            if pr_metrics_df is not None:
                st.dataframe(
                    pr_metrics_df.style.format(
                        {
                            "Average Precision": "{:.4f}",
                            "Precision (Threshold=0.5)": "{:.4f}",
                            "Recall (Threshold=0.5)": "{:.4f}",
                        }
                    ).highlight_max(subset=["Average Precision"], color="#bbf7d0"),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                """
                **Key Research Insights:**
                - **Decision Tree (AP: 0.7809)** achieves the highest Average Precision, demonstrating superior capability in ranking true high earners ahead of false positives across all decision thresholds.
                - **Support Vector Machine (AP: 0.7517)** is second, maintaining clean high-precision boundaries at low-to-medium recall.
                - **Naive Bayes (AP: 0.6210)** suffers from massive precision collapse due to conditional independence violations.
                - **Baseline Prevalence Reference (0.2409)**: Any classifier operating along or below this dashed horizontal baseline performs no better than random guessing without discrimination.
                """
            )

    # ---------------------------------------------------------
    # TAB 2: BASELINE 5-FOLD CROSS-VALIDATION
    # ---------------------------------------------------------
    with comp_tabs[1]:
        st.subheader("Baseline 5-Fold Stratified Cross-Validation Benchmark")
        st.write(
            "Before hyperparameter tuning, all 5 models were evaluated inside Scikit-learn Pipelines "
            "using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Preprocessing was fitted strictly "
            "on the 4 training folds with zero data leakage to the validation fold."
        )

        if baseline_df is not None:
            st.dataframe(
                baseline_df.style.format(
                    {
                        "Mean CV Accuracy": "{:.4f}",
                        "Mean CV Precision": "{:.4f}",
                        "Mean CV Recall": "{:.4f}",
                        "Mean CV F1-Score": "{:.4f}",
                        "Std CV F1-Score": "{:.4f}",
                        "Mean CV ROC-AUC": "{:.4f}",
                        "Mean Train Accuracy": "{:.4f}",
                        "CV Execution Time (s)": "{:.2f}",
                    }
                )
                .highlight_max(
                    subset=["Mean CV Accuracy", "Mean CV Precision", "Mean CV Recall", "Mean CV F1-Score", "Mean CV ROC-AUC"],
                    color="#bbf7d0",
                ),
                use_container_width=True,
            )

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                # F1 Score with Std Error Bars
                fig_base_f1 = px.bar(
                    baseline_df.reset_index(),
                    x="Model",
                    y="Mean CV F1-Score",
                    error_y="Std CV F1-Score",
                    title="Baseline 5-Fold CV F1-Score (with ± 1 Std Error Bars)",
                    color="Mean CV F1-Score",
                    color_continuous_scale="Blues",
                )
                fig_base_f1.update_layout(yaxis_range=[0, 0.8])
                st.plotly_chart(fig_base_f1, use_container_width=True)

            with col_b2:
                # Execution Time
                fig_base_time = px.bar(
                    baseline_df.reset_index(),
                    x="Model",
                    y="CV Execution Time (s)",
                    title="Baseline 5-Fold Cross-Validation Execution Time (s)",
                    color="CV Execution Time (s)",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_base_time, use_container_width=True)

            st.divider()

            # Cross-Validation Stability & Dispersion
            st.subheader("📦 Cross-Validation Stability & Dispersion (Fold-Wise F1-Score)")
            st.markdown(
                "Cross-validation stability assesses how consistently each algorithm performs across different stratified folds. "
                "A model with a high mean F1 score but wide dispersion carries higher operational variance when deployed in production."
            )

            cv_folds_df = get_cv_fold_scores()
            col_box1, col_box2 = st.columns([1.3, 1])

            with col_box1:
                if os.path.exists("results/cv_stability_boxplot.png"):
                    st.image("results/cv_stability_boxplot.png", use_container_width=True)
                elif cv_folds_df is not None:
                    fig_box = px.box(
                        cv_folds_df,
                        x="Model",
                        y="F1 Score",
                        color="Model",
                        points="all",
                        title="Fold-Wise F1-Score Dispersion Across 5 Folds",
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            with col_box2:
                st.markdown("#### Fold-Level Scores Table")
                if cv_folds_df is not None:
                    pivot_folds = cv_folds_df.pivot(index="Fold", columns="Model", values="F1 Score")
                    st.dataframe(
                        pivot_folds.style.format("{:.4f}").highlight_max(axis=1, color="#bbf7d0"),
                        use_container_width=True,
                    )

                st.markdown(
                    """
                    **Stability Highlights:**
                    - **Decision Tree** demonstrates the **tightest interquartile range** (Std = ±0.0039, range: 0.6545 – 0.6642), proving remarkable predictability across data partitions.
                    - **Support Vector Machine** achieves the **highest median F1 (0.6655)** and highest peak fold (0.6983), but exhibits wider fold-to-fold dispersion (Std = ±0.0145).
                    - **Naive Bayes** exhibits low variance (Std = ±0.0060) but is consistently trapped at low F1 (0.5113 – 0.5272) due to independence violations.
                    """
                )

    # ---------------------------------------------------------
    # TAB 3: HYPERPARAMETER OPTIMIZATION COCKPIT
    # ---------------------------------------------------------
    with comp_tabs[2]:
        st.subheader("Hyperparameter Optimization: GridSearchCV vs. RandomizedSearchCV")
        st.write(
            "Both search strategies evaluated complete pipelines using 5-fold Stratified CV optimizing F1-Score. "
            "Compare optimization yields, validation metrics, and computational budgets."
        )

        if tuning_df is not None:
            st.dataframe(
                tuning_df.style.format(
                    {
                        "Best CV F1-Score": "{:.4f}",
                        "CV Accuracy": "{:.4f}",
                        "CV Precision": "{:.4f}",
                        "CV Recall": "{:.4f}",
                        "CV ROC-AUC": "{:.4f}",
                        "Tuning Time (s)": "{:.2f}",
                    }
                )
                .highlight_max(subset=["Best CV F1-Score", "CV ROC-AUC"], color="#bbf7d0"),
                use_container_width=True,
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig_tune_comp = px.bar(
                    tuning_df,
                    x="Model",
                    y="Best CV F1-Score",
                    color="Search Method",
                    barmode="group",
                    title="Tuning Yield: GridSearchCV vs. RandomizedSearchCV (CV F1)",
                    color_discrete_map={"GridSearchCV": "#2563eb", "RandomizedSearchCV": "#10b981"},
                )
                fig_tune_comp.update_layout(yaxis_range=[0.45, 0.75])
                st.plotly_chart(fig_tune_comp, use_container_width=True)

            with col_t2:
                fig_tune_time = px.bar(
                    tuning_df,
                    x="Model",
                    y="Tuning Time (s)",
                    color="Search Method",
                    barmode="group",
                    title="Search Efficiency: Wall-Clock Tuning Duration (s)",
                    color_discrete_map={"GridSearchCV": "#2563eb", "RandomizedSearchCV": "#10b981"},
                )
                st.plotly_chart(fig_tune_time, use_container_width=True)

            st.markdown("#### 🔬 Optimal Hyperparameter Configurations Found")
            for m_name, meta in best_params_dict.items():
                with st.expander(f"⚙️ {m_name} (Best via {meta.get('search_method', 'N/A')} — CV F1: {meta.get('best_cv_f1', 0):.4f})"):
                    st.json(meta.get("best_params", {}))

            st.divider()

            # Hyperparameter Heatmaps & Parameter Sensitivity Analysis
            st.subheader("🗺️ Empirical Hyperparameter Heatmaps & Parameter Sensitivity Analysis")
            st.markdown(
                "Visualize the actual cross-validation F1 response surface across candidate hyperparameter combinations from exhaustive `GridSearchCV`. "
                "These empirical heatmaps reveal parameter sensitivities, optimal plateaus, and regularization trade-offs."
            )

            heatmap_models = {
                "Support Vector Machine": {
                    "slug": "svm",
                    "img": "results/svm_hyperparameter_heatmap.png",
                    "x_param": "Regularization Parameter C",
                    "y_param": "Kernel Coefficient gamma",
                    "desc": (
                        "Heatmap of Regularization C vs. RBF kernel coefficient gamma. Demonstrates that higher C (10.0) with scale gamma achieves peak F1 (0.6749) "
                        "by penalizing margin violations on the minority class."
                    ),
                },
                "Decision Tree": {
                    "slug": "decision_tree",
                    "img": "results/decision_tree_hyperparameter_heatmap.png",
                    "x_param": "max_depth",
                    "y_param": "min_samples_leaf",
                    "desc": (
                        "Heatmap of Maximum Tree Depth vs. Minimum Leaf Samples (criterion='gini', min_samples_split=2). Demonstrates that unpruned depth (None) "
                        "combined with min_samples_leaf=30 provides the optimal balance between tree depth and leaf regularization."
                    ),
                },
                "K-Nearest Neighbors": {
                    "slug": "knn",
                    "img": "results/knn_hyperparameter_heatmap.png",
                    "x_param": "n_neighbors",
                    "y_param": "weights",
                    "desc": (
                        "Heatmap of Number of Neighbors (k) vs. Weighting Function (distance metric p=2). Demonstrates that uniform weighting with k=11 achieves "
                        "optimal performance (0.6606), outperforming distance weighting in high-dimensional one-hot space."
                    ),
                },
                "Logistic Regression": {
                    "slug": "logistic_regression",
                    "img": "results/logistic_regression_hyperparameter_plot.png",
                    "x_param": "Inverse Regularization Strength C",
                    "y_param": "Mean CV F1",
                    "desc": (
                        "1D parameter sensitivity curve of C vs. Mean CV F1. Shows that F1 plateaus for C >= 1.0, reaching 0.6462 at C=100.0 "
                        "with minimal sensitivity to further regularization loosening."
                    ),
                },
                "Naive Bayes": {
                    "slug": "gaussian_nb",
                    "img": "results/gaussian_nb_hyperparameter_plot.png",
                    "x_param": "var_smoothing",
                    "y_param": "Mean CV F1",
                    "desc": (
                        "1D curve of Variance Smoothing parameter. Confirms that variance smoothing cannot overcome the severe conditional independence "
                        "violations inherent in correlated census features."
                    ),
                },
            }

            col_hm_sel, col_hm_view = st.columns([1, 2.5])
            with col_hm_sel:
                selected_hm_name = st.selectbox(
                    "Select Model to Inspect Hyperparameter Surface:",
                    list(heatmap_models.keys()),
                )
                hm_info = heatmap_models[selected_hm_name]
                st.info(f"**Research Insight:** {hm_info['desc']}")
                show_raw_candidates = st.checkbox("Show Complete Candidate Results Table", value=False)

            with col_hm_view:
                if os.path.exists(hm_info["img"]):
                    st.image(hm_info["img"], use_container_width=True)
                else:
                    st.warning(f"Heatmap image not found: {hm_info['img']}")

            if show_raw_candidates:
                cand_df = get_grid_cv_candidates(hm_info["slug"])
                if cand_df is not None:
                    st.markdown(f"#### Complete Candidate Evaluations ({selected_hm_name} — {len(cand_df)} candidates)")
                    fmt_dict = {
                        "mean_test_f1": "{:.4f}",
                        "std_test_f1": "{:.4f}",
                        "mean_test_accuracy": "{:.4f}",
                        "mean_test_roc_auc": "{:.4f}",
                        "mean_fit_time": "{:.2f}",
                    }
                    valid_fmt = {k: v for k, v in fmt_dict.items() if k in cand_df.columns}
                    st.dataframe(
                        cand_df.style.format(valid_fmt, na_rep="-").highlight_max(
                            subset=["mean_test_f1"] if "mean_test_f1" in cand_df.columns else None,
                            color="#bbf7d0",
                        ),
                        use_container_width=True,
                    )


# ---------------------------------------------------------
# Page 5: Live Predictor
# ---------------------------------------------------------
def render_live_predictor(models, preprocessor):
    st.title("🔮 Live Income Prediction Tool")
    st.write("Input individual demographic and employment details to forecast whether annual income exceeds $50,000.")

    best_pipe = load_best_pipeline()
    best_params_meta = get_best_hyperparameters()

    st.warning(
        "⚖️ **Historical Educational Disclaimer**: This model is trained on the 1994 United States Census dataset. "
        "Economic conditions, wages, and societal parity benchmarks have evolved substantially since 1994. "
        "This tool is built solely for educational and machine learning demonstration purposes."
    )

    with st.form("prediction_form"):
        st.markdown("### 📋 Individual Profile Inputs")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (years):", min_value=17, max_value=90, value=38)
            workclass = st.selectbox(
                "Workclass:",
                [
                    "Private",
                    "Self-emp-not-inc",
                    "Self-emp-inc",
                    "Federal-gov",
                    "Local-gov",
                    "State-gov",
                    "Without-pay",
                    "Never-worked",
                    "Unknown",
                ],
            )
            education_choice = st.selectbox(
                "Education Level:",
                [
                    "Bachelors (13)",
                    "HS-grad (9)",
                    "11th (7)",
                    "Masters (14)",
                    "9th (5)",
                    "Some-college (10)",
                    "Assoc-acdm (12)",
                    "Assoc-voc (11)",
                    "7th-8th (4)",
                    "Doctorate (16)",
                    "Prof-school (15)",
                    "5th-6th (3)",
                    "10th (6)",
                    "1st-4th (2)",
                    "Preschool (1)",
                    "12th (8)",
                ],
            )
            # Parse integer education.num from selection string
            education_num = int(education_choice.split("(")[1].replace(")", ""))

        with c2:
            marital_status = st.selectbox(
                "Marital Status:",
                [
                    "Married-civ-spouse",
                    "Never-married",
                    "Divorced",
                    "Separated",
                    "Widowed",
                    "Married-spouse-absent",
                    "Married-AF-spouse",
                ],
            )
            occupation = st.selectbox(
                "Occupation Role:",
                [
                    "Exec-managerial",
                    "Prof-specialty",
                    "Craft-repair",
                    "Sales",
                    "Adm-clerical",
                    "Other-service",
                    "Machine-op-inspct",
                    "Transport-moving",
                    "Handlers-cleaners",
                    "Farming-fishing",
                    "Tech-support",
                    "Protective-serv",
                    "Priv-house-serv",
                    "Armed-Forces",
                    "Unknown",
                ],
            )
            relationship = st.selectbox(
                "Household Relationship:",
                ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"],
            )

        with c3:
            sex = st.selectbox("Sex:", ["Male", "Female"])
            race = st.selectbox(
                "Race:",
                ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
            )
            hours_per_week = st.slider("Hours Worked per Week:", min_value=1, max_value=99, value=40)
            native_country = st.selectbox(
                "Native Country:",
                ["United-States", "Other", "Canada", "Germany", "Mexico", "Philippines", "India"],
            )

        c4, c5 = st.columns(2)
        with c4:
            capital_gain = st.number_input(
                "Capital Gain ($):", min_value=0, max_value=99999, value=0, step=100
            )
        with c5:
            capital_loss = st.number_input(
                "Capital Loss ($):", min_value=0, max_value=5000, value=0, step=100
            )

        st.markdown("### 🤖 Model Selection")
        selected_model_name = st.selectbox(
            "Select Inference Model Pipeline:",
            [
                "Support Vector Machine (Top Ranked CV Winner)",
                "Decision Tree (Top Holdout Test Performer)",
                "Logistic Regression",
                "K-Nearest Neighbors",
                "Naive Bayes",
            ],
        )

        submit_btn = st.form_submit_button("🚀 Run Income Prediction", use_container_width=True)

    if submit_btn:
        clean_model_key = selected_model_name.split(" (")[0]
        clf_pipeline = models.get(clean_model_key, best_pipe)

        # Raw user input dictionary
        raw_input = {
            "age": age,
            "education_num": education_num,
            "hours_per_week": hours_per_week,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "workclass": workclass,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "native_country": native_country,
        }

        # Pass raw DataFrame directly into Scikit-learn Pipeline (zero manual outside encoding!)
        input_df = pd.DataFrame([raw_input])

        # Execute end-to-end Pipeline prediction
        pred_class = clf_pipeline.predict(input_df)[0]
        if hasattr(clf_pipeline, "predict_proba"):
            proba = clf_pipeline.predict_proba(input_df)[0]
            prob_high = proba[1]
            prob_low = proba[0]
        else:
            prob_high = 0.5
            prob_low = 0.5

        st.markdown("---")
        st.subheader("🎯 Prediction Output")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred_class == 1:
                st.success("### 🎉 Predicted Class: **> $50,000 / year**")
                st.write("This applicant profile has high statistical alignment with upper-income bracket earners.")
            else:
                st.info("### 📘 Predicted Class: **≤ $50,000 / year**")
                st.write("This applicant profile aligns with baseline income bracket earners.")

        with res_col2:
            st.markdown(f"#### Model Probability Breakdown ({clean_model_key})")
            st.write(f"- Probability of **> $50K**: **{prob_high*100:.1f}%**")
            st.write(f"- Probability of **≤ $50K**: **{prob_low*100:.1f}%**")
            st.progress(float(np.clip(prob_high, 0.0, 1.0)))

        # Display optimal hyperparameters used for this prediction
        if clean_model_key in best_params_meta:
            m_info = best_params_meta[clean_model_key]
            st.caption(f"Pipeline: {clean_model_key} | Optimal Search: {m_info.get('search_method')} | Optimal Params: {m_info.get('best_params')}")


# ---------------------------------------------------------
# Page 6: Written Analysis
# ---------------------------------------------------------
def render_analysis(metrics_df):
    st.title("📝 Comprehensive Machine Learning Analysis")
    st.write("Rigorous, empirical analysis based on real cross-validation benchmarks and hyperparameter optimization.")

    st.markdown(
        """
        ### 1. Cross-Validation Model Selection & The Real-World Trade-Off
        - **Why Support Vector Machine (RBF) Won the CV Ranking (F1: 0.6749)**:
          - By optimizing `C=10.0` with RBF kernel scaling, SVM projected the 51-dimensional one-hot encoded demographic features into Hilbert space, successfully establishing non-linear margin boundaries with low variance across all 5 folds.
          - During cross-validation on the 26,029 training records, SVM achieved the highest mean F1-Score (**0.6749**) and balanced Precision (**73.88%**) with Recall (**62.14%**).
        - **Why Decision Tree was an Exceptional Production Contender**:
          - Decision Tree tuned via GridSearchCV (`min_samples_leaf=30`, `min_samples_split=2`, `criterion='gini'`) achieved **0.6712** CV F1-score (trailing by only 0.0037).
          - On the completely untouched holdout test partition (6,508 samples), Decision Tree generalized slightly better (**Test F1: 0.6716** vs. **0.6533** for SVM), while taking **0.20s** to train versus **106.97s** for SVM (over **500× faster**).
          - In business production environments where continuous retraining and microsecond inference latencies are required, the Decision Tree pipeline offers virtually identical accuracy at a fraction of the computational expense.

        ---

        ### 2. GridSearchCV versus RandomizedSearchCV Efficiency Analysis
        - **Computational Efficiency**:
          - For **Logistic Regression**, RandomizedSearch (`n_iter=5`) discovered the exact same optimal hyperparameter (`C=100.0`, CV F1: 0.6462) as exhaustive GridSearch in **2.46s vs. 7.37s** (3× speedup).
          - For **Decision Tree**, RandomizedSearch (`n_iter=25`) achieved **0.6669** F1 in **7.61s** compared to **46.77s** for exhaustive 160-candidate GridSearch (0.6712 F1) — capturing 99.4% of the performance in 16% of the wall-clock time.
          - For **KNN**, both search methods converged on `n_neighbors=11`, `weights='uniform'`, `p=2` (CV F1: 0.6606), with RandomizedSearch reducing tuning time from **197.13s to 122.29s**.

        ---

        ### 3. Why Gaussian Naive Bayes Struggled (Rank 5)
        - **Severe Violation of Feature Independence**: Naive Bayes assumes all features are conditionally independent given the class label. In reality, the adult dataset exhibits extreme redundancy (e.g., `marital_status_Married-civ-spouse` and `relationship_Husband`, or education and occupation).
        - **Probability Inflation & Poor Precision**: Tuning `var_smoothing=1e-08` marginally improved F1 from baseline 0.5227 to 0.5326, but the model remained hindered by high false-positive rates (**Precision of 37.01%** on test data), proving that mathematical assumption violations cannot be resolved through hyperparameter tuning alone.

        ---

        ### 4. Leakage-Free Preprocessing in Scikit-learn Pipelines
        - Moving all preprocessing into `ColumnTransformer` embedded inside `Pipeline` guarantees that `StandardScaler`, `SimpleImputer`, and `OneHotEncoder` are fit **strictly on training folds during cross-validation**.
        - Validation folds and test data remain completely unseen until inference.
        - Saved pipeline artifacts now encapsulate all preprocessing logic, enabling raw dataframes to be passed directly to `pipeline.predict()` with zero manual preprocessing steps and seamless handling of unseen categories.
        """
    )


# ---------------------------------------------------------
# Main Controller & Navigation Routing
# ---------------------------------------------------------
def main():
    raw_df, cleaned_df = get_raw_and_cleaned_data()
    metrics_df = get_metrics()
    confusion_matrices = get_confusion_matrices()
    roc_curves = get_roc_curves()
    models = load_all_models()
    preprocessor = load_fitted_preprocessor()

    # Sidebar Navigation with Query Param Support
    page_options = [
        "🏠 Home",
        "📊 EDA Explorer",
        "⚙️ Preprocessing",
        "🏆 Model Comparison",
        "🔮 Live Predictor",
        "📝 Analysis",
    ]
    query_param_map = {
        "home": "🏠 Home",
        "eda": "📊 EDA Explorer",
        "preprocessing": "⚙️ Preprocessing",
        "comparison": "🏆 Model Comparison",
        "predictor": "🔮 Live Predictor",
        "analysis": "📝 Analysis",
    }

    # Check if page was specified in query parameters
    default_index = 0
    requested_page = st.query_params.get("page", "").lower()
    if requested_page in query_param_map:
        target_name = query_param_map[requested_page]
        if target_name in page_options:
            default_index = page_options.index(target_name)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        page_options,
        index=default_index,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Adult Income Classification ML Project")
    st.sidebar.caption("Dataset: 1994 US Census (32,561 rows)")

    if page == "🏠 Home":
        render_home(raw_df, cleaned_df, metrics_df)
    elif page == "📊 EDA Explorer":
        render_eda(raw_df, cleaned_df)
    elif page == "⚙️ Preprocessing":
        render_preprocessing(raw_df, cleaned_df)
    elif page == "🏆 Model Comparison":
        render_model_comparison(metrics_df, confusion_matrices, roc_curves)
    elif page == "🔮 Live Predictor":
        render_live_predictor(models, preprocessor)
    elif page == "📝 Analysis":
        render_analysis(metrics_df)


if __name__ == "__main__":
    main()
