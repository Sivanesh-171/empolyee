"""Modify app.py in the project dir: load tuned pipelines and show new results."""
import sys
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir

D = project_dir()
P = lambda rel: f"{D}/{rel}"

src = open(P("app.py"), encoding="utf-8").read()

# ---------------------------------------------------------------------------
# 1. Add new loaders after load_fitted_preprocessor
# ---------------------------------------------------------------------------
old_loader = '''@st.cache_resource
def load_fitted_preprocessor():
    """Load fitted preprocessor bundle for live inference."""
    return joblib.load("models/preprocessor.joblib")
'''
new_loader = '''@st.cache_resource
def load_fitted_preprocessor():
    """Load fitted preprocessor bundle for live inference."""
    return joblib.load("models/preprocessor.joblib")


@st.cache_data
def get_baseline_cv():
    """Load the baseline 5-fold cross-validation comparison table."""
    return pd.read_csv("results/baseline_cv.csv", index_col="Rank")


@st.cache_data
def get_tuned_comparison():
    """Load the tuned grid/randomized comparison table."""
    return pd.read_csv("results/tuned_models_comparison.csv")


@st.cache_data
def get_final_metrics():
    """Load final holdout metrics of the selected best pipeline."""
    return pd.read_csv("results/final_test_metrics.csv")


@st.cache_resource
def load_tuned_models():
    """
    Load the tuned pipeline for each model (models/best_pipeline_<slug>.joblib).

    Each pipeline contains the full preprocessing (imputation, optional scaling,
    one-hot encoding) plus the tuned classifier, so raw inputs can be passed
    straight to ``predict`` without any manual transformation.
    """
    slugs = {
        "Decision Tree": "decision_tree",
        "Support Vector Machine": "svm",
        "Logistic Regression": "logistic_regression",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
    }
    tuned = {}
    for name, slug in slugs.items():
        tuned[name] = joblib.load(f"models/best_pipeline_{slug}.joblib")
    return tuned
'''
assert old_loader in src
src = src.replace(old_loader, new_loader)

# ---------------------------------------------------------------------------
# 2. Extend render_model_comparison with new workflow results
# ---------------------------------------------------------------------------
old_cm_end = '''        fig_cm.update_traces(text=display_cm, texttemplate=text_template)
        st.plotly_chart(fig_cm, use_container_width=True)


# ---------------------------------------------------------
# Page 5: Live Predictor
# ---------------------------------------------------------
def render_live_predictor(models, preprocessor):
'''
new_cm_end = '''        fig_cm.update_traces(text=display_cm, texttemplate=text_template)
        st.plotly_chart(fig_cm, use_container_width=True)

    # -----------------------------------------------------------------
    # New leakage-safe ML workflow results (baseline CV / tuning / final)
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Cross-Validation, Tuning & Final Evaluation Results")

    tuned_df = get_tuned_comparison()
    st.subheader("Tuned Models Comparison (Best CV F1 per Model/Method)")
    st.dataframe(
        tuned_df.style.format(
            {
                "Best CV F1": "{:.4f}",
                "CV Accuracy": "{:.4f}",
                "CV Precision": "{:.4f}",
                "CV Recall": "{:.4f}",
                "CV ROC-AUC": "{:.4f}",
                "Training Time (s)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    fig_tuned = px.bar(
        tuned_df,
        x="Model",
        y="Best CV F1",
        color="Search Method",
        barmode="group",
        title="GridSearchCV vs RandomizedSearchCV - Best CV F1 per Model",
        color_discrete_map={"GridSearchCV": "#2563eb", "RandomizedSearchCV": "#f97316"},
    )
    fig_tuned.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_tuned, use_container_width=True)

    baseline_df = get_baseline_cv()
    st.subheader("Baseline 5-Fold Cross-Validation Results (Mean F1 Sorted)")
    st.dataframe(
        baseline_df.style.format(
            {
                "Mean Accuracy": "{:.4f}",
                "Mean Precision": "{:.4f}",
                "Mean Recall": "{:.4f}",
                "Mean F1-Score": "{:.4f}",
                "Std F1-Score": "{:.4f}",
                "Mean ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    final_df = get_final_metrics()
    st.subheader("Final Holdout Test Metrics (Untouched Test Set)")
    st.dataframe(
        final_df.style.format(
            {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}"}
        ),
        use_container_width=True,
    )

    best = tuned_df.sort_values("Best CV F1", ascending=False).iloc[0]
    st.success(
        f"🏆 **Overall Best (by cross-validated F1): {best['Model']}** — "
        f"{best['Search Method']}, Best CV F1 = {best['Best CV F1']:.4f}, "
        f"CV ROC-AUC = {best['CV ROC-AUC']:.4f}"
    )


# ---------------------------------------------------------
# Page 5: Live Predictor
# ---------------------------------------------------------
def render_live_predictor(tuned_models):
'''
assert old_cm_end in src
src = src.replace(old_cm_end, new_cm_end)

# ---------------------------------------------------------------------------
# 3. Live Predictor: select tuned pipelines, pass raw df to the pipeline
# ---------------------------------------------------------------------------
old_select = '''        selected_model_name = st.selectbox(
            "Select Inference Model:",
            [
                "Decision Tree (Top Ranked)",
                "Support Vector Machine",
                "Logistic Regression",
                "K-Nearest Neighbors",
                "Naive Bayes",
            ],
        )

        submit_btn = st.form_submit_button("🚀 Run Income Prediction", use_container_width=True)

    if submit_btn:
        clean_model_key = selected_model_name.replace(" (Top Ranked)", "")
        clf = models[clean_model_key]
'''
new_select = '''        tuned_df = get_tuned_comparison()
        best_overall = tuned_df.sort_values("Best CV F1", ascending=False).iloc[0]
        st.markdown(
            f"**Selected model (overall best by CV F1): {best_overall['Model']} — "
            f"Best CV F1 = {best_overall['Best CV F1']}**"
        )

        selected_model_name = st.selectbox(
            "Select Inference Model:",
            list(tuned_models.keys()),
        )

        submit_btn = st.form_submit_button("🚀 Run Income Prediction", use_container_width=True)

    if submit_btn:
        clf = tuned_models[selected_model_name]
'''
assert old_select in src
src = src.replace(old_select, new_select)

old_xform = '''        # Transform using saved leak-free preprocessor
        X_input = transform_single_input(raw_input, preprocessor)

        # Make prediction
        pred_class = clf.predict(X_input)[0]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_input)[0]
'''
new_xform = '''        # The tuned pipeline contains preprocessing + model: raw inputs go straight in
        df_single = transform_single_input(raw_input)
        X_input = df_single

        # Make prediction
        pred_class = clf.predict(X_input)[0]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_input)[0]
'''
assert old_xform in src
src = src.replace(old_xform, new_xform)

old_proba_fallback = '''        else:
            # Fallback for models without direct probability
            prob_high = 0.5
            prob_low = 0.5
'''
new_proba_fallback = '''        else:
            # Fallback for models without direct probability
            prob_high = 0.5
            prob_low = 0.5

        tuned_map = tuned_df.set_index("Model")
        sel_f1 = float(tuned_map.loc[selected_model_name, "Best CV F1"])
        sel_method = tuned_map.loc[selected_model_name, "Search Method"]
'''
assert old_proba_fallback in src
src = src.replace(old_proba_fallback, new_proba_fallback)

old_res_col2 = '''        with res_col2:
            st.markdown(f"#### Model Probability Breakdown ({clean_model_key})")
            st.write(f"- Probability of **> $50K**: **{prob_high*100:.1f}%**")
            st.write(f"- Probability of **≤ $50K**: **{prob_low*100:.1f}%**")
            st.progress(float(np.clip(prob_high, 0.0, 1.0)))
'''
new_res_col2 = '''        with res_col2:
            st.markdown(f"#### 🤖 Model: {selected_model_name} ({sel_method}, Best CV F1 = {sel_f1:.4f})")
            st.markdown(f"#### Model Probability Breakdown ({selected_model_name})")
            st.write(f"- Probability of **> $50K**: **{prob_high*100:.1f}%**")
            st.write(f"- Probability of **≤ $50K**: **{prob_low*100:.1f}%**")
            st.progress(float(np.clip(prob_high, 0.0, 1.0)))
'''
assert old_res_col2 in src
src = src.replace(old_res_col2, new_res_col2)

# ---------------------------------------------------------------------------
# 4. main(): use tuned pipelines for the Live Predictor page
# ---------------------------------------------------------------------------
old_main = '''    models = load_all_models()
    preprocessor = load_fitted_preprocessor()
'''
new_main = '''    tuned_models = load_tuned_models()
'''
assert old_main in src
src = src.replace(old_main, new_main)

old_route = '''    elif page == "🔮 Live Predictor":
        render_live_predictor(models, preprocessor)
'''
new_route = '''    elif page == "🔮 Live Predictor":
        render_live_predictor(tuned_models)
'''
assert old_route in src
src = src.replace(old_route, new_route)

open(P("app.py"), "w", encoding="utf-8").write(src)
print("app.py updated OK, new length:", len(src))
