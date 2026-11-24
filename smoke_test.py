"""Smoke test: build pipelines for the 5 models, fit, predict; run baseline CV for fast models."""
import sys, os, time
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir
D = project_dir()
os.chdir(D)
sys.path.insert(0, os.path.join(D, "src"))

from preprocess import load_raw_data, clean_data, split_and_preprocess, build_model_pipeline, transform_single_input
from train_models import get_tunable_models, get_shared_cv, SCALE_NUMERIC_MAP

raw = load_raw_data("data/adult.csv")
df = clean_data(raw)
X_train, X_test, y_train, y_test, _, X_train_raw, X_test_raw = split_and_preprocess(df)
print("X_train_raw:", X_train_raw.shape, "y_train:", y_train.shape)

calib_cv = get_shared_cv()
models = get_tunable_models(calib_cv)
for name, m in models.items():
    t0 = time.time()
    pipe = build_model_pipeline(m, scale_numeric=SCALE_NUMERIC_MAP[name])
    pipe.fit(X_train_raw, y_train)
    acc = pipe.score(X_test_raw, y_test)
    df_single = transform_single_input({
        "age": 38, "workclass": "Private", "education_num": 13,
        "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
        "native_country": "United-States",
    })
    pred = pipe.predict(df_single)[0]
    print(f"{name}: fit {time.time()-t0:.1f}s | test acc {acc:.4f} | single pred {pred}")
print("SMOKE TEST OK")
