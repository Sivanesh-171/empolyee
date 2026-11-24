"""Patch src/train_models.py: add per-model tuned pipeline dumps; update requirements.txt."""
import sys
sys.path.insert(0, "C:/Users/Sivan/OneDrive/college/Intern")
from helpers import project_dir

D = project_dir()
P = lambda rel: f"{D}/{rel}"

src = open(P("src/train_models.py"), encoding="utf-8").read()

old = '''            joblib.dump(
                searches[(name, method)].best_estimator_,
                f"models/best_pipeline_{method}_{slug}.joblib",
            )

    best_row, best_method = select_best_model(tuned_df)
'''
new = '''            joblib.dump(
                searches[(name, method)].best_estimator_,
                f"models/best_pipeline_{method}_{slug}.joblib",
            )

    # Per-model tuned pipelines: winner between grid and random by Best CV F1.
    for name in models:
        slug = MODEL_SLUGS[name]
        mrows = tuned_df[tuned_df["Model"] == name]
        best_m = mrows.sort_values("Best CV F1", ascending=False).iloc[0]
        mmethod = "grid" if best_m["Search Method"] == "GridSearchCV" else "random"
        joblib.dump(
            searches[(name, mmethod)].best_estimator_,
            f"models/best_pipeline_{slug}.joblib",
        )

    best_row, best_method = select_best_model(tuned_df)
'''
assert old in src, "pattern not found in train_models.py"
src = src.replace(old, new)
open(P("src/train_models.py"), "w", encoding="utf-8").write(src)
print("patched per-model tuned dumps")

req = open(P("requirements.txt"), encoding="utf-8").read()
if "matplotlib" not in req:
    req = req.rstrip() + "\nmatplotlib>=3.7.0\n"
    open(P("requirements.txt"), "w", encoding="utf-8").write(req)
    print("requirements.txt updated")
else:
    print("matplotlib already in requirements")
