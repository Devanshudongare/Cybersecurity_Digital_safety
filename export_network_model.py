import json
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/network_detector.joblib")
OUTPUT_PATH = Path("android_network_export/network_model.json")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["model"]
imputer = pipeline.named_steps["imputer"]
forest = pipeline.named_steps["classifier"]

def export_tree(estimator):
    t = estimator.tree_
    values = t.value[:, 0, :]

    # Normalize leaf values into class probabilities
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        values, totals,
        out=np.zeros_like(values),
        where=totals != 0
    )

    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "probabilities": probabilities.tolist()
    }

output = {
    "features": bundle["feature_columns"],
    "medians": imputer.statistics_.tolist(),
    "classes": forest.classes_.tolist(),
    "trees": [export_tree(tree) for tree in forest.estimators_]
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, separators=(",", ":"))

print("Export completed:", OUTPUT_PATH)
print("Features:", len(output["features"]))
print("Trees:", len(output["trees"]))
print("JSON size:", round(OUTPUT_PATH.stat().st_size / 1024 / 1024, 2), "MB")
