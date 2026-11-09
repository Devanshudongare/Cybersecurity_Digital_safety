from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

data_path = ROOT / "data/raw/sms/SMSSpamCollection"
model_path = ROOT / "models/sms_detector_v3_experiment.joblib"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

df["label"] = df["label"].str.strip().str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].astype(str),
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

model = joblib.load(model_path)
predictions = model.predict(X_test)

results = pd.DataFrame({
    "message": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": predictions
})

false_negatives = results[
    (results["actual"] == "spam") &
    (results["predicted"] == "ham")
]

false_positives = results[
    (results["actual"] == "ham") &
    (results["predicted"] == "spam")
]

print("\n===== FALSE NEGATIVES =====")
print("Spam incorrectly classified as legitimate:", len(false_negatives))
print(false_negatives.to_string(index=False))

print("\n===== FALSE POSITIVES =====")
print("Legitimate messages incorrectly classified as spam:", len(false_positives))
print(false_positives.to_string(index=False))

output_path = ROOT / "reports/sms_v3_error_analysis.csv"
results[results["actual"] != results["predicted"]].to_csv(
    output_path, index=False
)

print("\nError report saved:", output_path)
