from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/raw/sms/SMSSpamCollection"
REPORT = ROOT / "reports/sms_fair_model_comparison.csv"

MODEL_PATHS = {
    "V3_LinearSVM": ROOT / "models/sms_detector_v3_experiment.joblib",
    "V6_LinearSVC": ROOT / "models/sms_detector_v6.joblib",
    "V7_Fair_Kaggle": ROOT / "models/sms_detector_v7_fair.joblib",
}

# Load original SMS dataset
df = pd.read_csv(
    DATA,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8",
)

df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]

# Identical original test split
_, X_test, _, y_test = train_test_split(
    df["text"].astype(str),
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

results = []

print("TEST SAMPLES:", len(X_test))
print("TEST HAM:", sum(y_test == "ham"))
print("TEST SPAM:", sum(y_test == "spam"))

for name, path in MODEL_PATHS.items():

    print("\n" + "=" * 65)
    print("MODEL:", name)
    print("PATH:", path)

    if not path.exists():
        print("MODEL NOT FOUND - SKIPPING")
        continue

    model = joblib.load(path)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ).ravel()

    print(f"Accuracy:          {accuracy:.4%}")
    print(f"Spam Precision:   {precision:.4%}")
    print(f"Spam Recall:      {recall:.4%}")
    print(f"Spam F1 Score:    {f1:.4%}")
    print(f"False Positives:   {fp}")
    print(f"False Negatives:   {fn}")
    print(f"True Negatives:    {tn}")
    print(f"True Positives:    {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Spam_Precision": precision,
        "Spam_Recall": recall,
        "Spam_F1": f1,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "True_Positives": tp,
    })

comparison = pd.DataFrame(results)

if not comparison.empty:
    comparison = comparison.sort_values(
        "Spam_F1", ascending=False
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT, index=False)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False))

    print("\nComparison report saved:")
    print(REPORT)
else:
    print("No models were evaluated.")
