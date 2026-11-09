import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "sms" / "SMSSpamCollection"
MODEL_PATH = ROOT / "models" / "sms_detector_v6.joblib"
REPORT_PATH = ROOT / "reports" / "sms_v6_report.json"

print("=" * 60)
print("CYBERGUARD AI - SMS V6 TRAINING")
print("=" * 60)

# 1. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SMS dataset not found: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]

X = df["message"].astype(str)
y = df["label"]

print(f"\nDataset samples: {len(df)}")
print(f"Ham: {(y == 'ham').sum()}")
print(f"Spam: {(y == 'spam').sum()}")

# 2. Same stratified split used by our model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 3. Word + Character TF-IDF
features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=100000,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=100000,
        ),
    ),
])

# 4. Linear SVM classifier
model = Pipeline([
    ("features", features),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
    ),
])

# 5. Train
print("\nTraining SMS V6...")
model.fit(X_train, y_train)
print("Training completed.")

# 6. Evaluate
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
matrix = confusion_matrix(
    y_test, predictions, labels=["ham", "spam"]
)

print("\n" + "=" * 60)
print("SMS V6 RESULTS")
print("=" * 60)
print(f"Accuracy:       {accuracy:.6f} ({accuracy * 100:.2f}%)")
print(f"Spam Precision:{precision:.6f}")
print(f"Spam Recall:   {recall:.6f}")
print(f"Spam F1:       {f1:.6f}")
print("\nConfusion Matrix [ham, spam]:")
print(matrix)
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 7. Save model and report
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

report = {
    "model": "SMS V6 Word + Character TF-IDF Linear SVM",
    "dataset_samples": int(len(df)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "random_state": 42,
    "test_size": 0.20,
    "accuracy": float(accuracy),
    "spam_precision": float(precision),
    "spam_recall": float(recall),
    "spam_f1": float(f1),
    "false_positives": int(matrix[0][1]),
    "false_negatives": int(matrix[1][0]),
    "confusion_matrix": matrix.tolist(),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print(f"\nModel saved: {MODEL_PATH}")
print(f"Report saved: {REPORT_PATH}")
print("\nSMS V6 TRAINING FINISHED.")
