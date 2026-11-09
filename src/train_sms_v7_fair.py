from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]

original_path = ROOT / "data/raw/sms/SMSSpamCollection"
kaggle_path = ROOT / "data/raw/sms/kaggle/archive/spam.csv"
model_path = ROOT / "models/sms_detector_v7_fair.joblib"

# Original dataset
original = pd.read_csv(
    original_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

original["label"] = original["label"].str.strip().str.lower()
original = original[original["label"].isin(["ham", "spam"])]

# Original common train/test split
X_train, X_test, y_train, y_test = train_test_split(
    original["text"].astype(str),
    original["label"],
    test_size=0.20,
    random_state=42,
    stratify=original["label"],
)

# Kaggle dataset
kaggle = pd.read_csv(kaggle_path, encoding="cp1252")
kaggle = kaggle[["v1", "v2"]].rename(
    columns={"v1": "label", "v2": "text"}
)

kaggle["label"] = kaggle["label"].str.strip().str.lower()
kaggle["text"] = kaggle["text"].astype(str).str.strip()

# Keep only Kaggle messages absent from original data
original_keys = set(original["text"].str.strip().str.lower())
kaggle["key"] = kaggle["text"].str.strip().str.lower()

extra = kaggle[~kaggle["key"].isin(original_keys)]
extra = extra.drop_duplicates("key")
extra = extra[extra["label"].isin(["ham", "spam"])]

# Add new messages ONLY to training data
X_train_final = pd.concat(
    [X_train.reset_index(drop=True), extra["text"].reset_index(drop=True)],
    ignore_index=True
)

y_train_final = pd.concat(
    [y_train.reset_index(drop=True), extra["label"].reset_index(drop=True)],
    ignore_index=True
)

# Model architecture
features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100000,
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100000,
    )),
])

model = Pipeline([
    ("features", features),
    ("classifier", LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42,
    )),
])

print("Original training samples:", len(X_train))
print("Additional Kaggle messages:", len(extra))
print("Final training samples:", len(X_train_final))
print("Original held-out test samples:", len(X_test))

print("\nTraining fair V7...")
model.fit(X_train_final, y_train_final)

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

print("\nFair V7 model saved:", model_path)
print("Original test data was excluded from training.")
