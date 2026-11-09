
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/raw/sms/SMSSpamCollection"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

df = pd.read_csv(
    DATA, sep="\t", header=None,
    names=["label", "message"]
).dropna()

df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates(subset=["label", "message"])

X = df["message"].astype(str)
y = df["label"].str.strip().str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20,
    random_state=42, stratify=y
)

# V2: Word + character TF-IDF with Logistic Regression
features = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    ))
])

v2 = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ))
])

# V3: Word + character TF-IDF with Linear SVM
v3 = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        ))
    ])),
    ("classifier", LinearSVC(class_weight="balanced"))
])

# V1: Existing model
v1 = joblib.load(MODEL_DIR / "sms_detector.joblib")

models = {
    "V1 Existing Logistic Regression": v1,
    "V2 Word + Character Logistic Regression": v2,
    "V3 Word + Character Linear SVM": v3
}

results = []

for name, model in models.items():
    print("\n" + "=" * 65)
    print(name)

    if name != "V1 Existing Logistic Regression":
        print("Training...")
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, zero_division=0
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    )

    print("\nAccuracy:", round(accuracy, 4))
    print(report)
    print("Confusion matrix [ham, spam]:")
    print(matrix)

    spam_report = classification_report(
        y_test, predictions,
        output_dict=True, zero_division=0
    )["spam"]

    results.append({
        "model": name,
        "accuracy": accuracy,
        "spam_precision": spam_report["precision"],
        "spam_recall": spam_report["recall"],
        "spam_f1": spam_report["f1-score"]
    })

    if name.startswith("V2"):
        joblib.dump(model, MODEL_DIR / "sms_detector_v2_experiment.joblib")

    if name.startswith("V3"):
        joblib.dump(model, MODEL_DIR / "sms_detector_v3_experiment.joblib")

summary = pd.DataFrame(results)
print("\n\nFINAL COMPARISON")
print(summary.to_string(index=False))

summary.to_csv(REPORT_DIR / "sms_experiment_comparison.csv", index=False)
print("\nSaved comparison to reports/sms_experiment_comparison.csv")
print("Experimental models saved separately. V1 remains unchanged.")
