
import json
import joblib
from pathlib import Path

model = joblib.load("models/sms_detector.joblib")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()

export_data = {
    "vocabulary": feature_names.tolist(),
    "idf": vectorizer.idf_.tolist(),
    "weights": classifier.coef_[0].tolist(),
    "intercept": float(classifier.intercept_[0]),
    "classes": classifier.classes_.tolist(),
    "ngram_range": list(vectorizer.ngram_range),
    "sublinear_tf": vectorizer.sublinear_tf,
    "norm": vectorizer.norm,
    "lowercase": vectorizer.lowercase,
    "token_pattern": vectorizer.token_pattern,
}

output_dir = Path("android_sms_export")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "sms_model.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(export_data, f, ensure_ascii=False)

print("SMS model exported successfully!")
print("Features:", len(feature_names))
print("Classes:", export_data["classes"])
print("Output:", output_file.resolve())
print("File size:", round(output_file.stat().st_size / 1024, 2), "KB")
