from pathlib import Path
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "data" / "raw" / "sms" / "kaggle"
OUTPUT = ROOT / "data" / "processed" / "sms_kaggle_merged.csv"

def normalize_label(value):
    value = str(value).strip().lower()

    if value in {"ham", "legitimate", "normal", "not spam", "0"}:
        return "ham"

    if value in {"spam", "phishing", "fraud", "malicious", "1"}:
        return "spam"

    return None

records = []

for path in KAGGLE.rglob("*.csv"):
    print(f"\nReading: {path.relative_to(ROOT)}")

    try:
        df = pd.read_csv(path, encoding="cp1252", on_bad_lines="skip")
    except Exception as exc:
        print("Skipped:", exc)
        continue

    columns = {str(c).strip().lower(): c for c in df.columns}

    label_col = next(
        (columns[c] for c in ["v1", "label", "labels", "category", "class", "type"]
         if c in columns),
        None,
    )

    text_col = next(
        (columns[c] for c in ["v2", "text", "message", "sms", "content"]
         if c in columns),
        None,
    )

    if label_col is None or text_col is None:
        print("Skipped: Could not identify label/text columns")
        print("Columns:", list(df.columns))
        continue

    added = 0

    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label = normalize_label(row[label_col])

        if not text or text.lower() == "nan" or label is None:
            continue

        records.append({
            "text": text,
            "label": label,
            "source": path.name,
        })
        added += 1

    print("Accepted rows:", added)

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError("No usable SMS records found.")

result["normalized_text"] = (
    result["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

before = len(result)
result = result.drop_duplicates(subset=["normalized_text", "label"])
after = len(result)

result = result.drop(columns=["normalized_text"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print("KAGGLE SMS DATASET REPORT")
print("=" * 60)
print("Total accepted records:", before)
print("Unique records:", after)
print("Duplicates removed:", before - after)
print("\nLabel distribution:")
print(result["label"].value_counts())
print("\nSaved:", OUTPUT)
