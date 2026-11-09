import zipfile
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "sms" / "kaggle"

csv_path = DATA_DIR / "spam.csv"
zip_path = DATA_DIR / "archive.zip"

with zipfile.ZipFile(zip_path) as z:
    zip_bytes = z.read("spam.csv")

csv_bytes = csv_path.read_bytes()

print("Existing CSV:", csv_path)
print("ZIP CSV:", zip_path)
print("Existing CSV bytes:", len(csv_bytes))
print("ZIP CSV bytes:", len(zip_bytes))

hash_csv = hashlib.sha256(csv_bytes).hexdigest()
hash_zip = hashlib.sha256(zip_bytes).hexdigest()

print("\nExisting CSV SHA256:", hash_csv)
print("ZIP CSV SHA256:", hash_zip)
print("\nIDENTICAL:", csv_bytes == zip_bytes)
