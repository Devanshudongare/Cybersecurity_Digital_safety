from pathlib import Path
import pandas as pd

root = Path("data/raw/sms/kaggle")

for path in root.rglob("*.csv"):
    print("\n" + "=" * 70)
    print("FILE:", path.name)
    print("=" * 70)

    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                on_bad_lines="skip",
                low_memory=False,
            )

            print("Encoding:", encoding)
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print("\nSample:")
            print(df.head(3).to_string(index=False))
            break

        except UnicodeDecodeError:
            continue

    else:
        print("Unable to decode this CSV.")
