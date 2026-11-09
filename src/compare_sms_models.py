
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


SEED = 42
DATA_PATH = Path("data/raw/sms/SMSSpamCollection")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# Load exactly the same SMS dataset and test split
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

X = df["message"].astype(str).tolist()
y = df["label"].astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

print("Common test samples:", len(X_test))


# LSTM architecture used in V5
class SMSLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)


# Load V5 model and vocabulary
with open(MODEL_DIR / "sms_lstm_v5_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

checkpoint = torch.load(
    MODEL_DIR / "sms_lstm_v5.pt",
    map_location="cpu",
    weights_only=True,
)

model = SMSLSTM(
    len(vocab),
    checkpoint["embed_dim"],
    checkpoint["hidden_dim"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# V5 tokenizer must match training
import re

def tokenize(text):
    return re.findall(
        r"[a-z]+|[0-9]+|[^\w\s]",
        str(text).lower(),
    )

def encode_text(text):
    ids = [vocab.get(token, 1) for token in tokenize(text)]
    ids = ids[:checkpoint["max_length"]]
    return ids if ids else [1]


def predict_lstm(messages):
    predictions = []

    with torch.no_grad():
        for message in messages:
            sequence = encode_text(message)
            tensor = torch.tensor(
                sequence, dtype=torch.long
            ).unsqueeze(0)

            probability = torch.sigmoid(model(tensor)).item()
            predictions.append(int(probability >= 0.5))

    return np.array(predictions)


# Load classical ML models
model_files = {
    "V1 Logistic Regression": "sms_detector.joblib",
    "V2 Logistic Regression": "sms_detector_v2_experiment.joblib",
    "V3 Linear SVM": "sms_detector_v3_experiment.joblib",
}

results = []
predictions_by_model = {}

for name, filename in model_files.items():
    path = MODEL_DIR / filename

    if not path.exists():
        print("Missing model:", path)
        continue

    clf = joblib.load(path)
    predictions = clf.predict(X_test)
    predictions = np.asarray(predictions)

    # Models trained with ham/spam labels
    predictions = np.where(predictions == "spam", 1, predictions)
    predictions = np.where(predictions == "ham", 0, predictions)
    predictions = predictions.astype(int)

    predictions_by_model[name] = predictions


# V5 prediction
predictions_by_model["V5 PyTorch LSTM"] = predict_lstm(X_test)


# Evaluate all models on the identical test set
for name, predictions in predictions_by_model.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Spam Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Spam Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Spam F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False Positives": int(
            confusion_matrix(y_test, predictions)[0, 1]
        ),
        "False Negatives": int(
            confusion_matrix(y_test, predictions)[1, 0]
        ),
    })

    print("\n", name)
    print(confusion_matrix(y_test, predictions))


comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Spam F1", ascending=False
)

print("\n========== SMS MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

output_path = REPORT_DIR / "sms_model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nSaved comparison:", output_path)
print("\nBest model by Spam F1:")
print(comparison.iloc[0]["Model"])
