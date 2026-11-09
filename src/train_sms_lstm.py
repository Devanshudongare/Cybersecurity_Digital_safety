
import re
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------- Configuration ----------------
SEED = 42
MAX_VOCAB = 15000
MAX_LENGTH = 100
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DATA_PATH = Path("data/raw/sms/SMSSpamCollection")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- Load dataset ----------------
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

print("Dataset rows:", len(df))
print("Class counts:")
print(df["label"].value_counts())

# ---------------- Tokenization ----------------
def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", text)

counter = Counter()
for message in df["message"]:
    counter.update(tokenize(message))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(MAX_VOCAB - 2):
    vocab[word] = len(vocab)

def encode_text(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, 1) for token in tokens]
    ids = ids[:MAX_LENGTH]
    if not ids:
        ids = [1]
    return ids

X = [encode_text(text) for text in df["message"]]
y = df["label"].astype(int).tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# ---------------- Dataset ----------------
class SMSDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32)
        )

def collate_batch(batch):
    sequences, labels = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )
    return padded, torch.stack(labels)

train_loader = DataLoader(
    SMSDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

# ---------------- LSTM model ----------------
class SMSLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]
        return self.fc(self.dropout(last_hidden)).squeeze(1)

model = SMSLSTM(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ---------------- Training ----------------
print("\nTraining PyTorch LSTM on CPU...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ---------------- Evaluation ----------------
def predict_sequences(sequences):
    model.eval()
    predictions = []

    with torch.no_grad():
        for sequence in sequences:
            tensor = torch.tensor(
                sequence, dtype=torch.long
            ).unsqueeze(0).to(DEVICE)

            probability = torch.sigmoid(model(tensor)).item()
            predictions.append(1 if probability >= 0.5 else 0)

    return predictions

y_pred = predict_sequences(X_test)

print("\n========== LSTM SMS RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Spam precision:", round(precision_score(y_test, y_pred), 4))
print("Spam recall:", round(recall_score(y_test, y_pred), 4))
print("Spam F1:", round(f1_score(y_test, y_pred), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# ---------------- Save model ----------------
model_path = MODEL_DIR / "sms_lstm_experiment.pt"
vocab_path = MODEL_DIR / "sms_lstm_vocab.json"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_length": MAX_LENGTH,
        "seed": SEED,
    },
    model_path
)

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("\nSaved experimental model:", model_path)
print("Saved vocabulary:", vocab_path)
print("Existing SMS models were not modified.")
