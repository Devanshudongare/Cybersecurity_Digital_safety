
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

SEED = 42
MAX_VOCAB = 15000
MAX_LENGTH = 100
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DATA_PATH = Path("data/raw/sms/SMSSpamCollection")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(
    DATA_PATH, sep="\t", header=None,
    names=["label", "message"], encoding="utf-8"
)
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna().reset_index(drop=True)

print("Dataset:", len(df))
print(df["label"].value_counts())

# Tokenization
def tokenize(text):
    return re.findall(
        r"[a-z]+|[0-9]+|[^\w\s]",
        str(text).lower()
    )

counter = Counter()
for message in df["message"]:
    counter.update(tokenize(message))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(MAX_VOCAB - 2):
    vocab[word] = len(vocab)

def encode_text(text):
    ids = [vocab.get(token, 1) for token in tokenize(text)]
    ids = ids[:MAX_LENGTH]
    return ids if ids else [1]

X = [encode_text(m) for m in df["message"]]
y = df["label"].astype(int).tolist()

# Train / validation / test split
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.15, random_state=SEED, stratify=y_train_all
)

class SMSDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = torch.tensor(self.sequences[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return seq, label, len(seq)

def collate_batch(batch):
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    return padded, torch.stack(labels), torch.tensor(lengths)

train_loader = DataLoader(
    SMSDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

# Improved LSTM with packed sequences
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

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)

model = SMSLSTM(
    len(vocab), EMBED_DIM, HIDDEN_DIM
).to(DEVICE)

# Handle class imbalance
positive_count = sum(y_train)
negative_count = len(y_train) - positive_count

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

print("\nTraining SMS LSTM V5 on CPU...")
print("Positive class weight:", round(pos_weight.item(), 3))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, labels, lengths in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Probability prediction
def predict_probabilities(sequences):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for sequence in sequences:
            tensor = torch.tensor(
                sequence, dtype=torch.long
            ).unsqueeze(0).to(DEVICE)

            lengths = torch.tensor([len(sequence)])

            probability = torch.sigmoid(
                model(tensor, lengths)
            ).item()

            probabilities.append(probability)

    return np.array(probabilities)

# Tune threshold on validation set only
val_prob = predict_probabilities(X_val)

thresholds = np.arange(0.30, 0.91, 0.02)
best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, pred, zero_division=0
    )
    recall = recall_score(
        y_val, pred, zero_division=0
    )
    f1 = f1_score(
        y_val, pred, zero_division=0
    )

    # Prefer precision >= 90% while preserving recall
    if precision >= 0.90 and f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

if best_f1 < 0:
    print("No threshold achieved 90% validation precision.")
    best_f1 = -1

    for threshold in thresholds:
        pred = (val_prob >= threshold).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

print("\nValidation threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))

# Final evaluation on untouched test set
test_prob = predict_probabilities(X_test)
y_pred = (test_prob >= best_threshold).astype(int)

print("\n========== SMS LSTM V5 RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Spam precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Spam recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))
print("Spam F1:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save V5 separately
model_path = MODEL_DIR / "sms_lstm_v5.pt"
vocab_path = MODEL_DIR / "sms_lstm_v5_vocab.json"

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": len(vocab),
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM,
    "max_length": MAX_LENGTH,
    "threshold": best_threshold,
    "seed": SEED,
}, model_path)

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("\nSaved:", model_path)
print("Saved:", vocab_path)
print("V1, V2, V3 and V4 remain unchanged.")
