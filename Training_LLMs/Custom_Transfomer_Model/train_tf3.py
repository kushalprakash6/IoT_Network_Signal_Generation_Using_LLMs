# =========================
# Training Script: train_transformer_iot.py
# =========================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
MAX_LEN = 100
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
LOSS_THRESHOLD = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs_tf3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_iot_model_3.pth")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab_3.pt")

# === Load Dataset ===
df = pd.read_csv("cleaned_file.csv")
df = df.dropna(subset=["Source", "Source port", "Destination", "Destination port", "Protocol", "Length", "Info"])

def row_to_text(row):
    return (
        f"<Source> {row['Source']} "
        f"<SrcPort> {int(float(row['Source port']))} "
        f"<Destination> {row['Destination']} "
        f"<DstPort> {int(float(row['Destination port']))} "
        f"<Protocol> {row['Protocol']} "
        f"<Length> {int(float(row['Length']))} "
        f"<Info> {row['Info']} <EOS>"
    )

print("Creating structured text format...")
df["full_text"] = df.apply(row_to_text, axis=1)

print("Tokenizing text...")
def tokenize(text):
    return str(text).strip().split()[:MAX_LEN]

df["tokens"] = df["full_text"].apply(tokenize)

# === Build Vocabulary ===
from collections import Counter
all_tokens = [tok for tokens in df["tokens"] for tok in tokens]
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({tok: idx + 2 for idx, (tok, _) in enumerate(Counter(all_tokens).most_common())})
torch.save(vocab, VOCAB_PATH)

# === Encode Tokens ===
def encode(tokens):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    return ids[:MAX_LEN] + [vocab["<PAD>"]] * (MAX_LEN - len(ids))

df["input_ids"] = df["tokens"].apply(encode)

# === Dataset ===
class IoTDataset(Dataset):
    def __init__(self, data):
        self.data = data["input_ids"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = IoTDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

model = TransformerModel(len(vocab), EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training ===
print("Training model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    if avg_loss < LOSS_THRESHOLD:
        print("Early stopping: loss threshold reached.")
        break

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")