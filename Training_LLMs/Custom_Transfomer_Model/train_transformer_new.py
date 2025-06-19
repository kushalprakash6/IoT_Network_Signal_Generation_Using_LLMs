import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter
import os

# === PARAMETERS ===
MAX_LEN = 128  # Max token length of "Info"
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
LOSS_THRESHOLD = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "transformer_iot_model.pth"

# === Load CSV ===
print("Loading dataset...")
df = pd.read_csv("cleaned_file.csv")  # Replace with your actual file
df = df.dropna(subset=["Info"])  # Remove rows with missing Info

# === Preprocessing ===
print("Preprocessing 'Info' column...")
def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens[:MAX_LEN]

df["tokens"] = df["Info"].apply(tokenize)

# Build vocab
print("Building vocabulary...")
all_tokens = [token for sublist in df["tokens"] for token in sublist]
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({token: i+2 for i, (token, _) in enumerate(Counter(all_tokens).most_common())})

def encode(tokens):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens] + [0] * (MAX_LEN - len(tokens))

df["input_ids"] = df["tokens"].apply(encode)

# === Dataset & DataLoader ===
class IoTDataset(Dataset):
    def __init__(self, data):
        self.data = data["input_ids"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x  # Autoencoder-style (reconstruct input)

print("Creating Dataset and DataLoaders...")
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = IoTDataset(train_data)
val_dataset = IoTDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Transformer Model ===
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    # def forward(self, x):
    #     embedded = self.embedding(x)  # [B, T, D]
    #     embedded = embedded.permute(1, 0, 2)  # Transformer expects [T, B, D]
    #     transformed = self.transformer(embedded)
    #     output = self.fc(transformed)  # [T, B, vocab_size]
    #     return output.permute(1, 0, 2)  # [B, T, vocab_size]

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, D]
        transformed = self.transformer(embedded)  # [B, T, D]
        output = self.fc(transformed)  # [B, T, vocab_size]
        return output

print("Initializing Transformer model...")
model = TransformerModel(vocab_size=len(vocab), embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
model = model.to(DEVICE)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
print("Starting training...")
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         inputs, targets = [b.to(DEVICE) for b in batch]
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

#     # Early stopping if loss is low enough
#     if avg_loss < LOSS_THRESHOLD:
#         print("Loss threshold reached. Stopping training.")
#         break

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs: [B, T, vocab_size]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    if avg_loss < LOSS_THRESHOLD:
        print("Loss threshold reached. Stopping training.")
        break

# === Save Model ===
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Training complete.")
