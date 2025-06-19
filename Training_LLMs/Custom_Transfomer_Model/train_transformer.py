import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# -----------------------------
# Parameters
# -----------------------------
SEQ_LEN = 100
BATCH_SIZE = 64
MAX_EPOCHS = 500
TARGET_LOSS = 1e8  # stop when loss < 100 million
MODEL_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CSV_PATH = "cleaned_file.csv"

print(f"[INFO] Loading data from: {CSV_PATH}")

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
df = pd.read_csv(CSV_PATH).fillna("missing")

cat_cols = ['Source', 'Destination', 'Protocol', 'Info']
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

num_cols = ['Time', 'Source port', 'Destination port', 'Length']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

drop_cols = ['No.', 'Date time']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

features = df.to_numpy().astype(np.float32)
sequences = []
for i in range(0, len(features) - SEQ_LEN):
    seq = features[i:i + SEQ_LEN]
    sequences.append(seq)

X = torch.tensor(np.array(sequences))  # (N, seq_len, feat_dim)
print(f"[INFO] Dataset shape: {X.shape}")

# -----------------------------
# 2. Dataset
# -----------------------------
class IoTDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target

train_loader = DataLoader(IoTDataset(X), batch_size=BATCH_SIZE, shuffle=True)
print("[INFO] DataLoader ready.")

# -----------------------------
# 3. Transformer Model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_layers=4, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        memory = torch.zeros_like(x)
        out = self.transformer(x, memory, tgt_mask=tgt_mask)
        return self.output_proj(out.transpose(0, 1))

model = SimpleTransformer(input_dim=X.shape[2], model_dim=MODEL_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("[INFO] Starting training loop...")

# -----------------------------
# 4. Training Loop with Logging
# -----------------------------
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0
    print(f"\n[INFO] Epoch {epoch} ------------------")

    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f"[Batch {i}] Loss: {loss.item():.4f}")

    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}")

    if total_loss < TARGET_LOSS:
        print(f"[INFO] Loss threshold reached: {total_loss:.4f} < {TARGET_LOSS}")
        break

print("[INFO] Training complete. Saving model...")
torch.save(model.state_dict(), 'iot_transformer.pth')
print("[INFO] Model saved as 'iot_transformer.pth'")

# -----------------------------
# 5. Generate Example Sequence
# -----------------------------
def generate_sequence(model, start_seq, steps=50):
    model.eval()
    generated = start_seq.clone().to(DEVICE)

    for _ in range(steps):
        with torch.no_grad():
            out = model(generated.unsqueeze(0))
        next_step = out[0, -1].unsqueeze(0)
        generated = torch.cat([generated, next_step], dim=0)

    return generated.cpu().numpy()

start_seq = X[0, :20]  # starting sequence
generated_seq = generate_sequence(model, start_seq, steps=50)
print("[INFO] Generated sequence shape:", generated_seq.shape)
print(generate_sequence)