import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# 1. Define Parameters
# -----------------------------
MODEL_PATH = "iot_transformer.pth"
OUTPUT_CSV = "generated_signals_transformer_10.csv"
SEQ_START_LEN = 20
GEN_STEPS = 500
DEVICE = "cpu"  # Force CPU for generation

# Feature mapping (must match training)
cat_cols = ['Source', 'Destination', 'Protocol', 'Info']
num_cols = ['Time', 'Source port', 'Destination port', 'Length']
all_cols = ['Time', 'Source', 'Source port', 'Destination', 'Protocol', 'Destination port', 'Length', 'Info']

# Dummy values for decoding (mock IDs)
source_decoder = {i: f"192.168.0.{i%255}" for i in range(1000)}
dest_decoder = {i: f"10.0.0.{i%255}" for i in range(1000)}
protocol_decoder = {0: 'TCP', 1: 'UDP', 2: 'ICMP'}
info_decoder = {i: f"Info_{i}" for i in range(1000)}

# -----------------------------
# 2. Define Model Architecture
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

# -----------------------------
# 3. Load Model
# -----------------------------
print("[INFO] Loading model...")
model_dim = 64
input_dim = 8  # Should match training features
model = SimpleTransformer(input_dim=input_dim, model_dim=model_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Model loaded.")

# -----------------------------
# 4. Seed Input
# -----------------------------
print("[INFO] Creating dummy seed sequence...")
start_seed = torch.randn(SEQ_START_LEN, input_dim).to(DEVICE)

# -----------------------------
# 5. Generate Signals
# -----------------------------
def generate_sequence(model, start_seq, steps=100):
    print("[INFO] Generating sequence...")
    generated = start_seq.clone()
    for _ in range(steps):
        with torch.no_grad():
            out = model(generated.unsqueeze(0))  # (1, seq, dim)
        next_step = out[0, -1].unsqueeze(0)
        generated = torch.cat([generated, next_step], dim=0)
    return generated.cpu().numpy()

generated_seq = generate_sequence(model, start_seed, steps=GEN_STEPS)
print(f"[INFO] Generated shape: {generated_seq.shape}")

# -----------------------------
# 6. Convert to Table Format
# -----------------------------
print("[INFO] Converting to human-readable format...")
rows = []
start_time = datetime.now()

for idx, row in enumerate(generated_seq):
    # print(row)
    # Map back approximate values
    time = abs(row[0]) * 100  # Just scale time
    source = source_decoder.get(int(abs(row[1]) * 100) % 1000, "192.168.0.1")
    sport = int(abs(row[2]) * 10000) % 65535
    destination = dest_decoder.get(int(abs(row[3]) * 100) % 1000, "10.0.0.1")
    proto = protocol_decoder.get(int(abs(row[4]) * 10) % 3, "TCP")
    dport = int(abs(row[5]) * 10000) % 65535
    length = max(1, int(abs(row[6]) * 1500))  # typical MTU
    info = info_decoder.get(int(abs(row[7]) * 1000) % 1000, "Info_0")
    timestamp = start_time + timedelta(seconds=float(idx))

    print(f"[{idx+1}] {time:.2f}s | {source}:{sport} -> {destination}:{dport} [{proto}] len={length} info={info}")
    rows.append([idx+1, time, source, sport, destination, proto, dport, length, info, timestamp.strftime("%Y-%m-%d %H:%M:%S")])

# -----------------------------
# 7. Save to CSV
# -----------------------------
df_out = pd.DataFrame(rows, columns=[
    "No.", "Time", "Source", "Source port", "Destination", "Protocol",
    "Destination port", "Length", "Info", "Date time"
])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Written {len(rows)} rows to {OUTPUT_CSV}")
