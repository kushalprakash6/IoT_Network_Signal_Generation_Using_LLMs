import torch
import torch.nn as nn
import pandas as pd
import random
from datetime import datetime
import csv

# === CONFIGURATION ===
NUM_SAMPLES = 15  # Change to desired number
MAX_GEN_LEN = 128
MODEL_PATH = "transformer_iot_model.pth"
OUTPUT_CSV = "generated_iot_signals_newtf.csv"

# Dummy IPs and Protocols for generation
IP_POOL = [f"192.168.0.{i}" for i in range(1, 255)]
PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "DNS", "MQTT"]
PORT_RANGE = range(1024, 65535)

DEVICE = torch.device("cpu")  # Run on CPU

# === Load Vocab ===
print("Loading vocabulary...")
# Assume vocab saved earlier as a .pt file
vocab = torch.load("vocab.pt")  # You must save this during training
inv_vocab = {v: k for k, v in vocab.items()}

# === Define the Transformer Model (must match training architecture) ===
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, D]
        transformed = self.transformer(embedded)
        output = self.fc(transformed)  # [B, T, vocab_size]
        return output

# === Load Model ===
print("Loading model...")
model = TransformerModel(vocab_size=len(vocab), embed_dim=128, num_heads=4, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Helper: Sample next token ===
def sample_next_token(logits):
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

# === Generate one Info string ===
def generate_info_sequence(start_token="<UNK>", max_len=MAX_GEN_LEN):
    input_ids = torch.tensor([[vocab.get(start_token, vocab["<UNK>"])]], dtype=torch.long, device=DEVICE)
    for _ in range(max_len - 1):
        with torch.no_grad():
            output = model(input_ids)
        next_token_logits = output[0, -1]  # Last token's output
        next_token_id = sample_next_token(next_token_logits)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
        if inv_vocab[next_token_id] == "<PAD>":
            break
    tokens = [inv_vocab.get(tok.item(), "<UNK>") for tok in input_ids[0]]
    return " ".join(tokens)

# === Generate full network signal ===
def generate_signal():
    source = random.choice(IP_POOL)
    destination = random.choice(IP_POOL)
    while destination == source:
        destination = random.choice(IP_POOL)
    src_port = random.randint(1024, 65535)
    dst_port = random.randint(1024, 65535)
    protocol = random.choice(PROTOCOLS)
    info = generate_info_sequence()
    length = len(info)  # Use length of info string as packet length
    return {
        "Source": source,
        "Source port": src_port,
        "Destination": destination,
        "Destination port": dst_port,
        "Protocol": protocol,
        "Length": length,
        "Info": info
    }

# === Generate and Save ===
print(f"Generating {NUM_SAMPLES} synthetic IoT signals...")
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = ["Source", "Source port", "Destination", "Destination port", "Protocol", "Length", "Info"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(NUM_SAMPLES):
        signal = generate_signal()
        print(f"[{i+1}] Generated Signal: {signal}")
        writer.writerow(signal)

print(f"\n✅ Done. Saved generated signals to '{OUTPUT_CSV}'.")
