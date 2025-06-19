# =============================
# Generation Script: generate_transformer_iot.py
# =============================

import os
import torch
import torch.nn as nn
import csv

# === CONFIGURATION ===
NUM_SAMPLES = 10
MAX_GEN_LEN = 80
TEMPERATURE = 1.0
TOP_K = 40
OUTPUT_CSV = "generated_structured_signals_tf3.csv"

# === Paths ===
OUTPUT_DIR = "outputs_tf3"
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_iot_model_3.pth")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab_3.pt")

DEVICE = torch.device("cpu")

# === Load Vocabulary ===
vocab = torch.load(VOCAB_PATH)
inv_vocab = {idx: tok for tok, idx in vocab.items() if isinstance(tok, str)}

# === Transformer Model ===
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

# === Load Model ===
model = TransformerModel(len(vocab), 128, 4, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Sampling with Top-K ===
def sample_next_token(logits, top_k=TOP_K, temperature=TEMPERATURE):
    logits = logits / temperature
    top_logits, top_indices = torch.topk(logits, top_k)
    probs = torch.softmax(top_logits, dim=-1)
    return top_indices[torch.multinomial(probs, num_samples=1)].item()

def generate_sequence():
    input_ids = torch.tensor([[vocab.get("<BOS>", 1)]], dtype=torch.long, device=DEVICE)
    generated_tokens = []

    for _ in range(MAX_GEN_LEN):
        with torch.no_grad():
            output = model(input_ids)
        next_token_logits = output[0, -1]
        next_token_id = sample_next_token(next_token_logits)
        token = inv_vocab.get(next_token_id, "<UNK>")
        if token == "<EOS>":
            break
        generated_tokens.append(token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)], dim=1)

    return generated_tokens

# === Parse Tokens ===
def parse_tokens(tokens):
    signal = {
        "Source": "",
        "Source port": "",
        "Destination": "",
        "Destination port": "",
        "Protocol": "",
        "Length": "",
        "Info": ""
    }
    current_field = None
    field_map = {
        "<Source>": "Source",
        "<SrcPort>": "Source port",
        "<Destination>": "Destination",
        "<DstPort>": "Destination port",
        "<Protocol>": "Protocol",
        "<Length>": "Length",
        "<Info>": "Info"
    }
    for token in tokens:
        if token in field_map:
            current_field = field_map[token]
        elif current_field:
            signal[current_field] += (" " if signal[current_field] else "") + token

    for field in ["Source port", "Destination port", "Length"]:
        try:
            signal[field] = int(float(signal[field].strip()))
        except:
            signal[field] = 0

    if not signal["Source"] or not signal["Destination"]:
        print("⚠️ Warning: Incomplete or malformed output:", tokens)

    return signal

# === Generate and Save ===
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = ["Source", "Source port", "Destination", "Destination port", "Protocol", "Length", "Info"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(NUM_SAMPLES):
        tokens = generate_sequence()
        print(f"[Sample {i+1}] Tokens:", " ".join(tokens))
        record = parse_tokens(tokens)
        print(f"[Sample {i+1}] Record:", record)
        writer.writerow(record)

print(f"\n✅ Generated {NUM_SAMPLES} records to {OUTPUT_CSV}")
