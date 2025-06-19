import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

# =========================
# Configuration
# =========================
csv_file = "cleaned_file.csv"           # Path to your CSV file
output_dir = "./model_output_qwen"   # Where to save model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2                   # Adjust based on your GPU memory
max_length = 512                 # Maximum token length per sample
learning_rate = 5e-5             # Learning rate for optimizer
num_warmup_steps = 100           # Warmup steps for scheduler
num_training_steps = 10000       # Total training steps (approximate)
patience = 3                     # Early stopping patience
# =========================

print("[Stage] Loading data...")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} samples from {csv_file}")

print("[Stage] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B",
    trust_remote_code=True
)

class IOTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = (
            f"Time: {row['Time']} | "
            f"Src: {row['Source']}:{row['Source port']} -> "
            f"Dst: {row['Destination']}:{row['Destination port']} | "
            f"Proto: {row['Protocol']} | "
            f"Len: {row['Length']} | "
            f"Info: {row['Info']} | "
            f"Date: {row['Date time']}"
        )
        # Tokenize with truncation only; padding handled by collator
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

print("[Stage] Creating Dataset...")
dataset = IOTDataset(df, tokenizer, max_length)

print("[Stage] Initializing Data Collator for dynamic padding...")
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

print("[Stage] Creating DataLoader...")
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator
)

print("[Stage] Initializing model config and model from scratch...")
config = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-3B",
    trust_remote_code=True
)
# Disable sliding-window attention to avoid SDPA warning
config.use_sliding_window = False
model = AutoModelForCausalLM.from_config(config)
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

best_loss = float('inf')
epochs_without_improve = 0
epoch = 0

print("[Stage] Starting training loop...")
while epochs_without_improve < patience:
    epoch += 1
    print(f"[Epoch {epoch}] Beginning...")
    running_loss = 0.0

    for step, batch in enumerate(dataloader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if step % 10 == 0:
            print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improve = 0
        print("[Info] Loss improved, saving model checkpoint...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        epochs_without_improve += 1
        print(f"[Info] No improvement for {epochs_without_improve} epoch(s)")

print("[Training Complete] Model training finished.")
