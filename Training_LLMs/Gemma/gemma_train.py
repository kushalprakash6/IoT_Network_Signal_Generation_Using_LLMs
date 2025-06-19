# gemma_train.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from config import (
    MODEL_NAME,
    TRAIN_FILE,
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    GRADIENT_ACCUMULATION_STEPS,
    DEVICE,
)

# Step 1: Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Step 2: Load Tokenizer
print(f"[INFO] Loading tokenizer for model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("[INFO] Tokenizer loaded and pad token set (if missing).")

# Step 3: Load Pretrained Model
print(f"[INFO] Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
print("[INFO] Model loaded and moved to device.")

# Step 4: Load and Prepare CSV Dataset
def load_csv_data(file_path):
    print(f"[INFO] Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded {len(df)} rows.")
    
    # Convert each row to a descriptive string
    return df.apply(
        lambda row: f"Source: {row['Source']} Source Port: {row['Source port']} "
                    f"Destination: {row['Destination']} Destination Port: {row['Destination port']} "
                    f"Protocol: {row['Protocol']} Length: {row['Length']} Info: {row['Info']}",
        axis=1
    ).tolist()

text_data = load_csv_data(TRAIN_FILE)
dataset = Dataset.from_dict({"text": text_data})
print("[INFO] Converted dataset to HuggingFace format.")

# Step 5: Tokenize Data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("[INFO] Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("[INFO] Tokenization complete.")

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="no",
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Step 8: Train the Model
print("[INFO] Starting training...")
trainer.train()
print("[INFO] Training completed.")

# Step 9: Save Fine-tuned Model and Tokenizer
print(f"[INFO] Saving model and tokenizer to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Gemma 2B fine-tuned model saved to {OUTPUT_DIR}")
