import torch
from transformers import (
    LlamaTokenizer,  # Explicitly use LlamaTokenizer
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

# Set the device to MPS (Apple GPU) or CPU
device = torch.device(DEVICE if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)  # Use LlamaTokenizer explicitly

# Set a padding token (LLaMA models often do not have one)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Ensure the model also recognizes the new padding token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to include the new pad token

# Move model to MPS
model = model.to(device)

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)  # Move model to MPS

# Load the training data from a CSV file
def load_csv_data(file_path):
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Combine relevant columns into a single string per row
    packet_data = df.apply(
        lambda row: f"Source: {row['Source']} Source Port: {row['Source port']} "
                    f"Destination: {row['Destination']} Destination Port: {row['Destination port']} "
                    f"Protocol: {row['Protocol']} Length: {row['Length']} Info: {row['Info']}",
        axis=1
    ).tolist()
    
    return packet_data

# Load the training data
text_data = load_csv_data(TRAIN_FILE)

# Convert the text data into a Hugging Face Dataset
dataset = Dataset.from_dict({"text": text_data})

# # Tokenize the dataset
def tokenize_function(examples):
    tokenized_outputs = tokenizer(
        examples["text"],  
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None  # Change this from "pt" to None
    )
    
    # Ensure "input_ids" and "attention_mask" are correctly set
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"][:]  # Copy as list

    return tokenized_outputs


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# Rename "input_ids" to "labels" for causal language modeling
# tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")


# Define training arguments
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
    fp16=False,  # Disable mixed precision (not supported on MPS)
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")
