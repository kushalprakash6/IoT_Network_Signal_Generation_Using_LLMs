import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

print("🟢 Loading CSV data...")
df = pd.read_csv("cleaned_file.csv")

print("🟢 Formatting data...")
def format_example(row):
    return (
        f"[INSTRUCTION] Generate a network signal log simulating behavior or attacks.\n"
        f"[INPUT] Time: {row['Time']}, Source: {row['Source']}:{row['Source port']}, "
        f"Destination: {row['Destination']}:{row['Destination port']}, Protocol: {row['Protocol']}, "
        f"Length: {row['Length']}, Info: {row['Info']}, Date: {row['Date time']}\n"
        f"[OUTPUT]"
    )

texts = [format_example(row) for _, row in df.iterrows()]
dataset = Dataset.from_dict({"text": texts})

print("🟢 Loading tokenizer and model...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Add custom token if necessary
special_tokens = {"additional_special_tokens": ["[INSTRUCTION]", "[INPUT]", "[OUTPUT]"]}
tokenizer.add_special_tokens(special_tokens)

# ✅ Fix missing pad_token error
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))

print("🟢 Tokenizing dataset...")
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

print("🟢 Setting training arguments...")
# training_args = TrainingArguments(
#     output_dir="./phi2-finetuned",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     num_train_epochs=3,
#     fp16=True,
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir='./logs',
#     logging_steps=50,
#     report_to="none",  # Avoid needing wandb etc.
#     evaluation_strategy="no",
# )

training_args = TrainingArguments(
    output_dir="./phi2-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=False,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",  # Avoid needing wandb etc.
)


print("🟢 Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🚀 Starting training...")
trainer.train()

print("✅ Training complete. Model saved to:", training_args.output_dir)
