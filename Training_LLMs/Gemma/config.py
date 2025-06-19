# config.py

# Model and Tokenizer Configuration
MODEL_NAME = "google/gemma-2b"  # Pretrained Gemma 2B model

# File Paths
TRAIN_FILE = "cleaned_file.csv"            # Your cleaned CSV with IoT data
OUTPUT_DIR = "./output_gemma"         # Directory to save model/tokenizer

# Training Configuration
BATCH_SIZE = 2                         # You can increase if memory allows (e.g., 4 for A100)
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 256
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = "cuda"                        # Automatically detects GPU; fallback handled in script
