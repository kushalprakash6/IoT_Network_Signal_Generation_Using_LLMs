# Model and Tokenizer Configuration
MODEL_NAME = "openlm-research/open_llama_3b"  # Or another pre-trained model you wish to fine-tune

# File Paths
TRAIN_FILE = "packets.csv"  # Path to your CSV file containing the training data
OUTPUT_DIR = "./output"  # Directory where the fine-tuned model and tokenizer will be saved

# Training Configuration
BATCH_SIZE = 2  # Adjust according to your GPU memory
EPOCHS = 3  # Number of epochs for fine-tuning
LEARNING_RATE = 5e-5  # Learning rate
MAX_SEQ_LENGTH = 256  # Maximum sequence length (you may adjust this)
GRADIENT_ACCUMULATION_STEPS = 2  # Steps to accumulate gradients before updating
DEVICE = "cpu"  # Set to 'cpu' if you're not using an Apple M1/M2 chip with Metal Performance Shaders

