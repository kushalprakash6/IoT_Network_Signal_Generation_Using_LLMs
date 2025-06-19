import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
from sklearn.model_selection import train_test_split
from collections import Counter

print("Starting IoT Network Traffic Transformer Training Pipeline")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
MAX_SEQ_LENGTH = 512  # Maximum sequence length
BATCH_SIZE = 32
EPOCHS = 20
D_MODEL = 256  # Embedding size
N_HEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 0.0001
SAVE_PATH = "iot_traffic_transformer_4.pt"

# Step 1: Data Loading and Preprocessing
print("Step 1: Loading and preprocessing data...")

def load_data(csv_path):
    """Load CSV data and preprocess it."""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select relevant columns for tokenization
    features = ['Source', 'Source port', 'Destination', 'Protocol', 
                'Destination port', 'Length', 'Info']
    
    # Check if all required columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    print(f"Selected {len(features)} features for tokenization")
    return df, features

class Tokenizer:
    """Custom tokenizer for network traffic data."""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_counter = Counter()
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        print("Building vocabulary...")
        # Count word frequencies
        for text in texts:
            if isinstance(text, (int, float)):
                text = str(text)
            words = str(text).split()
            self.word_counter.update(words)
        
        # Select top words based on frequency
        most_common = self.word_counter.most_common(self.vocab_size - 4)  # -4 for special tokens
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary built with {len(self.word2idx)} tokens")
        self.vocab_built = True
    
    def tokenize(self, text):
        """Convert text to token IDs."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        if isinstance(text, (int, float)):
            text = str(text)
        
        words = str(text).split()
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate or pad to MAX_SEQ_LENGTH
        if len(tokens) > MAX_SEQ_LENGTH - 2:  # -2 for SOS and EOS
            tokens = tokens[:MAX_SEQ_LENGTH - 2]
        
        # Add SOS and EOS tokens
        tokens = [self.word2idx['<SOS>']] + tokens + [self.word2idx['<EOS>']]
        
        # Pad sequence
        padding_length = MAX_SEQ_LENGTH - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.word2idx['<PAD>']] * padding_length
        
        return tokens
    
    def __len__(self):
        return len(self.word2idx)

# Step 2: Create dataset and dataloaders
class NetworkTrafficDataset(Dataset):
    """Dataset for network traffic data."""
    def __init__(self, data, features, tokenizer, target_col='Info'):
        self.data = data
        self.features = features
        self.tokenizer = tokenizer
        self.target_col = target_col
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize input features and concatenate
        input_tokens = []
        for feature in self.features:
            if feature != self.target_col:  # exclude target from input
                tokens = self.tokenizer.tokenize(row[feature])
                input_tokens.extend(tokens[:MAX_SEQ_LENGTH // len(self.features)])
        
        # Ensure input doesn't exceed MAX_SEQ_LENGTH
        input_tokens = input_tokens[:MAX_SEQ_LENGTH]
        
        # Pad input if necessary
        padding_length = MAX_SEQ_LENGTH - len(input_tokens)
        if padding_length > 0:
            input_tokens = input_tokens + [self.tokenizer.word2idx['<PAD>']] * padding_length
        
        # Tokenize target (Info column)
        target_tokens = self.tokenizer.tokenize(row[self.target_col])
        
        return torch.tensor(input_tokens), torch.tensor(target_tokens)

def prepare_data(df, features, test_size=0.2):
    """Prepare dataset and dataloaders."""
    print("Preparing dataset and dataloaders...")
    
    # Extract all text data for tokenizer training
    all_texts = []
    for feature in features:
        all_texts.extend(df[feature].astype(str).tolist())
    
    # Create and train tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(all_texts)
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    # Create datasets
    train_dataset = NetworkTrafficDataset(train_df, features, tokenizer)
    val_dataset = NetworkTrafficDataset(val_df, features, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Created dataloaders with batch size {BATCH_SIZE}")
    return train_loader, val_loader, tokenizer

# Step 3: Define Transformer Model
print("Step 3: Defining transformer model architecture...")

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NetworkTrafficTransformer(nn.Module):
    """Transformer model for network traffic analysis."""
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dropout):
        super(NetworkTrafficTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Output layer
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len]
        """
        # Create padding mask (True for padding positions)
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == 0)  # 0 is <PAD> token
        
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Linear layer to get logits
        output = self.decoder(output)
        
        return output

# Step 4: Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """Train the model and validate."""
    print("Step 4: Starting model training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Create padding mask
            src_padding_mask = (src == 0)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, src_padding_mask)
            
            # Calculate loss (only on non-padding tokens)
            output = output.view(-1, output.size(-1))
            tgt = tgt.view(-1)
            loss = criterion(output, tgt)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_padding_mask = (src == 0)
                output = model(src, src_padding_mask)
                output = output.view(-1, output.size(-1))
                tgt = tgt.view(-1)
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer,
                'val_loss': val_loss,
            }, SAVE_PATH)
            print(f"Model saved to {SAVE_PATH} with validation loss: {val_loss:.4f}")
        
        # Early stopping if loss doesn't improve
        if epoch > 15 and val_loss > best_val_loss:
            print("Early stopping as validation loss is not improving")
            break

# Main execution
if __name__ == "__main__":
    # Example usage - replace with your CSV path
    csv_path = "cleaned_file.csv"  # Update this with your CSV file path
    
    try:
        # Step 1: Load and preprocess data
        df, features = load_data(csv_path)
        
        # Step 2: Prepare datasets and dataloaders
        train_loader, val_loader, tokenizer = prepare_data(df, features)
        
        # Step 3: Initialize model
        vocab_size = len(tokenizer)
        print(f"Initializing transformer with vocabulary size: {vocab_size}")
        model = NetworkTrafficTransformer(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Step 4: Define loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens (index 0)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Step 5: Train model
        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
        
        print(f"Training completed. Final model saved to {SAVE_PATH}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")