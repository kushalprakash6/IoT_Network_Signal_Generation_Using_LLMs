import torch
import pandas as pd
from collections import Counter

# Define Tokenizer class
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


# Load the trained model and tokenizer from the checkpoint file
def load_model_and_tokenizer(model_path):
    # Load the checkpoint with the correct map_location and weights_only=False
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Initialize model
    model = NetworkTrafficTransformer(
        vocab_size=checkpoint['tokenizer'].vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dropout=DROPOUT
    )
    
    # Load model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieve tokenizer
    tokenizer = checkpoint['tokenizer']
    
    model.eval()  # Set model to evaluation mode
    return model, tokenizer


# Function to generate a signal (sequence of tokens)
def generate_signal(model, tokenizer, start_sequence, max_length=512):
    # Tokenize the start sequence
    tokens = tokenizer.tokenize(start_sequence)
    tokens = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

    generated_tokens = tokens.squeeze(0).tolist()  # Flatten to list
    
    for _ in range(max_length - len(generated_tokens)):
        # Create padding mask
        padding_mask = (tokens == 0)
        
        # Forward pass through the model to generate the next token
        with torch.no_grad():
            output = model(tokens, src_key_padding_mask=padding_mask)
        
        # Get the next token prediction (logits)
        next_token_logits = output[0, -1]  # Last token's logits
        next_token = torch.argmax(next_token_logits).item()
        
        if next_token == tokenizer.word2idx['<EOS>']:
            break  # Stop if end of sequence token is generated
        
        generated_tokens.append(next_token)
        
        # Update the tokens for next prediction
        tokens = torch.tensor(generated_tokens).unsqueeze(0)
    
    return generated_tokens


# Function to convert token IDs back to text
def tokens_to_text(tokens, tokenizer):
    return ' '.join([tokenizer.idx2word.get(token, '<UNK>') for token in tokens])


# Main function to generate raw and processed signals
def generate_and_display_signals(model, tokenizer, start_sequence="Source", max_length=512):
    # Generate raw signal (tokens)
    raw_signal = generate_signal(model, tokenizer, start_sequence, max_length)
    
    # Convert raw signal (token IDs) to readable text
    processed_signal = tokens_to_text(raw_signal, tokenizer)
    
    # Print raw and processed signals
    print(f"Raw Signal (Token IDs): {raw_signal}")
    print(f"Processed Signal (Text): {processed_signal}")
    
    # Prepare DataFrame to save to CSV
    df = pd.DataFrame({
        'Raw Signal (Token IDs)': [raw_signal],
        'Processed Signal (Text)': [processed_signal]
    })
    
    # Save to CSV
    df.to_csv("generated_signals_tf4.csv", index=False)
    print("Signals saved to 'generated_signals.csv'")


# Load the trained model and tokenizer
model_path = "iot_traffic_transformer_4.pt"  # Model file path
model, tokenizer = load_model_and_tokenizer(model_path)

# Generate and display signals
generate_and_display_signals(model, tokenizer, start_sequence="Source", max_length=512)
