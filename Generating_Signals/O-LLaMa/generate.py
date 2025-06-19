import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
import pandas as pd
import re
from config import OUTPUT_DIR, DEVICE  # Assumes you have a config.py with OUTPUT_DIR and DEVICE

# Set device (MPS, CUDA, or CPU)
device = torch.device(DEVICE if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(OUTPUT_DIR)

# Add padding token if missing (LLaMA usually doesn't have one)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))  # Ensure pad token is accounted for
model = model.to(device)

# Inference prompt
#prompt = "Generate network traffic:\n I need the following details in the generated traffic - Source, Source Port, Destination, Destination Port, Protocol, Length, Info"
prompt = "Generate network traffic:\n "

# Tokenize prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate text samples
outputs = model.generate(
    input_ids=input_ids,
    max_length=192,
    num_return_sequences=150,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)

# Decode outputs
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Show generated text for preview
print("\nGenerated Samples:")
for i, txt in enumerate(generated_texts):
    print(f"[{i+1}]\n{txt}\n")

packet_pattern = re.compile(
    r"Source: (?P<Source>.*?)\s+Source Port: (?P<Source_port>.*?)\s+"
    r"Destination: (?P<Destination>.*?)\s+Destination Port: (?P<Destination_port>.*?)\s+"
    r"Protocol: (?P<Protocol>.*?)\s+Length: (?P<Length>.*?)\s+Info: (?P<Info>.+)"
)


# Parse function
def parse_packet_text(text):
    match = packet_pattern.search(text)
    if match:
        return {
            'Source': match.group("Source").strip(),
            'Source port': match.group("Source_port").strip(),
            'Destination': match.group("Destination").strip(),
            'Destination port': match.group("Destination_port").strip(),
            'Protocol': match.group("Protocol").strip(),
            'Length': match.group("Length").strip(),
            'Info': match.group("Info").strip()
        }
    return None
    


# Apply parser
parsed_packets = [parse_packet_text(t) for t in generated_texts]
parsed_packets = [p for p in parsed_packets if p is not None]

# Save unparsed examples
if len(parsed_packets) < len(generated_texts):
    with open("unparsed_samples.log", "w") as f:
        for t in generated_texts:
            if parse_packet_text(t) is None:
                f.write(t + "\n---\n")
    print(f"⚠️ Some samples could not be parsed. Saved unparsed examples to 'unparsed_samples.log'")

# Export to CSV
if parsed_packets:
    df = pd.DataFrame(parsed_packets)
    output_file = "generated_signals_lowtemp_120.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Parsed and saved {len(df)} samples to: {output_file}")
else:
    print("\n⚠️ No valid samples parsed. Try adjusting the prompt or generation settings.")
