import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Configuration
# =========================
model_dir = "./Qwen_IEEE/model_output_qwen_IEEE"           # Path to your trained model/tokenizer
output_csv = "./Qwen_IEEE/generated_network_sig_qwen_ieee_36.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 5000                         # How many signals to generate
max_generate_length = 128                # Max tokens per generated sample
# Sampling parameters
temperature = 0.3
top_p = 0.95

print("[Stage] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to(device)
model.eval()

generated_rows = []
print(f"[Stage] Generating {num_samples} network signals...")
for i in range(num_samples):
    prompt = "Time:"  # or any starting template
    # Tokenize and get attention mask
    encoding = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_generate_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[Sample {i+1}] Raw generated text:\n{text}\n")

    # Parse fields separated by ' | '
    parts = [p.strip() for p in text.split('|')]
    row = {
        "No.": i + 1,
        "Time": None,
        "Source": None,
        "Source port": None,
        "Destination": None,
        "Destination port": None,
        "proto": None,
        "Length": None,
        "Info": None,
        "Date time": None
    }

    for part in parts:
        # Time
        if part.startswith("Time:"):
            row['Time'] = part.split(':', 1)[1].strip()

        # Source and Destination together
        elif part.startswith("Src:"):
            m = re.match(
                r"^Src:\s*([\d\.]+):([0-9]+)(?:\.[0-9]+)?\s*->\s*Dst:\s*([\d\.]+):([0-9]+)(?:\.[0-9]+)?$",
                part
            )
            if m:
                src_ip, src_port, dst_ip, dst_port = m.groups()
                row['Source'] = src_ip
                row['Source port'] = int(src_port)
                row['Destination'] = dst_ip
                row['Destination port'] = int(dst_port)

        # Fallback Dst only
        elif part.startswith("Dst:"):
            m = re.match(r"^Dst:\s*([\d\.]+):([0-9]+)(?:\.[0-9]+)?$", part)
            if m:
                dst_ip, dst_port = m.groups()
                row['Destination'] = dst_ip
                row['Destination port'] = int(dst_port)

        # Protocol as 'proto'
        elif part.lower().startswith("proto:"):
            row['proto'] = part.split(':', 1)[1].strip()

        # Length
        elif part.startswith("Len:") or part.startswith("Length:"):
            val = part.split(':', 1)[1].strip()
            row['Length'] = int(val) if val.isdigit() else None

        # Info
        elif part.startswith("Info:"):
            row['Info'] = part.split(':', 1)[1].strip()

        # Date time
        elif part.startswith("Date:"):
            row['Date time'] = part.split(':', 1)[1].strip()

    generated_rows.append(row)

# Build DataFrame with all expected columns
columns = [
    'No.', 'Time', 'Source', 'Source port', 'Destination', 'Destination port',
    'proto', 'Length', 'Info', 'Date time'
]
df = pd.DataFrame(generated_rows, columns=columns)

print("[Stage] Generated DataFrame:")
print(df.to_string(index=False))

print(f"[Stage] Writing to CSV: {output_csv}")
df.to_csv(output_csv, index=False)
print("[Done] CSV saved.")
