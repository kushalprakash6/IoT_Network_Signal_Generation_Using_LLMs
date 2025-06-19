# import csv
# import os
# import time
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Set model path
# model_path = "./granite3b_iot_final"

# # Load tokenizer and model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
#     model.to(device)
#     model.eval()
# except Exception as e:
#     print(f"Error loading model/tokenizer: {e}")
#     exit(1)

# # Output CSV file
# csv_file = "generated_network_signals_granite.csv"

# # Write header to CSV
# if not os.path.exists(csv_file):
#     with open(csv_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Source", "Source Port", "Destination", "Destination Port", "Protocol", "Length", "Info"])

# # Prompt template for generation
# PROMPT_TEMPLATE = """Generate a realistic network packet log in the format:
# Source: <IP>, Source Port: <PORT>, Destination: <IP>, Destination Port: <PORT>, Protocol: <PROTO>, Length: <LEN>, Info: <INFO>
# """

# # Regex pattern to extract fields
# PATTERN = re.compile(
#     r"Source:\s*([\d\.]+),\s*Source Port:\s*(\d+),\s*Destination:\s*([\d\.]+),\s*Destination Port:\s*(\d+),\s*Protocol:\s*(\w+),\s*Length:\s*(\d+),\s*Info:\s*(.*)",
#     re.IGNORECASE
# )

# # Function to parse generated output
# def parse_output(text):
#     match = PATTERN.search(text)
#     if match:
#         return match.groups()
#     return None

# # Loop to generate and save signals
# print("Generating network signals... Press Ctrl+C to stop.\n")
# try:
#     while True:
#         inputs = tokenizer(PROMPT_TEMPLATE, return_tensors="pt").to(device)

#         with torch.no_grad():
#             output = model.generate(
#                 **inputs,
#                 max_new_tokens=100,
#                 do_sample=True,
#                 top_p=0.9,
#                 temperature=0.8,
#                 pad_token_id=tokenizer.eos_token_id
#             )

#         generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#         generated_lines = generated_text.split('\n')
#         # print(generated_text)
#         print(generated_lines)

#         for line in generated_lines:
#             fields = parse_output(line)
#             if fields:
#                 print("🔹", fields)
#                 # Save to CSV
#                 with open(csv_file, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow(fields)

#         time.sleep(1)  # Delay between generations

# except KeyboardInterrupt:
#     print("\nStopped by user.")




import csv
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model path
model_path = "./granite3b_iot_final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    exit(1)

# Output CSV
csv_file = "granite_generated_network_signals.csv"

# Write headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Source Port", "Destination", "Destination Port", "Protocol", "Length", "Info"])

# Prompt template
PROMPT = """YOU MUST GENERATE ALL THE REQUESTED PARAMETERS!!! Generate a realistic network packet log in the format:
Source: <IP>, Source Port: <PORT>, Destination: <IP>, Destination Port: <PORT>, Protocol: <PROTO>, Length: <LEN>, Info: <INFO>
"""

# Regex to extract source/destination ports, protocol, and length
LOG_PATTERN = re.compile(
    r"\[(.*?)\]\s+(\d+)\s*>\s*(\d+).*?Len=(\d+)", re.IGNORECASE
)

# Generate and parse a single log entry
def generate_log():
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    lines = decoded.split('\n')

    for line in lines:
        match = LOG_PATTERN.search(line)
        if match:
            protocol, src_port, dst_port, length = match.groups()
            return {
                "Source": "N/A",
                "Source Port": src_port,
                "Destination": "N/A",
                "Destination Port": dst_port,
                "Protocol": protocol,
                "Length": length,
                "Info": line.strip()
            }
    return None

# Generate and save 100 samples
entries = []
NUM_ENTRIES = 2
print("Generating 100 network logs...")

for i in range(NUM_ENTRIES):
    entry = generate_log()
    if entry:
        print(f"{i+1}: {entry}")
        entries.append(entry)

# Save all to CSV
with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Source", "Source Port", "Destination", "Destination Port", "Protocol", "Length", "Info"])
    writer.writerows(entries)

print(f"\n✅ Done. {len(entries)} entries written to {csv_file}")
