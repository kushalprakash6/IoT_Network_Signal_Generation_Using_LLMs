import csv
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, InfNanRemoveLogitsProcessor

# === CONFIG ===
model_dir = "./phi2-finetuned/checkpoint-12243"  # Path to your fine-tuned model directory
output_csv = "generated_network_signals_phi2.csv"
num_samples = 20
device = torch.device("cpu")  # Force CPU usage

# === LOAD MODEL & TOKENIZER ===
print("🔄 Loading model and tokenizer on CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

# Enable cache usage for generation to prevent recomputation
model.config.use_cache = True

# === LOGITS PROCESSOR TO REMOVE NaNs/Infs ===
logits_processor = LogitsProcessorList([
    InfNanRemoveLogitsProcessor(),  # Remove NaNs/Infs in logits before sampling
])

# === PROMPT TEMPLATE ===
prompt_template = (
    "Generate a realistic network log entry:\n"
    "Source: 192.168.1.10:1234, Destination: 192.168.1.20:80, Protocol: TCP, Length: 64, Info: SYN\n"
)

# === GENERATE FUNCTION ===
def generate_signal(prompt, max_new_tokens=100):
    # Tokenize and move to the correct device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,  # Slightly relaxed top-p
            temperature=1.0,  # Set temperature to avoid too deterministic results
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,  # Apply the NaN/Inf filter
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

# === PARSE LOG OUTPUT INTO FIELDS ===
def extract_fields(text):
    try:
        ip_port_pattern = r"(\d{1,3}(?:\.\d{1,3}){3}):(\d+)"
        src_match = re.search(r"Source[:\s]+" + ip_port_pattern, text)
        dst_match = re.search(r"Destination[:\s]+" + ip_port_pattern, text)
        proto_match = re.search(r"Protocol[:\s]+(\w+)", text)
        len_match = re.search(r"Length[:\s]+(\d+)", text)
        info_match = re.search(r"Info[:\s]+([\w\-]+)", text)

        if not all([src_match, dst_match, proto_match, len_match, info_match]):
            return None

        return {
            "Source": src_match.group(1),
            "Source port": src_match.group(2),
            "Destination": dst_match.group(1),
            "Destination port": dst_match.group(2),
            "Protocol": proto_match.group(1),
            "Length": len_match.group(1),
            "Info": info_match.group(1),
        }
    except Exception as e:
        print(f"❌ Parsing error: {e}")
        return None

# === MAIN LOOP ===
print(f"🚀 Generating {num_samples} samples...")
rows = []
for i in range(num_samples):
    print(f"▶️ Generating sample {i + 1}...")
    try:
        raw = generate_signal(prompt_template)
        print("🧾 Raw Output:", raw)
        parsed = extract_fields(raw)
        if parsed:
            rows.append(parsed)
        else:
            print("⚠️ Skipped: Output could not be parsed into structured fields.")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

# === SAVE TO CSV ===
if rows:
    print(f"💾 Writing {len(rows)} rows to {output_csv}")
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Source", "Source port", "Destination", "Destination port", "Protocol", "Length", "Info"])
        writer.writeheader()
        writer.writerows(rows)
    print("✅ Finished.")
else:
    print("❌ No valid samples to save.")
