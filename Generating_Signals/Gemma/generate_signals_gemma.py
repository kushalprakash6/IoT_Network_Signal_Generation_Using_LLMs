# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import random

# # Load the trained model and tokenizer
# model_path = "./output_gemma"  # Path to your fine-tuned model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
# model.eval()  # Set the model to evaluation mode

# # Force the model and tensors to use CPU only
# device = torch.device("cpu")
# model = model.to(device)

# # Define a function to generate network signals
# def generate_network_signal(prompt, max_length=100):
#     # Tokenize the input prompt and generate the output
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to(device)  # Move input_ids to the CPU
    
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    
#     # Decode the generated sequence and return it
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return generated_text

# # Define a function to create synthetic network signals
# def generate_synthetic_signals(num_signals=100):
#     generated_data = []
#     for _ in range(num_signals):
#         # Generate random parameters for Source, Source port, Destination, etc.
#         source_ip = f"192.168.1.{random.randint(1, 255)}"
#         destination_ip = f"192.168.1.{random.randint(1, 255)}"
#         source_port = random.randint(1024, 65535)
#         destination_port = random.randint(1024, 65535)
#         protocol = random.choice(["TCP", "UDP", "ICMP"])
#         length = random.randint(40, 1500)  # Common packet sizes
#         info = random.choice(["Normal", "Attack", "Suspected Malicious"])

#         # Create a prompt based on these parameters
#         # prompt = f"Source: {source_ip} Source Port: {source_port} Destination: {destination_ip} Destination Port: {destination_port} Protocol: {protocol} Length: {length} Info: {info}"
#         prompt = ("Generate network signals\n")

#         # Generate synthetic signal using the model
#         generated_signal = generate_network_signal(prompt)

#         # Parse the generated signal into relevant fields
#         generated_data.append({
#             "Source": source_ip,
#             "Source port": source_port,
#             "Destination": destination_ip,
#             "Destination port": destination_port,
#             "Protocol": protocol,
#             "Length": length,
#             "Info": info,
#         })
    
#     # Convert the generated data to a pandas DataFrame
#     df = pd.DataFrame(generated_data)
#     return df

# # Generate synthetic signals (you can change the number of signals as needed)
# num_signals = 100  # Adjust the number of generated signals
# generated_df = generate_synthetic_signals(num_signals)

# # Save the generated signals to a CSV file
# output_file = "generated_network_signals_gemma.csv"
# generated_df.to_csv(output_file, index=False)
# print(f"[INFO] Generated network signals saved to {output_file}")



import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time

# Load the trained model and tokenizer
model_path = "./output_gemma"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Force the model and tensors to use CPU only
device = torch.device("cpu")
model = model.to(device)

# Define a function to generate network signals
def generate_network_signal(prompt, max_length=100):
    # Tokenize the input prompt and generate the output
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)  # Move input_ids to the CPU
    
    with torch.no_grad():
        start_time = time.time()  # Start the timer
        output = model.generate(
            input_ids,
            max_length=max_length,  # Adjusted max_length for faster generation
            do_sample=True,
            top_k=50,  # Reducing randomness
            top_p=0.9,  # Reducing randomness
            temperature=0.7  # Adding temperature for control
        )
        end_time = time.time()  # End the timer
        print(f"[INFO] Generation time for one signal: {end_time - start_time:.4f} seconds")
    
    # Decode the generated sequence and return it
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Define a function to create synthetic network signals
def generate_synthetic_signals(num_signals=100):
    generated_data = []
    for i in range(num_signals):
        # Generate random parameters for Source, Source port, Destination, etc.
        source_ip = f"192.168.1.{random.randint(1, 255)}"
        destination_ip = f"192.168.1.{random.randint(1, 255)}"
        source_port = random.randint(1024, 65535)
        destination_port = random.randint(1024, 65535)
        protocol = random.choice(["TCP", "UDP", "ICMP"])
        length = random.randint(40, 1500)  # Common packet sizes
        info = random.choice(["Normal", "Attack", "Suspected Malicious"])

        # Create a prompt based on these parameters
        prompt = f"Source: {source_ip} Source Port: {source_port} Destination: {destination_ip} Destination Port: {destination_port} Protocol: {protocol} Length: {length} Info: {info}"
        #prompt = ("Generate network signals\n")

        # Generate synthetic signal using the model
        generated_signal = generate_network_signal(prompt)

        # Print the generated signal in real-time
        print(f"Generated Signal {i + 1}: {generated_signal}")
        
        # Parse the generated signal into relevant fields
        generated_data.append({
            "Source": source_ip,
            "Source port": source_port,
            "Destination": destination_ip,
            "Destination port": destination_port,
            "Protocol": protocol,
            "Length": length,
            "Info": info,
        })
    
    # Convert the generated data to a pandas DataFrame
    df = pd.DataFrame(generated_data)
    return df

# Generate synthetic signals (you can change the number of signals as needed)
num_signals = 100  # Adjust the number of generated signals
generated_df = generate_synthetic_signals(num_signals)

# Save the generated signals to a CSV file
output_file = "generated_network_signals.csv"
generated_df.to_csv(output_file, index=False)
print(f"[INFO] Generated network signals saved to {output_file}")

