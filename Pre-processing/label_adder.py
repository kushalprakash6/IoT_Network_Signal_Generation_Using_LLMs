import pandas as pd

def packet_to_text(row):
    """
    Converts a row of packet data into a human-readable text format.
    """
    text = f"Packet {row['No.']}:\n"
    text += f"- Time: {row['Time']}\n"
    text += f"- Source: {row['Source']}\n"
    #text += f"- Source Port: {int(row['Source port'])}\n"
    text += f"- Source Port: {int(row['Source port']) if not pd.isna(row['Source port']) else 'N/A'}\n"  # Handle NaN
    text += f"- Destination: {row['Destination']}\n"
    text += f"- Protocol: {row['Protocol']}\n"
    #text += f"- Destination Port: {int(row['Destination port'])}\n"
    text += f"- Destination Port: {int(row['Destination port']) if not pd.isna(row['Destination port']) else 'N/A'}\n"  # Handle NaN
    text += f"- Length: {row['Length']}\n"
    text += f"- Info: {row['Info']}\n"
    return text

def convert_to_text(input_file, output_file):
    """
    Converts the packet data in the CSV file to a text format for LLM training.
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Generate text for each packet
    df['packet_text'] = df.apply(packet_to_text, axis=1)

    # Save the text data to a file
    with open(output_file, 'w') as f:
        for text in df['packet_text']:
            f.write(text + '\n\n')
    
    print(f"Text data saved to {output_file}")


if __name__ == "__main__":
    input_file = "packets.csv"  # Replace with your CSV file name
    output_file = "packet_text.txt"
    
    convert_to_text(input_file, output_file)