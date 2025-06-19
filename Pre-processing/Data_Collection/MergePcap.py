import pandas as pd
from scapy.all import rdpcap

def extract_packets(pcap_file):
    # Read the .pcap file
    packets = rdpcap(pcap_file)
    
    # Extract relevant fields
    data = []
    for packet in packets:
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
            dst_ip = packet['IP'].dst
            protocol = packet['IP'].proto
            payload = str(packet['IP'].payload) if packet.haslayer('Raw') else ""
            timestamp = packet.time
            
            data.append({
                'frame_number': len(data) + 1,
                'source_ip': src_ip,
                'destination_ip': dst_ip,
                'protocol': protocol,
                'payload': payload,
                'timestamp': timestamp
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def save_dataset(df, output_file):
    # Save the dataset to a CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    pcap_file = "/Users/kushalprakash/Desktop/UNI/Thesis/TrainingData/Filtered/FinalMerged.pcap"
    output_file = "packets.csv"
    
    df = extract_packets(pcap_file)
    save_dataset(df, output_file)
    print(f"Dataset saved to {output_file}")