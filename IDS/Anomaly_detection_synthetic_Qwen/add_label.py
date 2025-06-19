import pandas as pd

# 1. Load your single file
df = pd.read_csv('/Users/kushalprakash/Downloads/iot_intrusion_dataset/dos_synflood_1.csv')

# 2. Add the Label column (1 = attack)
df['Label'] = 1

# 3. Save it out
df.to_csv('/Users/kushalprakash/Desktop/UNI/Thesis/ThesisPrj/Anomaly_detection/dos_synflood_1.csv', index=False)

print(f"Saved labeled file to 'dos_synflood_1.csv' ({len(df)} rows)")
