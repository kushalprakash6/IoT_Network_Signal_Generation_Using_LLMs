# Requirements:
# pip install pandas numpy scikit-learn tensorflow joblib
import pandas as pd
import numpy as np
import ipaddress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time
import joblib

# 1. Load your data
# Replace 'network_traffic.csv' with your dataset path
print("Loading data...")
df = pd.read_csv('/Users/kushalprakash/Desktop/UNI/Thesis/ThesisPrj/cleaned_file.csv')

# 2. Preprocessing helper functions
print("Preprocessing...")
def ip2int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0

# 3. Feature engineering
# Convert IPs to integers
df['src_ip_int'] = df['Source'].apply(ip2int)
df['dst_ip_int'] = df['Destination'].apply(ip2int)

# Numeric fields
numeric_features = ['src_ip_int', 'Source port', 'dst_ip_int', 'Destination port', 'Length']
num_data = df[numeric_features].fillna(0)

# Protocol one-hot encoding
ohe = OneHotEncoder(sparse_output=False)
proto_encoded = ohe.fit_transform(df[['Protocol']])

# 'Info' text vectorization
tfidf = TfidfVectorizer(max_features=50)
info_encoded = tfidf.fit_transform(df['Info'].astype(str)).toarray()

# 4. Combine and scale features
print("Combining and scaling features...")
X = np.hstack([num_data.values, proto_encoded, info_encoded])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/validation split
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 6. Build autoencoder model
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='linear')(encoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 7. Callbacks for progress logging
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
csv_logger = callbacks.CSVLogger('training_log.csv')

# 8. Training with progress
print("Starting training...")
start_time = time.time()
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stop, csv_logger],
    verbose=1  # prints per-epoch progress
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# 9. Save preprocessing objects and model
print("Saving artifacts...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ohe, 'protocol_ohe.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
autoencoder.save('autoencoder_model.h5')

# 10. Determine anomaly threshold
print("Computing threshold...")
reconstructions = autoencoder.predict(X_val)
mse = np.mean(np.square(reconstructions - X_val), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
print(f"Anomaly threshold set to: {threshold:.4f}")

# 11. Function to detect anomalies on new data
def detect_anomalies(df_new):
    df_new['src_ip_int'] = df_new['Source'].apply(ip2int)
    df_new['dst_ip_int'] = df_new['Destination'].apply(ip2int)
    num_new = df_new[numeric_features].fillna(0).values
    proto_new = ohe.transform(df_new[['Protocol']])
    info_new = tfidf.transform(df_new['Info'].astype(str)).toarray()
    X_new = np.hstack([num_new, proto_new, info_new])
    X_new_scaled = scaler.transform(X_new)
    recon_new = autoencoder.predict(X_new_scaled)
    mse_new = np.mean(np.square(recon_new - X_new_scaled), axis=1)
    df_new['reconstruction_error'] = mse_new
    df_new['anomaly'] = mse_new > threshold
    return df_new

# # 12. Example usage for testing
# df_test = pd.read_csv('new_network_traffic.csv')
# results = detect_anomalies(df_test)
# print(results[['Source', 'Destination', 'anomaly', 'reconstruction_error']].head())
