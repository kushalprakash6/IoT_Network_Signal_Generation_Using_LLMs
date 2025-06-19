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
print("Loading data...")
df = pd.read_csv('/Users/kushalprakash/Desktop/UNI/Thesis/ThesisPrj/Anomaly_detection_synthetic_Llama/Training_data/merged_output_LlamaAll_final.csv')

# 2. Preprocessing helper
def ip2int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0

# 3. Feature engineering
print("Preprocessing...")
df['src_ip_int'] = df['Source'].apply(ip2int)
df['dst_ip_int'] = df['Destination'].apply(ip2int)

numeric_features = ['src_ip_int', 'Source port', 'dst_ip_int', 'Destination port', 'Length']
num_data = df[numeric_features].fillna(0)

ohe = OneHotEncoder(sparse_output=False)
proto_encoded = ohe.fit_transform(df[['Protocol']])

tfidf = TfidfVectorizer(max_features=50)
info_encoded = tfidf.fit_transform(df['Info'].astype(str)).toarray()

# 4. Combine and scale
print("Combining and scaling features...")
X = np.hstack([num_data.values, proto_encoded, info_encoded])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/val split
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 6. Build autoencoder
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='linear')(encoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 7. Custom callback to stop at loss ≤ 0.01
class LossThreshold(callbacks.Callback):
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is not None and current_loss <= self.threshold:
            print(f"\nReached loss ≤ {self.threshold:.4f} (loss: {current_loss:.4f}) at epoch {epoch+1}. Stopping training.")
            self.model.stop_training = True

csv_logger = callbacks.CSVLogger('training_log.csv')

# 8. Training
print("Starting training...")
start_time = time.time()
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,                     # maximum of 100 epochs
    batch_size=256,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[LossThreshold(0.01), csv_logger],
    verbose=1
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# 9. Save artifacts
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

# 11. Anomaly detection function remains unchanged
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
