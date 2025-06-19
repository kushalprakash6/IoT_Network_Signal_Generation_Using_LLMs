# evaluate_iot_model.py

# Requirements:
#   pip install pandas numpy ipaddress joblib tensorflow scikit-learn matplotlib

import pandas as pd
import numpy as np
import ipaddress
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# ----------------------------------------
# Helpers
# ----------------------------------------
def ip2int(ip: str) -> int:
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return 0

def load_artifacts(
    scaler_path='scaler.pkl',
    ohe_path='protocol_ohe.pkl',
    tfidf_path='tfidf.pkl',
    model_path='autoencoder_model.h5'
):
    scaler = joblib.load(scaler_path)
    ohe    = joblib.load(ohe_path)
    tfidf  = joblib.load(tfidf_path)
    # load with compile=False to avoid custom loss lookup errors
    model  = load_model(model_path, compile=False)
    return scaler, ohe, tfidf, model

def preprocess(df: pd.DataFrame, scaler, ohe, tfidf) -> np.ndarray:
    # Convert IPs
    df['src_ip_int'] = df['Source'].apply(ip2int)
    df['dst_ip_int'] = df['Destination'].apply(ip2int)

    # Numeric features
    num_feats = ['src_ip_int','Source port','dst_ip_int','Destination port','Length']
    X_num = df[num_feats].fillna(0).values

    # Manual one-hot for Protocol to ignore unseen categories
    known = list(ohe.categories_[0])
    mapping = {proto: idx for idx, proto in enumerate(known)}
    X_proto = np.zeros((len(df), len(known)), dtype=float)
    for i, p in enumerate(df['Protocol'].astype(str)):
        j = mapping.get(p)
        if j is not None:
            X_proto[i, j] = 1.0

    # Info text → TF–IDF (unknown words are ignored by design)
    X_info = tfidf.transform(df['Info'].astype(str)).toarray()

    # Combine + scale
    X = np.hstack([X_num, X_proto, X_info])
    return scaler.transform(X)

# ----------------------------------------
# Evaluation routine
# ----------------------------------------
def evaluate(test_csv_path: str, label_col: str = 'Label'):
    # 1. Load data
    df = pd.read_csv(test_csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Expected a '{label_col}' column in your CSV.")
    y_true = df[label_col].astype(int).values

    # 2. Load artifacts & preprocess
    scaler, ohe, tfidf, model = load_artifacts()
    X = preprocess(df, scaler, ohe, tfidf)

    # 3. Predict via reconstruction error
    recon = model.predict(X)
    mse   = np.mean((recon - X)**2, axis=1)

    # 4. Threshold based on benign portion of this test set
    benign_mask = (y_true == 0)
    thresh      = mse[benign_mask].mean() + 2 * mse[benign_mask].std()
    y_pred      = (mse > thresh).astype(int)

    # 5. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['True Benign','True Attack'],
        columns=['Pred Benign','Pred Attack']
    )
    print("\nConfusion Matrix:")
    print(cm_df, "\n")

    # 6. Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # 7. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, mse)
    roc_auc     = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

    # 8. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, mse)
    ap                  = average_precision_score(y_true, mse)
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

# ----------------------------------------
# Main: evaluate your single labeled file
# ----------------------------------------
if __name__ == "__main__":
    # Replace this with your single labeled CSV path
    test_file = "/Users/kushalprakash/Desktop/UNI/Thesis/ThesisPrj/Anomaly_detection/benign_IoT.csv"
    print(f"\n======== Evaluating on {test_file} ========\n")
    evaluate(test_file, label_col="Label")
