#!/usr/bin/env python3
"""
evaluate_combined_iot_model.py

Usage:
  python evaluate_combined_iot_model.py \
    --benign benign1.csv [benign2.csv ...] \
    --attack attack1.csv [attack2.csv ...]
"""

import argparse
import pandas as pd
import numpy as np
import ipaddress
import joblib
import os
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

# ---- Helpers ----
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
    model  = load_model(model_path, compile=False)
    return scaler, ohe, tfidf, model

def preprocess(df: pd.DataFrame, scaler, ohe, tfidf) -> np.ndarray:
    # IP → int
    df['src_ip_int'] = df['Source'].apply(ip2int)
    df['dst_ip_int'] = df['Destination'].apply(ip2int)
    # Numeric features
    num_feats = ['src_ip_int','Source port','dst_ip_int','Destination port','Length']
    X_num = df[num_feats].fillna(0).values
    # Protocol one-hot, ignore unseen
    known = list(ohe.categories_[0])
    mapping = {p:i for i,p in enumerate(known)}
    X_proto = np.zeros((len(df), len(known)), dtype=float)
    for i,p in enumerate(df['Protocol'].astype(str)):
        j = mapping.get(p)
        if j is not None:
            X_proto[i,j] = 1.0
    # Info → TF-IDF
    X_info = tfidf.transform(df['Info'].astype(str)).toarray()
    # Combine + scale
    X = np.hstack([X_num, X_proto, X_info])
    return scaler.transform(X)

# ---- Main Evaluation ----
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IoT anomaly model on combined benign+attack CSVs"
    )
    parser.add_argument(
        '--benign', '-b',
        nargs='+',
        required=True,
        help="Path(s) to benign CSV(s)"
    )
    parser.add_argument(
        '--attack', '-a',
        nargs='+',
        required=True,
        help="Path(s) to attack CSV(s)"
    )
    args = parser.parse_args()

    # 1) Load & label
    dfs = []
    for p in args.benign:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p!r} not found")
        d = pd.read_csv(p)
        d['Label'] = 0
        dfs.append(d)
    for p in args.attack:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p!r} not found")
        d = pd.read_csv(p)
        d['Label'] = 1
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(df)} rows ({len(args.benign)} benign file(s), {len(args.attack)} attack file(s))")

    # 2) Load artifacts
    scaler, ohe, tfidf, model = load_artifacts()

    # 3) Preprocess + predict
    X     = preprocess(df, scaler, ohe, tfidf)
    recon = model.predict(X)
    mse   = np.mean((recon - X)**2, axis=1)

    # 4) Threshold on benign
    mask0 = (df['Label'] == 0).values
    thr   = mse[mask0].mean() + 2*mse[mask0].std()
    print(f"Using threshold = {thr:.6f}")

    y_true = df['Label'].values
    y_pred = (mse > thr).astype(int)

    # 5) Metrics
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm,
                       index=['True 0','True 1'],
                       columns=['Pred 0','Pred 1']), "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    fpr, tpr, _ = roc_curve(y_true, mse)
    roc_auc     = auc(fpr, tpr)
    prec, rec, _= precision_recall_curve(y_true, mse)
    ap          = average_precision_score(y_true, mse)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    # 6) Plot
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f'ROC (AUC={roc_auc:.4f})')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(rec, prec)
    plt.title(f'PR (AP={ap:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
