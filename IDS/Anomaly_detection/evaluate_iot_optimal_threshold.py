#!/usr/bin/env python3
"""
evaluate_iot_optimal_threshold.py

Combine benign/attack CSVs into one labeled test set,
then find the threshold that maximizes Youden's J (TPR - FPR) to balance false positives and false negatives.

Usage:
  python evaluate_iot_optimal_threshold.py \
    --benign benign1.csv [benign2.csv ...] \
    --attack attack1.csv [attack2.csv ...]
"""

import argparse
import os
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

# --------- Helpers ---------
def ip2int(ip: str) -> int:
    """Convert IP string to int, fallback to 0."""
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
    """Load preprocessing objects and trained autoencoder."""
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    tfidf = joblib.load(tfidf_path)
    model = load_model(model_path, compile=False)
    return scaler, ohe, tfidf, model


def preprocess(df: pd.DataFrame, scaler, ohe, tfidf) -> np.ndarray:
    """Convert raw DataFrame to scaled feature matrix."""
    df['src_ip_int'] = df['Source'].apply(ip2int)
    df['dst_ip_int'] = df['Destination'].apply(ip2int)
    num_feats = ['src_ip_int','Source port','dst_ip_int','Destination port','Length']
    X_num = df[num_feats].fillna(0).values
    # protocol one-hot
    known = list(ohe.categories_[0])
    mapping = {p:i for i,p in enumerate(known)}
    X_proto = np.zeros((len(df), len(known)), dtype=float)
    for i,p in enumerate(df['Protocol'].astype(str)):
        idx = mapping.get(p)
        if idx is not None:
            X_proto[i,idx] = 1.0
    X_info = tfidf.transform(df['Info'].astype(str)).toarray()
    X = np.hstack([X_num, X_proto, X_info])
    return scaler.transform(X)


def find_youden_threshold(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute threshold maximizing Youden's J = TPR - FPR.
    Returns threshold, J, FPR, TPR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    J = tpr - fpr
    best_idx = np.nanargmax(J)
    return thresholds[best_idx], J[best_idx], fpr[best_idx], tpr[best_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model using Youden's J to balance FP and FN"
    )
    parser.add_argument('--benign', '-b', nargs='+', required=True, help="Benign-only CSVs")
    parser.add_argument('--attack', '-a', nargs='+', required=True, help="Attack-only CSVs")
    args = parser.parse_args()

    # Load and label
    dfs = []
    for p in args.benign:
        if not os.path.isfile(p): raise FileNotFoundError(p)
        dfb = pd.read_csv(p); dfb['Label'] = 0; dfs.append(dfb)
    for p in args.attack:
        if not os.path.isfile(p): raise FileNotFoundError(p)
        dfa = pd.read_csv(p); dfa['Label'] = 1; dfs.append(dfa)
    data = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(data)} rows: {sum(data['Label']==0)} benign, {sum(data['Label']==1)} attack.")

    # Load artifacts
    scaler, ohe, tfidf, model = load_artifacts()

    # Preprocess & compute reconstruction error
    X = preprocess(data, scaler, ohe, tfidf)
    recon = model.predict(X)
    scores = np.mean((recon - X)**2, axis=1)
    y_true = data['Label'].values

    # Find Youden's J-optimal threshold
    thr, J, fpr_j, tpr_j = find_youden_threshold(y_true, scores)
    print(f"Youden's J-optimal threshold = {thr:.6f} (J={J:.4f}, FPR={fpr_j:.4f}, TPR={tpr_j:.4f})")
    y_pred = (scores > thr).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=['True 0','True 1'], columns=['Pred 0','Pred 1']), "\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # ROC & PR metrics
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    print(f"ROC AUC: {roc_auc:.4f}, Average Precision (AP): {ap:.4f}\n")

    # Plot ROC with Youden point
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.scatter(fpr_j, tpr_j, c='red', s=50, label='Youden-opt')
    plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True)
    plt.show()

    # Plot PR curve
    idx_f1 = np.nanargmax(2*(precision*recall)/(precision+recall+1e-8))
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.scatter(recall[idx_f1], precision[idx_f1], c='red', s=50, label='F1-opt')
    plt.title('Precision–Recall Curve'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
