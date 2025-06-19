#!/usr/bin/env python3
"""
evaluate_combined_iot_transformer_model.py

Usage:
  python evaluate_combined_iot_transformer_model.py \
    --benign benign1.csv [benign2.csv ...] \
    --attack attack1.csv [attack2.csv ...]
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# -----------------------------------------------------------------------------
# 1) Dataset: same seven features, unseen categories→0
# -----------------------------------------------------------------------------
class IoTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, le_dict, scaler):
        self.cats = ['Source','Source port','Destination','Protocol','Destination port','Info']
        self.nums = ['Length']
        # verify
        missing = [c for c in self.cats + self.nums if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # categorical encoding
        cat_arrays = []
        for c in self.cats:
            vals = df[c].astype(str).values
            classes = le_dict[c].classes_
            mapping = {v:i for i,v in enumerate(classes)}
            unseen = set(vals) - set(classes)
            if unseen:
                warnings.warn(f"Column '{c}' has {len(unseen)} unseen values → mapping to 0")
            idxs = [mapping.get(v, 0) for v in vals]
            cat_arrays.append(np.array(idxs, dtype=np.int64))
        self.X_cat = torch.LongTensor(np.stack(cat_arrays, axis=1))
        # numeric scaling
        num_array = scaler.transform(df[self.nums])
        self.X_num = torch.FloatTensor(num_array)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx]

# -----------------------------------------------------------------------------
# 2) Transformer autoencoder (must match your train script)
# -----------------------------------------------------------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, num_categories, emb_dim=32,
                 num_numeric=1, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=128):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(nc, emb_dim)
                                   for nc in num_categories])
        self.num_proj = nn.Linear(num_numeric, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(7, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(enc_layer,
                                             num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 7, emb_dim * (len(num_categories) + 1)),
            nn.ReLU(),
            nn.Linear(emb_dim * (len(num_categories) + 1),
                      emb_dim * (len(num_categories) + 1))
        )

    def forward(self, X_cat, X_num):
        cat_emb = [emb(X_cat[:, i]) for i, emb in enumerate(self.embs)]
        num_emb = self.num_proj(X_num)
        feats = torch.stack(cat_emb + [num_emb], dim=1)   # (B,7,emb_dim)
        B, S, E = feats.shape
        d_model = self.pos_emb.shape[1]
        if E != d_model:
            proj = nn.Linear(E, d_model).to(feats.device)
            feats = proj(feats)
        feats = feats + self.pos_emb.unsqueeze(0)
        out = self.encoder(feats.permute(1, 0, 2))
        out = out.permute(1, 0, 2).reshape(B, -1)
        return self.decoder(out)

# -----------------------------------------------------------------------------
# 3) Load artifacts
# -----------------------------------------------------------------------------
def load_artifacts(weights_path='model_weights.pth',
                   le_path='le_dict.pkl',
                   scaler_path='scaler.pkl'):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(weights_path)
    if not os.path.isfile(le_path):
        raise FileNotFoundError(le_path)
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(scaler_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    le_dict = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    num_categories = [len(le_dict[c].classes_) for c in le_dict]
    model = TransformerAutoencoder(num_categories)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, le_dict, scaler, device

# -----------------------------------------------------------------------------
# 4) Main flow
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer IoT Anomaly Model on combined benign+attack CSVs"
    )
    parser.add_argument('-b','--benign', nargs='+', required=True,
                        help="One or more benign CSV files")
    parser.add_argument('-a','--attack', nargs='+', required=True,
                        help="One or more attack CSV files")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Inference batch size")
    parser.add_argument('--weights', default='model_weights.pth')
    parser.add_argument('--le', default='le_dict.pkl')
    parser.add_argument('--scaler', default='scaler.pkl')
    args = parser.parse_args()

    # 1) Load & label
    dfs = []
    for p in args.benign:
        dfb = pd.read_csv(p)
        dfb['Label'] = 0
        dfs.append(dfb)
    for p in args.attack:
        dfa = pd.read_csv(p)
        dfa['Label'] = 1
        dfs.append(dfa)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(df)} rows "
          f"({len(args.benign)} benign, {len(args.attack)} attack)")

    # 2) Load model + preprocessors
    model, le_dict, scaler, device = load_artifacts(
        args.weights, args.le, args.scaler
    )

    # 3) Build dataset & dataloader
    ds = IoTDataset(df, le_dict, scaler)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # 4) Run inference & compute MSE per sample
    errors = []
    with torch.no_grad():
        for cat, num in tqdm(loader, desc="Inferring"):
            cat = cat.to(device)
            num = num.to(device)
            recon = model(cat, num)
            # build target embedding
            tgt_embs = [emb(cat[:, j]).detach()
                        for j, emb in enumerate(model.embs)]
            tgt_embs.append(model.num_proj(num).detach())
            target = torch.cat(tgt_embs, dim=1).reshape_as(recon)
            mse = ((recon - target)**2).mean(dim=1).cpu().numpy()
            errors.append(mse)
    errors = np.concatenate(errors)
    y_true = df['Label'].values

    # 5) Threshold on benign only
    benign_err = errors[y_true == 0]
    thr = benign_err.mean() + 2 * benign_err.std()
    print(f"\nThreshold = mean(benign)+2·std = {thr:.6f}")

    # 6) Predictions & metrics
    y_pred = (errors > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm,
                       index=['True 0','True 1'],
                       columns=['Pred 0','Pred 1']), "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # 7) ROC & PR
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, errors)
    ap = average_precision_score(y_true, errors)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")

    # 8) Plot
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
