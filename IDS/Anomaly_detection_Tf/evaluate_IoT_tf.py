import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ----------------------------
# 1) Dataset class (reuse yours)
# ----------------------------
class IoTDataset(Dataset):
    def __init__(self, df, le_dict, scaler):
        cats = ['Source','Source port','Destination','Protocol','Destination port','Info']
        nums = ['Length']
        self.le_dict = le_dict
        self.scaler = scaler

        # encode categoricals
        cat_arrays = []
        for c in cats:
            classes = le_dict[c].classes_
            vals = df[c].astype(str).values
            # map each to its index
            idx = np.array([np.where(classes == v)[0][0] for v in vals])
            cat_arrays.append(idx)
        self.X_cat = torch.LongTensor(np.stack(cat_arrays, axis=1))
        # scale numeric
        self.X_num = torch.FloatTensor(scaler.transform(df[nums]))

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, i):
        return {'cat': self.X_cat[i], 'num': self.X_num[i]}

# ----------------------------
# 2) Model definition (must match training)
# ----------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, num_categories, emb_dim=32, num_numeric=1,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=128):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(nc, emb_dim) for nc in num_categories])
        self.num_proj = nn.Linear(num_numeric, emb_dim)
        self.input_dim = emb_dim * (len(num_categories) + 1)
        self.pos_emb = nn.Parameter(torch.randn(7, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model*7, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        )

    def forward(self, X_cat, X_num):
        cat_emb = [emb(X_cat[:,i]) for i,emb in enumerate(self.embs)]
        num_emb = self.num_proj(X_num)
        feats = torch.stack(cat_emb + [num_emb], dim=1)  # (B,7,emb_dim)
        B, S, E = feats.shape
        if E != self.pos_emb.shape[1]:
            proj = nn.Linear(E, self.pos_emb.shape[1]).to(feats.device)
            feats = proj(feats)
        feats = feats + self.pos_emb.unsqueeze(0)
        out = self.encoder(feats.permute(1,0,2))
        out = out.permute(1,0,2).contiguous().view(B, -1)
        return self.decoder(out)

# ----------------------------
# 3) Helpers: load checkpoint
# ----------------------------
def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    # rebuild model
    num_cats = [len(ckpt['le_dict'][c].classes_) for c in ckpt['le_dict']]
    model = TransformerAutoencoder(num_cats)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt['le_dict'], ckpt['scaler']

# ----------------------------
# 4) Evaluate one file
# ----------------------------
def evaluate_file(model, le_dict, scaler, file_path, device, batch_size):
    # load data
    df = pd.read_csv(file_path)
    # binary labels: Normal=0, Attack=1
    y_true = (df['Label'] != 'Normal').astype(int).values

    ds = IoTDataset(df, le_dict, scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # inference + error collection
    errors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inferring {file_path.name}", leave=False):
            Xc = batch['cat'].to(device)
            Xn = batch['num'].to(device)
            recon = model(Xc, Xn)
            # build target emb
            tgt = torch.cat(
                [emb(Xc[:,j]).detach() for j,emb in enumerate(model.embs)]
                + [model.num_proj(Xn).detach()],
                dim=1
            ).view(recon.shape)
            batch_err = ((recon - tgt)**2).mean(dim=1).cpu().numpy()
            errors.append(batch_err)
    errors = np.concatenate(errors)

    # compute metrics
    roc_auc = roc_auc_score(y_true, errors)
    fpr, tpr, roc_th = roc_curve(y_true, errors)
    avg_prec = average_precision_score(y_true, errors)
    prec, rec, pr_th = precision_recall_curve(y_true, errors)
    # choose threshold via Youden's J
    j_scores = tpr - fpr
    best_thresh = roc_th[np.argmax(j_scores)]
    y_pred = (errors >= best_thresh).astype(int)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'roc_auc': roc_auc, 'avg_prec': avg_prec,
        'fpr': fpr, 'tpr': tpr, 'prec': prec, 'rec': rec,
        'best_thresh': best_thresh, 'report': report, 'cm': cm
    }

# ----------------------------
# 5) Plotting
# ----------------------------
def plot_results(metrics, title_suffix):
    # ROC
    plt.figure()
    plt.plot(metrics['fpr'], metrics['tpr'])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {title_suffix}")
    plt.grid(True)
    # PR
    plt.figure()
    plt.plot(metrics['rec'], metrics['prec'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {title_suffix}")
    plt.grid(True)
    # Confusion Matrix
    cm = metrics['cm']
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(f"Confusion Matrix: {title_suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.show()

# ----------------------------
# 6) Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IoT Anomaly Detector on one or more attack CSVs"
    )
    parser.add_argument(
        'input_files', nargs='+',
        help="Paths to one or more attack-labeled CSV files"
    )
    parser.add_argument(
        '--checkpoint', default='iot_transformer_ae.pth',
        help="Your trained model checkpoint (default: %(default)s)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help="Batch size for inference (default: %(default)s)"
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, le_dict, scaler = load_checkpoint(args.checkpoint, device)

    for fp in args.input_files:
        path = Path(fp)
        print(f"\n=== Evaluating {path.name} ===")
        met = evaluate_file(
            model, le_dict, scaler,
            path, device, args.batch_size
        )
        print(f"ROC AUC:      {met['roc_auc']:.4f}")
        print(f"Avg Precision:{met['avg_prec']:.4f}")
        print(f"Best Threshold (Youden): {met['best_thresh']:.6f}\n")
        print("Classification Report:")
        print(met['report'])
        plot_results(met, path.name)

if __name__ == '__main__':
    main()
