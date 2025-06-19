# train_iot_transformer.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import joblib

# -----------------------------------------------------------------------------
# 1) Dataset & Preprocessing
# -----------------------------------------------------------------------------
class IoTDataset(Dataset):
    def __init__(self, csv_path, le_dict=None, scaler=None, fit=False):
        df = pd.read_csv(csv_path)
        cats = ['Source','Source port','Destination','Protocol','Destination port','Info']
        nums = ['Length']

        if fit:
            # fit LabelEncoders & scaler
            le_dict = {c: LabelEncoder().fit(df[c].astype(str)) for c in cats}
            scaler = StandardScaler().fit(df[nums])

        # encode categoricals
        cat_arrays = []
        for c in cats:
            arr = le_dict[c].transform(df[c].astype(str))
            cat_arrays.append(arr)
        self.X_cat = torch.LongTensor(np.stack(cat_arrays, axis=1))

        # scale numeric
        self.X_num = torch.FloatTensor(scaler.transform(df[nums]))

        self.le_dict = le_dict
        self.scaler = scaler

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return {'cat': self.X_cat[idx], 'num': self.X_num[idx]}

# -----------------------------------------------------------------------------
# 2) Model Definition
# -----------------------------------------------------------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self,
                 num_categories,
                 emb_dim=32,
                 num_numeric=1,
                 d_model=64,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=128):
        super().__init__()
        # embeddings for each categorical feature
        self.embs = nn.ModuleList([
            nn.Embedding(nc, emb_dim) for nc in num_categories
        ])
        # project numeric to same embedding dim
        self.num_proj = nn.Linear(num_numeric, emb_dim)

        # positional encoding
        self.pos_emb = nn.Parameter(torch.randn(7, d_model))

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # decoder: MLP from (d_model*7) back to original embedding vector
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 7, emb_dim * (len(num_categories) + 1)),
            nn.ReLU(),
            nn.Linear(emb_dim * (len(num_categories) + 1),
                      emb_dim * (len(num_categories) + 1))
        )

    def forward(self, X_cat, X_num):
        # X_cat: (B,6), X_num: (B,1)
        cat_emb = [emb(X_cat[:, i]) for i, emb in enumerate(self.embs)]
        num_emb = self.num_proj(X_num)  # (B, emb_dim)
        feats = torch.stack(cat_emb + [num_emb], dim=1)  # (B,7,emb_dim)

        B, S, E = feats.shape
        # if embedding dim != d_model, project
        d_model = self.pos_emb.shape[1]
        if E != d_model:
            proj = nn.Linear(E, d_model).to(feats.device)
            feats = proj(feats)

        # add positional encoding
        feats = feats + self.pos_emb.unsqueeze(0)

        # transformer expects (S, B, d_model)
        out = self.encoder(feats.permute(1, 0, 2))
        out = out.permute(1, 0, 2).contiguous().view(B, -1)

        return self.decoder(out)

# -----------------------------------------------------------------------------
# 3) Training Loop
# -----------------------------------------------------------------------------
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    desc=f"Epoch {epoch}/{epochs}")
        for i, batch in pbar:
            Xc = batch['cat'].to(device)
            Xn = batch['num'].to(device)

            # forward
            recon = model(Xc, Xn)

            # build target embedding
            with torch.no_grad():
                tgt_embs = [emb(Xc[:, j]) for j, emb in enumerate(model.embs)]
                tgt_embs.append(model.num_proj(Xn))
                target = torch.cat(tgt_embs, dim=1).view(recon.shape)

            loss = criterion(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                pbar.set_postfix(loss=running_loss / (i + 1))

        avg = running_loss / len(dataloader)
        print(f"→ Epoch {epoch} complete, avg loss: {avg:.6f}")

# -----------------------------------------------------------------------------
# 4) Main: Put it all together and save weights + preprocessors
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Train IoT anomaly detector autoencoder"
    )
    parser.add_argument('csv_path', help="Path to your training CSV")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # prepare dataset
    ds = IoTDataset(args.csv_path, fit=True)
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=2)

    # model
    num_categories = [
        len(ds.le_dict[c].classes_)
        for c in ['Source','Source port','Destination','Protocol','Destination port','Info']
    ]
    model = TransformerAutoencoder(num_categories)

    # optimizer + loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # train
    train(model, loader, optimizer, criterion, device, epochs=args.epochs)

    # save only model weights
    torch.save(model.state_dict(), 'model_weights.pth')
    print("→ Saved model weights to model_weights.pth")

    # save preprocessing objects
    joblib.dump(ds.le_dict, 'le_dict.pkl')
    joblib.dump(ds.scaler, 'scaler.pkl')
    print("→ Saved LabelEncoders to le_dict.pkl and scaler to scaler.pkl")
