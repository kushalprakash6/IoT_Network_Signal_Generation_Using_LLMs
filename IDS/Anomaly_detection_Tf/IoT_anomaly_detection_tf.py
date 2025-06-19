import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm

# 1. Dataset & Preprocessing
class IoTDataset(Dataset):
    def __init__(self, csv_path, le_dict=None, scaler=None, fit=True):
        df = pd.read_csv(csv_path)
        
        # split out features
        cats = ['Source','Source port','Destination','Protocol','Destination port','Info']
        nums = ['Length']
        
        if fit:
            # fit label‐encoders
            le_dict = {c: LabelEncoder().fit(df[c].astype(str)) for c in cats}
            scaler = StandardScaler().fit(df[nums])
        
        # encode categoricals
        cat_arrays = []
        for c in cats:
            arr = le_dict[c].transform(df[c].astype(str))
            cat_arrays.append(arr)
        cat_arrays = np.stack(cat_arrays, axis=1)
        # scale numeric
        num_array = scaler.transform(df[nums])
        
        self.X_cat = torch.LongTensor(cat_arrays)       # (N,6)
        self.X_num = torch.FloatTensor(num_array)      # (N,1)
        self.le_dict, self.scaler = le_dict, scaler

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, i):
        return {'cat': self.X_cat[i], 'num': self.X_num[i]}

# 2. Model
class TransformerAutoencoder(nn.Module):
    def __init__(self, num_categories, emb_dim=32, num_numeric=1,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=128):
        super().__init__()
        # embeddings for each categorical feature
        self.embs = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim) 
            for n_cat in num_categories
        ])
        # project numeric to same dim
        self.num_proj = nn.Linear(num_numeric, emb_dim)
        
        self.input_dim = emb_dim*(len(num_categories)+1)
        # positional encoding (optional; here we treat features as "sequence" of length 7)
        self.pos_emb = nn.Parameter(torch.randn(7, d_model))
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # decoder: simple MLP to reconstruct original inputs
        self.decoder = nn.Sequential(
            nn.Linear(d_model*7, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        )
    
    def forward(self, X_cat, X_num):
        # X_cat: (B,6), X_num: (B,1)
        cat_emb = [emb(X_cat[:,i]) for i,emb in enumerate(self.embs)]
        num_emb = self.num_proj(X_num)            # (B,emb_dim)
        feats = torch.stack(cat_emb + [num_emb], dim=1)  # (B,7,emb_dim)
        # map to d_model
        B, S, E = feats.shape
        if E != self.pos_emb.shape[1]:
            # project to d_model
            feats = nn.Linear(E, self.pos_emb.shape[1]).to(feats.device)(feats)
        feats = feats + self.pos_emb.unsqueeze(0)  # add pos encoding
        # transformer expects (S,B,E)
        out = self.encoder(feats.permute(1,0,2))
        out = out.permute(1,0,2).contiguous().view(B, -1)
        recon = self.decoder(out)
        return recon

# 3. Training loop with progress stages
def train(
    model, 
    dataloader, 
    epochs=100, 
    lr=1e-3, 
    device='mps' if torch.backends.mps.is_available() else 'cpu'
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")
        for i, batch in pbar:
            Xc = batch['cat'].to(device)
            Xn = batch['num'].to(device)
            
            # forward
            recon = model(Xc, Xn)
            # build target vector
            target = torch.cat(
                [emb(Xc[:,j]).detach() for j,emb in enumerate(model.embs)]
                + [model.num_proj(Xn).detach()],
                dim=1
            ).view(recon.shape)
            
            loss = criterion(recon, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                pbar.set_postfix(loss=running_loss/(i+1))
        
        avg = running_loss / len(dataloader)
        print(f"→ Stage [Epoch {epoch}] complete, avg loss: {avg:.6f}")

# 4. Usage
if __name__ == '__main__':
    # parameters
    CSV_PATH = '/Users/kushalprakash/Desktop/UNI/Thesis/ThesisPrj/cleaned_file.csv'
    BATCH_SIZE = 64
    EPOCHS = 100

    # prepare data
    ds = IoTDataset(CSV_PATH, fit=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # instantiate model
    num_categories = [len(ds.le_dict[c].classes_) for c in ['Source','Source port','Destination','Protocol','Destination port','Info']]
    model = TransformerAutoencoder(num_categories)

    # train
    train(model, loader, epochs=EPOCHS, device='mps' if torch.backends.mps.is_available() else 'cpu')

    # save
    torch.save({
        'model_state': model.state_dict(),
        'le_dict': ds.le_dict,
        'scaler': ds.scaler
    }, 'iot_transformer_ae.pth')
