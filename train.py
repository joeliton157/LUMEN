import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse

from baseline_model import BaselineModel
from lumen_model import LUMENModel

# Configurações
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

# Argumento de escolha de modelo
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['baseline', 'lumen'], default='baseline')
args = parser.parse_args()

# Carregar os dados
df = pd.read_csv("data/data.csv")
X = df.drop("TARGET", axis=1).values
y = df["TARGET"].values

# Separar os domínios para LUMEN
X_gen = X[:, 0:10]
X_amb = X[:, 10:18]
X_rel = X[:, 18:24]

# Split
X_gen_train, X_gen_test, X_amb_train, X_amb_test, X_rel_train, X_rel_test, y_train, y_test = train_test_split(
    X_gen, X_amb, X_rel, y, test_size=0.2, random_state=42
)

# Tensores PyTorch
def to_tensor(*arrays):
    return [torch.tensor(arr, dtype=torch.float32) for arr in arrays]

X_gen_train, X_gen_test, X_amb_train, X_amb_test, X_rel_train, X_rel_test, y_train, y_test = to_tensor(
    X_gen_train, X_gen_test, X_amb_train, X_amb_test, X_rel_train, X_rel_test, y_train, y_test
)

# Dataset & Loader
if args.model == "baseline":
    X_train = torch.cat([X_gen_train, X_amb_train, X_rel_train], dim=1)
    X_test = torch.cat([X_gen_test, X_amb_test, X_rel_test], dim=1)
    train_ds = TensorDataset(X_train, y_train.unsqueeze(1))
    test_ds = TensorDataset(X_test, y_test.unsqueeze(1))
else:
    train_ds = TensorDataset(X_gen_train, X_amb_train, X_rel_train, y_train.unsqueeze(1))
    test_ds = TensorDataset(X_gen_test, X_amb_test, X_rel_test, y_test.unsqueeze(1))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Inicializar o modelo
if args.model == "baseline":
    model = BaselineModel(input_dim=24)
else:
    model = LUMENModel(input_dims=(10, 8, 6))

# Treinamento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        if args.model == "baseline":
            inputs, targets = batch
            outputs = model(inputs.to(device))
        else:
            xg, xa, xr, targets = batch
            outputs = model(xg.to(device), xa.to(device), xr.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

# Avaliação
model.eval()
y_preds = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        if args.model == "baseline":
            inputs, targets = batch
            outputs = model(inputs.to(device))
        else:
            xg, xa, xr, targets = batch
            outputs = model(xg.to(device), xa.to(device), xr.to(device))
        y_preds.extend(outputs.cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# Métricas
y_preds_binary = [1 if p > 0.5 else 0 for p in y_preds]
acc = accuracy_score(y_true, y_preds_binary)
auc = roc_auc_score(y_true, y_preds)
print(f"\n✅ Modelo: {args.model.upper()}")
print(f"🎯 Acurácia: {acc:.4f}")
print(f"📈 AUC: {auc:.4f}")

torch.save(model.state_dict(), "models/lumen.pt")
