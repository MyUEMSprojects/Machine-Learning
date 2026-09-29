"""MLP em PyTorch: classificação de dígitos (MNIST-like 8x8 do scikit-learn) com loop de treino completo.

pip install torch scikit-learn
"""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

modelo = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),  # logits; CrossEntropyLoss aplica softmax internamente
)
opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
perda = nn.CrossEntropyLoss()

for epoca in range(1, 61):
    modelo.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), 64):  # mini-batches
        b = perm[i : i + 64]
        opt.zero_grad()
        loss = perda(modelo(X_tr[b]), y_tr[b])
        loss.backward()  # backprop automático
        opt.step()
    if epoca % 10 == 0:
        modelo.eval()
        with torch.no_grad():
            acc = (modelo(X_te).argmax(1) == y_te).float().mean().item()
        print(f"época {epoca:3d}  loss={loss.item():.4f}  acc teste={acc:.3f}")
