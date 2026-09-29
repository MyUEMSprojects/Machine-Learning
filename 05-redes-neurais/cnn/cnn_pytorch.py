"""CNN em PyTorch nos dígitos 8x8 (scikit-learn) + inspeção de formas e nº de parâmetros vs. MLP.

pip install torch scikit-learn
"""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)  # (N, canais, H, W)
y = torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 8x8 -> 4x4
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 4x4 -> 2x2
    nn.Flatten(), nn.Linear(32 * 2 * 2, 10),
)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
n_par = lambda m: sum(p.numel() for p in m.parameters())
print(f"parâmetros: CNN={n_par(cnn)}  MLP={n_par(mlp)}")

# formas ao longo da rede
z = X_tr[:1]
for camada in cnn:
    z = camada(z)
    print(f"{camada.__class__.__name__:12s} -> {tuple(z.shape)}")


def treina(m, epocas=30):
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(epocas):
        m.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(m(X_tr[b]), y_tr[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return (m(X_te).argmax(1) == y_te).float().mean().item()


print(f"acc teste  CNN={treina(cnn):.3f}  MLP={treina(mlp):.3f}")
