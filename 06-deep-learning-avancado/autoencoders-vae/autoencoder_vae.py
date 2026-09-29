"""Autoencoder e VAE em PyTorch (dígitos 8x8). Compara o AE com o PCA e gera dígitos novos com o VAE.

pip install torch scikit-learn
"""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xt = torch.tensor(X)
LAT = 4  # dimensão latente


# ------------------------------- Autoencoder -------------------------------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, LAT))
        self.dec = nn.Sequential(nn.Linear(LAT, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


ae = AE()
opt = torch.optim.Adam(ae.parameters(), 3e-3)
for ep in range(400):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(perm), 128):
        b = Xt[perm[i : i + 128]]
        loss = F.mse_loss(ae(b), b)
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    mse_ae = F.mse_loss(ae(Xt), Xt).item()
pca = PCA(LAT).fit(X)
mse_pca = float(((pca.inverse_transform(pca.transform(X)) - X) ** 2).mean())
print(f"Reconstrução com {LAT} dims latentes: MSE  AE={mse_ae:.4f}  PCA={mse_pca:.4f}  (AE não linear captura mais)")


# ---------------------------------- VAE ------------------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.mu, self.logvar = nn.Linear(64, LAT), nn.Linear(64, LAT)
        self.dec = nn.Sequential(nn.Linear(LAT, 64), nn.ReLU(), nn.Linear(64, 64))  # logits

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # truque da reparametrização
        return self.dec(z), mu, logvar


def loss_vae(xh, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(xh, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)  # KL(q(z|x) || N(0, I))
    return rec + beta * kl, rec, kl


vae = VAE()
opt = torch.optim.Adam(vae.parameters(), 3e-3)
for ep in range(400):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(perm), 128):
        b = Xt[perm[i : i + 128]]
        xh, mu, lv = vae(b)
        loss, rec, kl = loss_vae(xh, b, mu, lv)
        opt.zero_grad(); loss.backward(); opt.step()
print(f"VAE: reconstrução (BCE)={rec.item():.2f}  KL={kl.item():.2f} nats")

# Geração: amostrar z ~ N(0, I) e decodificar. Um classificador treinado em dígitos reais
# diz quão "reconhecíveis" são as amostras.
clf = LogisticRegression(max_iter=2000).fit(X, y)
with torch.no_grad():
    amostras = torch.sigmoid(vae.dec(torch.randn(2000, LAT))).numpy()
conf_gen = clf.predict_proba(amostras).max(1).mean()
conf_ruido = clf.predict_proba(np.random.rand(2000, 64).astype(np.float32)).max(1).mean()
print(f"Confiança média do classificador — amostras do VAE: {conf_gen:.2f} | ruído uniforme: {conf_ruido:.2f}")
print("Dígitos gerados (contagem por classe prevista):", np.bincount(clf.predict(amostras), minlength=10))

# Interpolação no espaço latente entre dois dígitos reais
with torch.no_grad():
    a, b = vae.mu(vae.enc(Xt[y == 3][0])), vae.mu(vae.enc(Xt[y == 8][0]))
    caminho = torch.stack([a * (1 - t) + b * t for t in torch.linspace(0, 1, 6)])
    prev = clf.predict(torch.sigmoid(vae.dec(caminho)).numpy())
print("Interpolação 3 -> 8, classes previstas ao longo do caminho:", prev.tolist())
