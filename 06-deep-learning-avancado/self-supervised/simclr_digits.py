"""Aprendizado auto-supervisionado contrastivo (SimCLR) nos dígitos 8x8, sem usar rótulos no pré-treino.

Duas "visões" aumentadas da mesma imagem devem ter embeddings próximos; visões de imagens diferentes, distantes
(perda NT-Xent / InfoNCE). Avalia com uma sonda linear treinada com POUCOS rótulos.

pip install torch scikit-learn
"""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)


def aumenta(x):
    """Aumentações: deslocamento aleatório de ±1 pixel, ruído gaussiano e dropout de pixels."""
    img = x.view(-1, 8, 8)
    out = torch.empty_like(img)
    for i in range(len(img)):
        dx, dy = np.random.randint(-1, 2, 2)
        out[i] = torch.roll(img[i], shifts=(int(dx), int(dy)), dims=(0, 1))
    out = out + 0.1 * torch.randn_like(out)
    out = out * (torch.rand_like(out) > 0.1)
    return out.view(-1, 64)


def nt_xent(z1, z2, tau=0.5):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)                      # ignora a similaridade consigo mesmo
    alvo = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])  # o positivo de i é a outra visão
    return F.cross_entropy(sim, alvo)


encoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))
proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))  # cabeça de projeção (descartada depois)


def sonda(feats_tr, feats_te, n_rotulos):
    """Regressão logística com apenas n_rotulos exemplos rotulados sobre features congeladas (média de 5 sorteios)."""
    accs = []
    for seed in range(5):
        idx = np.random.RandomState(seed).choice(len(feats_tr), n_rotulos, replace=False)
        sc = StandardScaler().fit(feats_tr[idx])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(feats_tr[idx]), y_tr[idx])
        accs.append(clf.score(sc.transform(feats_te), y_te))
    return float(np.mean(accs))


def feats(m, x):
    with torch.no_grad():
        return m(x).numpy()


def relatorio(nome, ftr, fte):
    print(f"  {nome:26s}: 10 rótulos={sonda(ftr, fte, 10):.3f}   50 rótulos={sonda(ftr, fte, 50):.3f}")


print("Sonda linear (acurácia de teste com poucos rótulos):")
relatorio("pixels brutos", X_tr.numpy(), X_te.numpy())
relatorio("encoder aleatório", feats(encoder, X_tr), feats(encoder, X_te))

opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), 1e-3)
for ep in range(1, 101):
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), 128):
        b = X_tr[perm[i : i + 128]]
        loss = nt_xent(proj(encoder(aumenta(b))), proj(encoder(aumenta(b))))
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 25 == 0:
        print(f"época {ep:3d}  NT-Xent={loss.item():.3f}")
relatorio("encoder SimCLR (sem rótulos)", feats(encoder, X_tr), feats(encoder, X_te))
