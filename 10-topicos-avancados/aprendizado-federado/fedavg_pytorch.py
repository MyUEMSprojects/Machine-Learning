"""Aprendizado Federado em PyTorch: FedAvg em dígitos com clientes NÃO-IID (cada cliente vê poucas classes), e
Privacidade Diferencial simples (recorte + ruído gaussiano nas atualizações).

pip install torch scikit-learn
"""
import copy

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

K = 10
# partição não-IID: cada cliente recebe amostras de apenas 2 classes ("shards")
ordem = torch.argsort(y_tr)
shards = np.array_split(ordem.numpy(), 2 * K)
perm = rng.permutation(2 * K)
clientes = [np.concatenate([shards[perm[2 * k]], shards[perm[2 * k + 1]]]) for k in range(K)]
print("classes por cliente:", [sorted(set(y_tr[c].tolist())) for c in clientes][:4], "...")


def modelo():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def acc(m):
    with torch.no_grad():
        return (m(X_te).argmax(1) == y_te).float().mean().item()


def treino_local(global_m, idx, epocas, lr=0.05):
    m = copy.deepcopy(global_m)
    opt = torch.optim.SGD(m.parameters(), lr)
    xi, yi = X_tr[idx], y_tr[idx]
    for _ in range(epocas):
        p = torch.randperm(len(idx))
        for i in range(0, len(p), 32):
            b = p[i : i + 32]
            opt.zero_grad(); nn.functional.cross_entropy(m(xi[b]), yi[b]).backward(); opt.step()
    return m


def fedavg(rodadas=30, epocas=3, dp_clip=None, dp_ruido=0.0):
    g = modelo()
    for r in range(1, rodadas + 1):
        deltas, pesos = [], []
        for idx in clientes:
            local = treino_local(g, idx, epocas)
            d = [pl.data - pg.data for pl, pg in zip(local.parameters(), g.parameters())]  # atualização do cliente
            if dp_clip:  # DP: limita a norma da contribuição de cada cliente e (depois) soma ruído
                norma = torch.sqrt(sum((x ** 2).sum() for x in d))
                d = [x * min(1.0, dp_clip / (norma.item() + 1e-12)) for x in d]
            deltas.append(d); pesos.append(len(idx))
        w = np.array(pesos) / sum(pesos)
        with torch.no_grad():
            for j, p in enumerate(g.parameters()):
                agg = sum(w[k] * deltas[k][j] for k in range(K))
                if dp_ruido:
                    agg = agg + torch.randn_like(agg) * dp_ruido * (dp_clip or 1.0) / K
                p += agg
    return acc(g)


central = modelo()
opt = torch.optim.SGD(central.parameters(), 0.05)
for _ in range(60):
    p = torch.randperm(len(X_tr))
    for i in range(0, len(p), 32):
        b = p[i : i + 32]
        opt.zero_grad(); nn.functional.cross_entropy(central(X_tr[b]), y_tr[b]).backward(); opt.step()
print(f"centralizado (todos os dados)         : acc = {acc(central):.3f}")
print(f"FedAvg (30 rodadas, 3 épocas locais)  : acc = {fedavg():.3f}")
print(f"FedAvg + recorte de norma (C=1)       : acc = {fedavg(dp_clip=1.0):.3f}")
print(f"FedAvg + DP (C=1, ruído σ=1·C)        : acc = {fedavg(dp_clip=1.0, dp_ruido=1.0):.3f}  <- ruído para privacidade custa utilidade")
