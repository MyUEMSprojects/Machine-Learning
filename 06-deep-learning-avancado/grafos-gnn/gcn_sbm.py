"""Graph Convolutional Network (GCN) do zero em PyTorch para classificação semi-supervisionada de nós.

Grafo sintético (Stochastic Block Model, 3 comunidades) com features de nó fracamente informativas e só
5 rótulos por classe. A GCN usa a estrutura do grafo; uma MLP só as features.

pip install torch numpy
"""
import numpy as np
import torch
from torch import nn

rng = np.random.default_rng(0)
torch.manual_seed(0)
C, N_POR, DIM = 3, 60, 16
N = C * N_POR
y = np.repeat(np.arange(C), N_POR)

# Grafo: arestas dentro da comunidade com p=0.12, entre comunidades com p=0.01
P = np.where(y[:, None] == y[None, :], 0.12, 0.01)
A = (rng.random((N, N)) < P).astype(np.float32)
A = np.triu(A, 1)
A = A + A.T

# Features: média por classe fraca + muito ruído
medias = rng.normal(size=(C, DIM))
X = (0.35 * medias[y] + rng.normal(size=(N, DIM))).astype(np.float32)

# Normalização simétrica com self-loops: A_hat = D^-1/2 (A + I) D^-1/2
At = A + np.eye(N, dtype=np.float32)
d = At.sum(1)
A_hat = torch.tensor(At / np.sqrt(d[:, None] * d[None, :]), dtype=torch.float32)

X, yt = torch.tensor(X), torch.tensor(y)
treino = torch.tensor(np.concatenate([np.where(y == c)[0][:5] for c in range(C)]))  # 5 rótulos/classe
mask_te = torch.ones(N, dtype=torch.bool)
mask_te[treino] = False


class GCN(nn.Module):
    """H' = ReLU(A_hat H W): cada nó agrega (média ponderada) as features dos vizinhos e transforma."""

    def __init__(self, i, h, o):
        super().__init__()
        self.l1, self.l2 = nn.Linear(i, h), nn.Linear(h, o)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        h = torch.relu(A_hat @ self.l1(self.drop(x)))
        return A_hat @ self.l2(self.drop(h))  # logits por nó


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.5), nn.Linear(i, h), nn.ReLU(), nn.Dropout(0.5), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def roda(modelo, epocas=200):
    opt = torch.optim.Adam(modelo.parameters(), 1e-2, weight_decay=5e-4)
    for _ in range(epocas):
        modelo.train()
        opt.zero_grad()
        nn.functional.cross_entropy(modelo(X)[treino], yt[treino]).backward()
        opt.step()
    modelo.eval()
    with torch.no_grad():
        return (modelo(X).argmax(1)[mask_te] == yt[mask_te]).float().mean().item()


print(f"nós={N}, arestas={int(A.sum() / 2)}, rótulos usados={len(treino)}")
print(f"MLP (só features)      : acc nos nós não rotulados = {np.mean([roda(MLP(DIM, 16, C)) for _ in range(5)]):.3f}")
print(f"GCN (features + grafo) : acc nos nós não rotulados = {np.mean([roda(GCN(DIM, 16, C)) for _ in range(5)]):.3f}")
