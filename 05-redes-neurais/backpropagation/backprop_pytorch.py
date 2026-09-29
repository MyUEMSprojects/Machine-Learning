"""Backpropagation: implementação manual em NumPy vs. autograd do PyTorch (devem coincidir).

pip install numpy torch
"""
import numpy as np
import torch

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
y = rng.integers(0, 2, size=(5, 1)).astype(float)
W1, b1 = rng.normal(size=(3, 4)) * 0.5, np.zeros(4)
W2, b2 = rng.normal(size=(4, 1)) * 0.5, np.zeros(1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---- forward manual ----
z1 = X @ W1 + b1
h = np.tanh(z1)
z2 = h @ W2 + b2
p = sigmoid(z2)
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ---- backward manual (regra da cadeia, camada por camada) ----
dz2 = (p - y) / len(X)          # dL/dz2 (sigmoide + entropia cruzada)
dW2 = h.T @ dz2
db2 = dz2.sum(0)
dh = dz2 @ W2.T
dz1 = dh * (1 - h**2)           # derivada do tanh
dW1 = X.T @ dz1
db1 = dz1.sum(0)

# ---- autograd do PyTorch ----
t = lambda a: torch.tensor(a, requires_grad=True, dtype=torch.float64)
tW1, tb1, tW2, tb2 = t(W1), t(b1), t(W2), t(b2)
tp = torch.sigmoid(torch.tanh(torch.tensor(X) @ tW1 + tb1) @ tW2 + tb2)
tl = torch.nn.functional.binary_cross_entropy(tp, torch.tensor(y))
tl.backward()

print(f"loss numpy={loss:.6f} torch={tl.item():.6f}")
for nome, manual, auto in [("W1", dW1, tW1), ("b1", db1, tb1), ("W2", dW2, tW2), ("b2", db2, tb2)]:
    print(f"{nome}: max|manual - autograd| = {np.abs(manual - auto.grad.numpy()).max():.2e}")
