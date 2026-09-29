"""LoRA (Low-Rank Adaptation) implementado do zero em PyTorch e comparado com fine-tuning completo.

Cenário: uma MLP treinada em dígitos 8x8 é adaptada a um domínio novo (imagens espelhadas horizontalmente) com
poucos exemplos. Compara: sem adaptação, só a última camada, LoRA (ranks 1/4/8) e fine-tuning completo,
mostrando a acurácia e a fração de parâmetros treináveis.

pip install torch scikit-learn
"""
import copy

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)


def espelha(x):  # novo domínio: imagens espelhadas horizontalmente
    return x.view(-1, 8, 8).flip(2).reshape(-1, 64)


class LoRALinear(nn.Module):
    """y = x W^T + b + (alpha/r) * x A^T B^T ; W e b congelados; só A (r x in) e B (out x r) são treinados."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B=0 => começa idêntico ao modelo original

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merge(self):  # depois do treino, funde o delta em W: custo zero de inferência
        with torch.no_grad():
            self.base.weight += self.scale * self.B @ self.A
        return self.base


def mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def treina(m, X, y, epocas, lr, params=None):
    opt = torch.optim.Adam(params if params is not None else [p for p in m.parameters() if p.requires_grad], lr)
    for _ in range(epocas):
        opt.zero_grad()
        nn.functional.cross_entropy(m(X), y).backward()
        opt.step()


def acc(m, X, y):
    with torch.no_grad():
        return (m(X).argmax(1) == y).float().mean().item()


def n_treinaveis(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# 1) "pré-treino" no domínio original
base = mlp()
treina(base, X_tr, y_tr, 400, 3e-3, params=base.parameters())
total = sum(p.numel() for p in base.parameters())
print(f"modelo base: {total:,} parâmetros | acc domínio original = {acc(base, X_te, y_te):.3f} | acc domínio espelhado = {acc(base, espelha(X_te), y_te):.3f}")

# 2) adaptação com poucos exemplos do domínio novo
idx = torch.randperm(len(X_tr))[:150]
Xa, ya = espelha(X_tr[idx]), y_tr[idx]
Xt, yt = espelha(X_te), y_te
print(f"adaptação com {len(idx)} exemplos do domínio novo:\n")
print(f"{'método':28s} {'parâmetros treináveis':>22s}  acc domínio novo   acc domínio original (esquecimento)")


def relatorio(nome, m):
    print(f"{nome:28s} {n_treinaveis(m):>12,} ({100 * n_treinaveis(m) / total:5.1f}%)   {acc(m, Xt, yt):.3f}              {acc(m, X_te, y_te):.3f}")


relatorio("sem adaptação", nn.Sequential(*[copy.deepcopy(l) for l in base]))

m = copy.deepcopy(base)
for p in m[:-1].parameters():
    p.requires_grad = False
treina(m, Xa, ya, 200, 3e-3)
relatorio("só última camada", m)

for r in (1, 4, 8):
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad = False
    m[0], m[2] = LoRALinear(m[0], r, 2 * r), LoRALinear(m[2], r, 2 * r)  # LoRA nas duas camadas ocultas
    treina(m, Xa, ya, 200, 3e-3)
    relatorio(f"LoRA r={r} (2 camadas)", m)
    if r == 4:  # verifica que o merge não muda as predições
        antes = m(Xt)
        m[0], m[2] = m[0].merge(), m[2].merge()
        print(f"{'':28s} (merge de r=4: diferença máxima nas saídas = {(m(Xt) - antes).abs().max().item():.1e})")

m = copy.deepcopy(base)
treina(m, Xa, ya, 200, 1e-3, params=m.parameters())
relatorio("fine-tuning completo", m)
