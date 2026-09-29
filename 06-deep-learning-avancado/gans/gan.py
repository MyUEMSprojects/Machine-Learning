"""GAN em PyTorch aprendendo uma mistura de 8 gaussianas dispostas em círculo (2D).

Mede a cobertura dos modos e a qualidade das amostras. Rode com sementes diferentes para ver a instabilidade
típica do treino de GANs (colapso de modos).

pip install torch numpy
"""
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
np.random.seed(0)
K, RAIO, SIGMA = 8, 2.0, 0.05
centros = torch.tensor([[RAIO * np.cos(2 * np.pi * k / K), RAIO * np.sin(2 * np.pi * k / K)] for k in range(K)], dtype=torch.float32)


def dados_reais(n):
    idx = torch.randint(0, K, (n,))
    return centros[idx] + SIGMA * torch.randn(n, 2)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, o))


G, D = mlp(8, 2), mlp(2, 1)
optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
bce = nn.functional.binary_cross_entropy_with_logits


def avalia(n=4000):
    with torch.no_grad():
        x = G(torch.randn(n, 8))
        d = torch.cdist(x, centros)                      # distância a cada modo
        perto = d.min(1).values < 3 * SIGMA * 1.5        # amostra "de boa qualidade"
        modos = torch.unique(d.argmin(1)[perto]).numel() # modos cobertos
    return perto.float().mean().item(), modos


for passo in range(1, 6001):
    real, z = dados_reais(256), torch.randn(256, 8)
    fake = G(z)
    # 1) discriminador: real -> 1, falso -> 0
    lossD = bce(D(real), torch.ones(256, 1)) + bce(D(fake.detach()), torch.zeros(256, 1))
    optD.zero_grad(); lossD.backward(); optD.step()
    # 2) gerador (loss não saturante): quer que D diga "real" para o falso
    lossG = bce(D(fake), torch.ones(256, 1))
    optG.zero_grad(); lossG.backward(); optG.step()
    if passo % 1000 == 0:
        q, m = avalia()
        print(f"passo {passo:4d}  lossD={lossD.item():.3f}  lossG={lossG.item():.3f}  qualidade={q:.2f}  modos cobertos={m}/{K}")
