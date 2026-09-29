"""DDPM (Denoising Diffusion Probabilistic Model) em dados 2D (duas luas), do zero em PyTorch.

Processo direto: adiciona ruído gaussiano em T passos. Rede aprende a prever o ruído; a geração
inverte o processo partindo de ruído puro.

pip install torch scikit-learn
"""
import math

import torch
from sklearn.datasets import make_moons
from torch import nn

torch.manual_seed(0)
T = 100
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)  # \bar{alpha}_t


def dados(n):
    x, _ = make_moons(n, noise=0.05, random_state=None)
    return (torch.tensor(x, dtype=torch.float32) - torch.tensor([0.5, 0.25])) * 2.0


class EpsNet(nn.Module):
    """epsilon_theta(x_t, t): MLP com embedding sinusoidal do passo t."""

    def __init__(self, h=128, dt=32):
        super().__init__()
        self.dt = dt
        self.net = nn.Sequential(nn.Linear(2 + dt, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def temb(self, t):
        freqs = torch.exp(-math.log(1000) * torch.arange(self.dt // 2) / (self.dt // 2))
        a = t[:, None].float() * freqs[None]
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], 1))


net = EpsNet()
opt = torch.optim.Adam(net.parameters(), 2e-3)
for passo in range(1, 5001):
    x0 = dados(512)
    t = torch.randint(0, T, (512,))
    eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps  # q(x_t | x_0) em forma fechada
    loss = nn.functional.mse_loss(net(xt, t), eps)                            # prever o ruído adicionado
    opt.zero_grad(); loss.backward(); opt.step()
    if passo % 1000 == 0:
        print(f"passo {passo}  loss={loss.item():.4f}")


@torch.no_grad()
def amostra(n):
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps = net(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)  # ruído só até t=1
    return x


gerado = amostra(2000)
real = dados(2000)
# Métrica simples: distância de cada amostra ao ponto real mais próximo (menor = mais "na variedade")
d_gen = torch.cdist(gerado, real).min(1).values.mean().item()
d_ruido = torch.cdist(torch.randn(2000, 2), real).min(1).values.mean().item()
d_real = torch.cdist(dados(2000), real).min(1).values.mean().item()
print(f"dist. média ao vizinho real mais próximo — gerado: {d_gen:.3f} | real vs real: {d_real:.3f} | ruído gaussiano: {d_ruido:.3f}")
print("média/desvio dos dados reais:", real.mean(0).numpy().round(2), real.std(0).numpy().round(2))
print("média/desvio dos gerados    :", gerado.mean(0).numpy().round(2), gerado.std(0).numpy().round(2))
