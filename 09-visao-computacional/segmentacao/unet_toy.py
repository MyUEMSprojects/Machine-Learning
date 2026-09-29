"""Segmentação semântica com uma U-Net minúscula em PyTorch, em imagens sintéticas (formas ruidosas com iluminação
variável). Compara com o limiar de Otsu (scikit-image) usando IoU e Dice.

pip install torch scikit-image numpy
"""
import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
S = 32


def amostra():
    """Imagem SxS com 1-3 formas (círculos/quadrados) mais claras que o fundo, gradiente de luz e ruído forte."""
    yy, xx = np.mgrid[:S, :S]
    m = np.zeros((S, S), dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        cy, cx, r = rng.integers(6, S - 6, 2).tolist() + [rng.integers(3, 7)]
        if rng.random() < 0.5:
            m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
        else:
            m[(abs(yy - cy) <= r) & (abs(xx - cx) <= r)] = 1
    luz = rng.uniform(-0.2, 0.2) * xx / S + rng.uniform(-0.2, 0.2) * yy / S
    im = 0.3 + 0.3 * m + luz + rng.normal(0, 0.15, (S, S))
    return im.astype(np.float32), m


def dataset(n):
    xs, ms = zip(*[amostra() for _ in range(n)])
    return torch.tensor(np.stack(xs))[:, None], torch.tensor(np.stack(ms))[:, None]


def bloco(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    """Encoder (reduz resolução, aumenta canais) + decoder (aumenta resolução) com **skip connections** que
    reinjetam detalhes espaciais finos do encoder no decoder."""

    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.mid = bloco(1, 16), bloco(16, 32), bloco(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.d2 = nn.ConvTranspose2d(64, 32, 2, stride=2), bloco(64, 32)
        self.up1, self.d1 = nn.ConvTranspose2d(32, 16, 2, stride=2), bloco(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)                                   # S x S
        e2 = self.e2(self.pool(e1))                       # S/2
        m = self.mid(self.pool(e2))                       # S/4
        d2 = self.d2(torch.cat([self.up2(m), e2], 1))     # skip connection
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))    # skip connection
        return self.out(d1)                               # logits por pixel


def metricas(pred, gt):
    inter = (pred * gt).sum((1, 2, 3))
    uniao = pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) - inter
    return (inter / (uniao + 1e-9)).mean().item(), (2 * inter / (pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1e-9)).mean().item()


X_tr, M_tr = dataset(1200)
X_te, M_te = dataset(200)
net = UNet()
opt = torch.optim.Adam(net.parameters(), 3e-3)
for ep in range(1, 9):
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), 32):
        b = perm[i : i + 32]
        opt.zero_grad()
        p = net(X_tr[b])
        # perda = BCE + Dice (Dice lida melhor com desbalanceamento fundo/objeto)
        prob = p.sigmoid()
        dice = 1 - (2 * (prob * M_tr[b]).sum() + 1) / (prob.sum() + M_tr[b].sum() + 1)
        (nn.functional.binary_cross_entropy_with_logits(p, M_tr[b]) + dice).backward()
        opt.step()
    if ep % 2 == 0:
        net.eval()
        with torch.no_grad():
            iou, d = metricas((net(X_te) > 0).float(), M_te)
        net.train()
        print(f"época {ep}: U-Net  IoU={iou:.3f}  Dice={d:.3f}")

otsu = torch.stack([torch.tensor((x[0].numpy() > threshold_otsu(x[0].numpy())).astype(np.float32))[None] for x in X_te])
iou, d = metricas(otsu, M_te)
print(f"baseline limiar de Otsu (global): IoU={iou:.3f}  Dice={d:.3f}  <- falha com iluminação variável e ruído")
