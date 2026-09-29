"""Classificação de imagens em PyTorch: CNN treinada no MNIST (baixado via torchvision) e inferência
com uma ResNet-18 pré-treinada da ImageNet (torchvision) numa foto de exemplo.

pip install torch torchvision scikit-learn pillow   (baixa dados/pesos na 1ª execução)
"""
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)

# ---------------------------------------------------------------- 1) CNN no MNIST
def carrega_mnist():
    """MNIST via torchvision (baixa ~11 MB para ~/.cache); se a rede falhar, usa os dígitos 8x8 do scikit-learn."""
    from pathlib import Path

    import torchvision
    from sklearn.datasets import load_digits

    try:
        raiz = Path.home() / ".cache" / "mnist"
        tr = torchvision.datasets.MNIST(raiz, train=True, download=True)
        te = torchvision.datasets.MNIST(raiz, train=False, download=True)
        f = lambda d: (d.data[:, None].float() / 255.0, d.targets)
        (X_tr, y_tr), (X_te, y_te) = f(tr), f(te)
        return "MNIST", X_tr[:20000], y_tr[:20000], X_te, y_te  # subconjunto de treino p/ rodar rápido em CPU
    except Exception as e:
        print("  falha ao baixar MNIST:", type(e).__name__, "-> usando dígitos 8x8")
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    y = torch.tensor(y)
    return "dígitos 8x8 (fallback)", X[:1200], y[:1200], X[1200:], y[1200:]


try:
    nome_ds, X_tr, y_tr, X_te, y_te = carrega_mnist()
    print("dataset:", nome_ds, tuple(X_tr.shape))

    cnn = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        nn.AdaptiveAvgPool2d(4),                                      # tamanho fixo p/ qualquer resolução de entrada
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 4 * 4, 10),
    )
    opt = torch.optim.Adam(cnn.parameters(), 2e-3)
    for ep in range(1, 4):
        cnn.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), 128):
            b = perm[i : i + 128]
            opt.zero_grad()
            nn.functional.cross_entropy(cnn(X_tr[b]), y_tr[b]).backward()
            opt.step()
        cnn.eval()
        with torch.no_grad():
            acc = np.mean([(cnn(X_te[i : i + 1000]).argmax(1) == y_te[i : i + 1000]).float().mean().item() for i in range(0, len(X_te), 1000)])
        print(f"época {ep}: acc teste = {acc:.4f}")
except Exception as e:
    print("(pulando treino da CNN:", type(e).__name__, str(e)[:80], ")")

# ------------------------------------------------- 2) modelo pré-treinado (ImageNet) em uma foto
try:
    import torchvision
    from sklearn.datasets import load_sample_image

    pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    modelo = torchvision.models.resnet18(weights=pesos).eval()
    preprocessa = pesos.transforms()  # resize, crop, normalização usados no pré-treino
    for nome in ["china.jpg", "flower.jpg"]:
        img = torch.tensor(load_sample_image(nome)).permute(2, 0, 1)  # (C, H, W) uint8
        with torch.no_grad():
            p = modelo(preprocessa(img).unsqueeze(0)).softmax(-1)[0]
        top = p.topk(3)
        print(nome, "->", [(pesos.meta["categories"][i], round(v.item(), 3)) for v, i in zip(top.values, top.indices)])
except Exception as e:
    print("(pulando ResNet pré-treinada:", type(e).__name__, str(e)[:80], ")")
