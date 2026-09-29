"""Transfer learning em PyTorch: o que é aprendido numa tarefa de origem ajuda numa tarefa nova com poucos dados?

Origem: dígitos 0-4, treinados com deslocamentos aleatórios (±2 px) -> a rede aprende a ser robusta a
translações. Destino: dígitos 5-9 (classes NOVAS), só 10 exemplos/classe, avaliados em imagens deslocadas.
Compara (a) do zero, (b) extrator congelado + nova cabeça, (c) fine-tuning com LR diferencial.
Média de 5 sementes. Ao final, a receita padrão com modelos pré-treinados do torchvision.

pip install torch scikit-learn
"""
import copy

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
POUCOS = 10  # exemplos por classe no destino


def desloca(x, mx=2):
    img = x.view(-1, 8, 8)
    out = torch.empty_like(img)
    for i in range(len(img)):
        dx, dy = np.random.randint(-mx, mx + 1, 2)
        out[i] = torch.roll(img[i], shifts=(int(dx), int(dy)), dims=(0, 1))
    return out.view(-1, 64)


def corpo():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())


def treina(params, forward, X, y, epocas=300, lr=1e-2):
    opt = torch.optim.Adam(params, lr)
    for _ in range(epocas):
        opt.zero_grad()
        nn.functional.cross_entropy(forward(X), y).backward()
        opt.step()


def experimento(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    src = y < 5
    Xs, ys = X[src], y[src]
    Xd, yd = X[~src], y[~src] - 5
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.6, stratify=yd, random_state=seed)
    Xd_te = desloca(Xd_te)
    idx = torch.cat([torch.where(yd_tr == c)[0][:POUCOS] for c in range(5)])
    Xf, yf = desloca(Xd_tr[idx]), yd_tr[idx]

    def acc(f):
        with torch.no_grad():
            return (f(Xd_te).argmax(1) == yd_te).float().mean().item()

    # 1) pré-treino na origem (com deslocamentos novos a cada época = aumento de dados)
    base, cab_src = corpo(), nn.Linear(64, 5)
    opt = torch.optim.Adam(list(base.parameters()) + list(cab_src.parameters()), 3e-3)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(cab_src(base(desloca(Xs))), ys).backward()
        opt.step()

    # (a) do zero
    z, c = corpo(), nn.Linear(64, 5)
    treina(list(z.parameters()) + list(c.parameters()), lambda x: c(z(x)), Xf, yf)
    a = acc(lambda x: c(z(x)))
    # (b) extrator congelado + nova cabeça
    b_corpo, b_cab = copy.deepcopy(base), nn.Linear(64, 5)
    for p in b_corpo.parameters():
        p.requires_grad = False
    treina(b_cab.parameters(), lambda x: b_cab(b_corpo(x)), Xf, yf)
    b = acc(lambda x: b_cab(b_corpo(x)))
    # (c) fine-tuning: LR menor no corpo pré-treinado, maior na cabeça nova (LR diferencial)
    c_corpo, c_cab = copy.deepcopy(base), nn.Linear(64, 5)
    opt = torch.optim.Adam([{"params": c_corpo.parameters(), "lr": 1e-3}, {"params": c_cab.parameters(), "lr": 1e-2}])
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(c_cab(c_corpo(Xf)), yf).backward()
        opt.step()
    return a, b, acc(lambda x: c_cab(c_corpo(x)))


r = np.mean([experimento(s) for s in range(5)], axis=0)
print(f"Destino: dígitos 5-9, {POUCOS} exemplos/classe, teste com imagens deslocadas (média de 5 sementes)")
print(f"  (a) do zero                      : {r[0]:.3f}")
print(f"  (b) extrator congelado + cabeça  : {r[1]:.3f}")
print(f"  (c) fine-tuning (LR diferencial) : {r[2]:.3f}")
print("""
Obs.: nos dígitos "limpos" e sem deslocamento, treinar do zero com poucos exemplos já vai bem e a transferência
não ajuda — transfer learning brilha quando a origem ensina algo (aqui, invariância a translação) que o
destino, com poucos dados, não conseguiria aprender sozinho.

Receita com modelos de visão pré-treinados (torchvision; baixa pesos da internet):

    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    for p in m.parameters(): p.requires_grad = False        # congela o backbone
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)   # nova cabeça treinável
    # depois, opcionalmente, descongele as últimas camadas e reduza o LR (fine-tuning)
""")
