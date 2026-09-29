"""Meta-learning ("aprender a aprender"): MAML e Reptile em regressão de senoides few-shot (PyTorch).

Cada TAREFA é uma senoide y = A sin(x + fase) com A, fase sorteados. O objetivo é achar uma inicialização dos pesos
da qual poucos passos de gradiente em K=5 exemplos de uma tarefa NOVA já ajustem bem a curva inteira.
Compara: inicialização aleatória, pré-treino "joint" (média de todas as tarefas), Reptile e MAML.

pip install torch
"""
import math

import torch

torch.manual_seed(0)
K, INNER_LR = 5, 0.01


def tarefa():
    A, fase = torch.empty(1).uniform_(0.5, 5.0), torch.empty(1).uniform_(0, math.pi)
    return lambda x: A * torch.sin(x + fase)


def amostra(f, n):
    x = torch.empty(n, 1).uniform_(-5, 5)
    return x, f(x)


def init_params():
    def lin(i, o):
        return [torch.randn(i, o) * math.sqrt(2 / i), torch.zeros(o)]
    return [p.requires_grad_() for p in lin(1, 40) + lin(40, 40) + lin(40, 1)]


def forward(p, x):
    h = torch.relu((x / 5) @ p[0] + p[1])  # entrada escalada para [-1, 1]: estabiliza os passos internos
    h = torch.relu(h @ p[2] + p[3])
    return h @ p[4] + p[5]


def mse(p, x, y):
    return ((forward(p, x) - y) ** 2).mean()


def adapta(p, x, y, passos, lr=INNER_LR, create_graph=False):
    """Passos de gradiente descendente 'internos' numa tarefa; devolve os parâmetros adaptados."""
    p = list(p)
    for _ in range(passos):
        g = torch.autograd.grad(mse(p, x, y), p, create_graph=create_graph)
        p = [w - lr * gw for w, gw in zip(p, g)]
    return p


def avalia(p, passos_lista=(0, 1, 5, 10), n_tarefas=200):
    """MSE em 100 pontos de tarefas NOVAS após adaptar em K=5 exemplos."""
    torch.manual_seed(123)
    res = {s: 0.0 for s in passos_lista}
    for _ in range(n_tarefas):
        f = tarefa()
        xs, ys = amostra(f, K)
        xq, yq = amostra(f, 100)
        for s in passos_lista:
            q = adapta([w.detach().clone().requires_grad_() for w in p], xs, ys, s)
            res[s] += mse(q, xq, yq).item() / n_tarefas
    return res


# ------------------------------- 1) MAML (segunda ordem, 1 passo interno, batch de 10 tarefas)
maml = init_params()
opt = torch.optim.Adam(maml, 1e-3)
for it in range(1, 5001):
    perda = 0.0
    for _ in range(10):
        f = tarefa()
        xs, ys = amostra(f, K)      # conjunto de suporte (para adaptar)
        xq, yq = amostra(f, 10)     # conjunto de consulta (para avaliar a adaptação)
        q = adapta(maml, xs, ys, 1, create_graph=True)   # o grafo guarda como q depende da inicialização
        perda = perda + mse(q, xq, yq) / 10
    opt.zero_grad(); perda.backward(); opt.step()   # gradiente através do passo interno
    if it % 1000 == 0:
        print(f"MAML iteração {it}: perda meta = {perda.item():.3f}")

# ------------------------------- 2) Reptile (só primeira ordem): move a inicialização na direção dos pesos adaptados
rep = init_params()
for it in range(1, 20001):
    f = tarefa()
    xs, ys = amostra(f, 10)
    q = adapta([w.detach().clone().requires_grad_() for w in rep], xs, ys, 5, lr=INNER_LR)
    with torch.no_grad():
        for w, wq in zip(rep, q):
            w += 0.1 * (wq.detach() - w)   # theta <- theta + eps (theta_tarefa - theta)

# ------------------------------- 3) pré-treino "joint": regressão em todas as tarefas misturadas (sem meta-objetivo)
joint = init_params()
opt = torch.optim.Adam(joint, 1e-3)
for it in range(5000):
    x = torch.empty(64, 1).uniform_(-5, 5)
    y = torch.cat([tarefa()(x[i : i + 1]) for i in range(64)])
    opt.zero_grad(); mse(joint, x, y).backward(); opt.step()

print(f"\nMSE em tarefas novas após N passos de gradiente com K={K} exemplos (menor = melhor):")
print(f"{'inicialização':22s} " + " ".join(f"{s:>7d} passos" for s in (0, 1, 5, 10)))
for nome, p in [("aleatória", init_params()), ("pré-treino joint", joint), ("Reptile", rep), ("MAML", maml)]:
    r = avalia(p)
    print(f"{nome:22s} " + " ".join(f"{r[s]:>13.3f}" for s in (0, 1, 5, 10)))
