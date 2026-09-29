"""Comparação de otimizadores e schedulers do PyTorch em uma regressão logística mal condicionada.

pip install torch
"""
import torch

torch.manual_seed(0)
n = 400
Xb = torch.randn(n, 3) * torch.tensor([1.0, 30.0, 0.03])  # escalas muito diferentes
w_true = torch.tensor([1.0, 0.05, -20.0])
y = ((Xb @ w_true + 0.5 * torch.randn(n)) > 0).float()


def treina(make_opt, steps=500, sched=None):
    w = torch.zeros(3, requires_grad=True)
    opt = make_opt([w])
    s = sched(opt) if sched else None
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xb @ w, y)
        loss.backward()
        opt.step()
        if s:
            s.step()
    return loss.item()


P = torch.optim
print(f"SGD              {treina(lambda p: P.SGD(p, lr=1e-3)):.4f}")
print(f"SGD+momentum     {treina(lambda p: P.SGD(p, lr=1e-3, momentum=0.9)):.4f}")
print(f"SGD+Nesterov     {treina(lambda p: P.SGD(p, lr=1e-3, momentum=0.9, nesterov=True)):.4f}")
print(f"RMSprop          {treina(lambda p: P.RMSprop(p, lr=1e-2)):.4f}")
print(f"Adam             {treina(lambda p: P.Adam(p, lr=5e-2)):.4f}")
print(f"AdamW            {treina(lambda p: P.AdamW(p, lr=5e-2, weight_decay=1e-2)):.4f}")
print(f"Adam + cosine LR {treina(lambda p: P.Adam(p, lr=5e-2), sched=lambda o: P.lr_scheduler.CosineAnnealingLR(o, 500)):.4f}")
