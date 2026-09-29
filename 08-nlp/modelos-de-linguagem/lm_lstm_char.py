"""Modelo de linguagem neural em nível de caractere (LSTM) treinado nos próprios READMEs deste repositório.

Mostra o ciclo completo: dados -> treino (prever o próximo caractere) -> perplexidade -> geração com temperatura.

pip install torch
"""
import math
from pathlib import Path

import torch
from torch import nn

torch.manual_seed(0)
raiz = Path(__file__).resolve().parents[2]
texto = "\n".join(p.read_text(encoding="utf-8") for p in sorted(raiz.rglob("README.md")))
chars = sorted(set(texto))
c2i = {c: i for i, c in enumerate(chars)}
dados = torch.tensor([c2i[c] for c in texto])
corte = int(0.9 * len(dados))
treino, val = dados[:corte], dados[corte:]
print(f"corpus: {len(texto):,} caracteres, vocabulário: {len(chars)}")

SEQ, BATCH, H = 128, 64, 256


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(chars), 64)
        self.lstm = nn.LSTM(64, H, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(H, len(chars))

    def forward(self, x, h=None):
        y, h = self.lstm(self.emb(x), h)
        return self.out(y), h


def lote(d):
    ix = torch.randint(0, len(d) - SEQ - 1, (BATCH,))
    x = torch.stack([d[i : i + SEQ] for i in ix])
    y = torch.stack([d[i + 1 : i + SEQ + 1] for i in ix])
    return x, y


@torch.no_grad()
def perda_val(m):
    m.eval()
    ls = []
    for _ in range(20):
        x, y = lote(val)
        ls.append(nn.functional.cross_entropy(m(x)[0].reshape(-1, len(chars)), y.reshape(-1)).item())
    m.train()
    return sum(ls) / len(ls)


m = LM()
opt = torch.optim.Adam(m.parameters(), 3e-3)
print(f"perplexidade inicial (~vocabulário = {len(chars)}): {math.exp(perda_val(m)):.1f}")
for passo in range(1, 701):
    x, y = lote(treino)
    loss = nn.functional.cross_entropy(m(x)[0].reshape(-1, len(chars)), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if passo % 100 == 0:
        v = perda_val(m)
        print(f"passo {passo:4d}  loss treino={loss.item():.3f}  loss val={v:.3f}  perplexidade val={math.exp(v):.2f}  ({v / math.log(2):.2f} bits/char)")


@torch.no_grad()
def gera(prompt, n=300, temp=0.8):
    m.eval()
    x = torch.tensor([[c2i.get(c, 0) for c in prompt]])
    out, h = list(prompt), None
    logits, h = m(x, h)
    for _ in range(n):
        p = (logits[0, -1] / temp).softmax(-1)  # temperatura: <1 conservador, >1 criativo/caótico
        i = torch.multinomial(p, 1).item()
        out.append(chars[i])
        logits, h = m(torch.tensor([[i]]), h)
    return "".join(out)


for t in (0.5, 1.0):
    print(f"\n--- temperatura {t} ---")
    print(gera("O algoritmo ", 250, t))
