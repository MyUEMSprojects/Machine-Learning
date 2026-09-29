"""Mini-GPT (Transformer decoder-only) do zero em PyTorch, em nível de caractere, treinado nos READMEs do repositório.

Inclui atenção causal multi-cabeça, pré-LayerNorm, MLP com GELU, embeddings de token+posição, perda de próximo token
e geração autorregressiva com temperatura/top-k.

pip install torch
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
raiz = Path(__file__).resolve().parents[2]
texto = "\n".join(p.read_text(encoding="utf-8") for p in sorted(raiz.rglob("README.md")))
chars = sorted(set(texto))
c2i = {c: i for i, c in enumerate(chars)}
dados = torch.tensor([c2i[c] for c in texto])
corte = int(0.9 * len(dados))
treino, val = dados[:corte], dados[corte:]
V, T, D, H, L, B = len(chars), 128, 128, 4, 4, 32  # vocab, contexto, dim, cabeças, camadas, batch
print(f"corpus: {len(texto):,} caracteres, vocabulário: {V}")


class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = (z.view(b, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # softmax(QK^T/sqrt(d)+máscara causal)V
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocos = nn.Sequential(*[Bloco() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)
        self.head.weight = self.tok.weight  # amarra os pesos de entrada e saída (weight tying)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        return self.head(self.ln(self.blocos(h)))

    @torch.no_grad()
    def gerar(self, prompt, n=300, temp=0.8, top_k=20):
        self.eval()
        x = torch.tensor([[c2i.get(c, 0) for c in prompt]])
        for _ in range(n):
            logits = self(x[:, -T:])[0, -1] / temp
            if top_k:
                corte_k = logits.topk(top_k).values[-1]
                logits[logits < corte_k] = -float("inf")  # descarta tudo fora dos k mais prováveis
            x = torch.cat([x, torch.multinomial(logits.softmax(-1), 1)[None]], 1)
        self.train()
        return "".join(chars[i] for i in x[0])


def lote(d):
    ix = torch.randint(0, len(d) - T - 1, (B,))
    return torch.stack([d[i : i + T] for i in ix]), torch.stack([d[i + 1 : i + T + 1] for i in ix])


@torch.no_grad()
def perda_val(m):
    m.eval()
    ls = [F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item() for x, y in (lote(val) for _ in range(20))]
    m.train()
    return sum(ls) / len(ls)


m = GPT()
print(f"parâmetros: {sum(p.numel() for p in m.parameters()):,}")
opt = torch.optim.AdamW(m.parameters(), 1e-3 * 3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=1500)
for passo in range(1, 1501):
    x, y = lote(treino)
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sched.step()
    if passo % 300 == 0:
        v = perda_val(m)
        print(f"passo {passo:4d}  loss treino={loss.item():.3f}  loss val={v:.3f}  perplexidade val={math.exp(v):.2f}")
print("\n--- amostra (temp=0.8, top-k=20) ---")
print(m.gerar("O algoritmo ", 300))
