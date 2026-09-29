"""Mini Transformer em PyTorch: atenção multi-cabeça escrita à mão + encoder treinado para INVERTER sequências.

Cada posição de saída precisa "olhar" a posição espelhada da entrada: o modelo aprende isso só pela atenção
e pelos embeddings posicionais. Também imprime o mapa de atenção aprendido.

pip install torch
"""
import math

import torch
from torch import nn

torch.manual_seed(0)
VOCAB, N, D, HEADS, LAYERS = 12, 8, 64, 4, 2


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, return_weights=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))  # (B, h, T, dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                          # (B, h, T, T)
        w = scores.softmax(-1)
        o = (w @ v).transpose(1, 2).reshape(B, T, d)
        return (self.out(o), w) if return_weights else self.out(o)


class Bloco(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MultiHeadAttention(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, return_weights=False):
        a = self.att(self.ln1(x), return_weights)
        w = None
        if return_weights:
            a, w = a
        x = x + a                       # conexão residual
        x = x + self.ffn(self.ln2(x))
        return (x, w) if return_weights else x


class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(N, D)   # posição aprendida (alternativa à sinusoidal)
        self.blocos = nn.ModuleList(Bloco(D, HEADS) for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, x, return_weights=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        ws = []
        for b in self.blocos:
            h, w = b(h, True) if return_weights else (b(h), None)
            ws.append(w)
        logits = self.head(self.ln(h))
        return (logits, ws) if return_weights else logits


modelo = MiniTransformer()
opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
print(f"parâmetros: {sum(p.numel() for p in modelo.parameters()):,}")
for passo in range(1, 1501):
    x = torch.randint(0, VOCAB, (128, N))
    loss = nn.functional.cross_entropy(modelo(x).reshape(-1, VOCAB), x.flip(1).reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if passo % 250 == 0:
        with torch.no_grad():
            xt = torch.randint(0, VOCAB, (1000, N))
            acc = (modelo(xt).argmax(-1) == xt.flip(1)).float().mean().item()
        print(f"passo {passo:4d}  loss={loss.item():.4f}  acc por token (teste)={acc:.3f}")

x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
with torch.no_grad():
    logits, ws = modelo(x, return_weights=True)
print("entrada:", x[0].tolist(), " saída:", logits.argmax(-1)[0].tolist())
print("atenção média (cabeças) da 1ª camada — linha i = onde a saída i olha:")
print((ws[0][0].mean(0) * 100).round().int())
