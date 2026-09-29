"""RNN, GRU e LSTM em PyTorch na tarefa de memória: lembrar o sinal do primeiro elemento (T passos atrás).

pip install torch
"""
import torch
from torch import nn

torch.manual_seed(0)


class Classificador(nn.Module):
    def __init__(self, tipo, hidden=32):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[tipo](1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)                    # (num_layers, B, H) ou tupla (h, c) para LSTM
        h = h[0] if isinstance(h, tuple) else h
        return self.out(h[-1]).squeeze(-1)    # logit


def lote(B, T):
    x = 0.2 * torch.randn(B, T, 1)          # ruído distrator pequeno
    y = (torch.rand(B) > 0.5).float()
    x[:, 0, 0] = 2 * y - 1                  # o sinal a lembrar (+-1) aparece só no 1º passo
    return x, y


for T in [10, 40, 80]:
    for tipo in ["RNN", "GRU", "LSTM"]:
        m = Classificador(tipo)
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        for _ in range(2000):
            x, y = lote(64, T)
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(x), y).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # clipping contra explosão de gradiente
            opt.step()
        with torch.no_grad():
            x, y = lote(2000, T)
            acc = ((m(x) > 0).float() == y).float().mean().item()
        print(f"T={T:3d}  {tipo:5s} acc={acc:.3f}")
