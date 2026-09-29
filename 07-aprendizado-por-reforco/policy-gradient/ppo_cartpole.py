"""PPO (Proximal Policy Optimization, versão mínima com GAE) no CartPole, em PyTorch + Gymnasium.

pip install torch gymnasium
"""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

torch.manual_seed(0)
np.random.seed(0)
env = gym.make("CartPole-v1")


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, o))


ator, critico = mlp(4, 2), mlp(4, 1)  # política pi(a|s) e função de valor V(s)
opt = torch.optim.Adam(list(ator.parameters()) + list(critico.parameters()), 3e-4)
GAMMA, LAM, CLIP, PASSOS, EPOCAS, MB = 0.99, 0.95, 0.2, 2048, 10, 64

obs, _ = env.reset(seed=0)
ret_ep, retornos_recentes = 0.0, []
for iteracao in range(1, 31):
    O, A, LP, R, D, V = [], [], [], [], [], []
    for _ in range(PASSOS):  # 1) coletar experiência com a política atual
        o = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = Categorical(logits=ator(o))
            a = dist.sample()
            V.append(critico(o).item())
        O.append(o); A.append(a); LP.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(a.item())
        R.append(r); D.append(term)
        ret_ep += r
        if term or trunc:
            retornos_recentes.append(ret_ep); ret_ep = 0.0
            obs, _ = env.reset()
    with torch.no_grad():
        ultimo_v = critico(torch.tensor(obs, dtype=torch.float32)).item()

    # 2) vantagens por GAE(lambda): A_t = sum (gamma*lambda)^k delta_{t+k},  delta = r + gamma V' - V
    adv, g = np.zeros(PASSOS), 0.0
    for t in reversed(range(PASSOS)):
        prox_v = ultimo_v if t == PASSOS - 1 else V[t + 1]
        nao_fim = 1.0 - float(D[t])
        delta = R[t] + GAMMA * prox_v * nao_fim - V[t]
        g = delta + GAMMA * LAM * nao_fim * g
        adv[t] = g
    alvo_v = torch.tensor(adv + np.array(V), dtype=torch.float32)
    adv = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
    O, A, LP = torch.stack(O), torch.stack(A), torch.stack(LP)

    # 3) várias épocas de otimização nos mesmos dados, com a razão de probabilidades "clipada"
    for _ in range(EPOCAS):
        perm = torch.randperm(PASSOS)
        for i in range(0, PASSOS, MB):
            b = perm[i : i + MB]
            dist = Categorical(logits=ator(O[b]))
            razao = (dist.log_prob(A[b]) - LP[b]).exp()                 # pi_novo / pi_velho
            perda_pi = -torch.min(razao * adv[b], razao.clamp(1 - CLIP, 1 + CLIP) * adv[b]).mean()
            perda_v = (critico(O[b]).squeeze(-1) - alvo_v[b]).pow(2).mean()
            perda = perda_pi + 0.5 * perda_v - 0.01 * dist.entropy().mean()  # bônus de entropia
            opt.zero_grad(); perda.backward()
            nn.utils.clip_grad_norm_(list(ator.parameters()) + list(critico.parameters()), 0.5)
            opt.step()
    if iteracao % 3 == 0:
        print(f"iteração {iteracao:2d}  passos={iteracao * PASSOS:6d}  retorno médio (últimos 10 ep.) = {np.mean(retornos_recentes[-10:]):.1f}")
