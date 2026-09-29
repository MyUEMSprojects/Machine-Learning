"""DQN (Deep Q-Network) no CartPole com PyTorch + Gymnasium: replay buffer, rede alvo e epsilon-greedy.

pip install torch gymnasium
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn

random.seed(0); np.random.seed(0); torch.manual_seed(0)
env = gym.make("CartPole-v1")

q = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
q_alvo = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
q_alvo.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), 1e-3)
buffer = deque(maxlen=20_000)  # replay buffer: quebra a correlação temporal das amostras
GAMMA, BATCH, SYNC = 0.99, 64, 200

passos, retornos = 0, []
for ep in range(1, 251):
    obs, _ = env.reset(seed=ep)
    ret, done = 0.0, False
    while not done:
        eps = max(0.05, 1.0 - passos / 5000)  # decaimento linear da exploração
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q(torch.tensor(obs, dtype=torch.float32)).argmax().item()
        prox, r, term, trunc, _ = env.step(a)
        buffer.append((obs, a, r, prox, float(term)))  # 'term' (não 'trunc'): fim por tempo não é estado terminal
        obs, ret, done, passos = prox, ret + r, term or trunc, passos + 1

        if len(buffer) >= 1000:
            o, ac, rw, on, dn = map(np.array, zip(*random.sample(buffer, BATCH)))
            o, on = torch.tensor(o, dtype=torch.float32), torch.tensor(on, dtype=torch.float32)
            ac = torch.tensor(ac, dtype=torch.int64)
            rw, dn = torch.tensor(rw, dtype=torch.float32), torch.tensor(dn, dtype=torch.float32)
            with torch.no_grad():  # alvo TD: r + gamma * max_a' Q_alvo(s', a')  (rede alvo estabiliza o treino)
                alvo = rw + GAMMA * (1 - dn) * q_alvo(on).max(1).values
            pred = q(o).gather(1, ac[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, alvo)  # Huber
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10)
            opt.step()
        if passos % SYNC == 0:
            q_alvo.load_state_dict(q.state_dict())
    retornos.append(ret)
    if ep % 25 == 0:
        print(f"episódio {ep:3d}  epsilon={eps:.2f}  retorno médio (últimos 25) = {np.mean(retornos[-25:]):.1f}")
