# Deep Reinforcement Learning (Deep Q-Networks)

Quando o espaço de estados é enorme ou contínuo (pixels, sensores), aproximamos $Q(s,a;\theta)$ por uma **rede neural** em vez de uma tabela.

## DQN (Mnih et al., 2015)

Q-learning com rede neural minimizando o erro TD:

$$L(\theta)=\mathbb E\Big[\big(r+\gamma\max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta)\big)^2\Big]$$

Combinar Q-learning, redes profundas e amostras correlacionadas é instável (a "tríade mortal"). Dois truques o tornaram viável:

1. **Replay buffer**: guarda transições $(s,a,r,s',d)$ e treina em mini-lotes **aleatórios** → quebra correlação temporal e reutiliza dados (eficiência amostral).
2. **Rede alvo** $\theta^-$: cópia congelada de $\theta$, atualizada a cada $C$ passos (ou média móvel) → alvo estável, sem "perseguir o próprio rabo".

Mais: $\varepsilon$-greedy com decaimento, perda de Huber, *gradient clipping*. Detalhe importante: use `terminated` (estado terminal real) — e não `truncated` (limite de tempo) — para zerar o valor futuro.

## Melhorias (Rainbow)

**Double DQN** (reduz superestimação: escolhe ação com $\theta$, avalia com $\theta^-$), **Dueling** (separa $V(s)$ e vantagem $A(s,a)$), **Prioritized Replay** (amostra transições de maior erro TD), **Noisy Nets**, **multi-step**, **distribucional (C51)**.

## Panorama

| Família | Exemplos | Ações | Nota |
|---|---|---|---|
| Baseada em valor | DQN, Rainbow | discretas | *off-policy*, eficiente em amostras |
| [Gradiente de política](../policy-gradient/) | REINFORCE, PPO | discretas/contínuas | *on-policy* (PPO), estável |
| Ator-crítico *off-policy* | DDPG, TD3, SAC | contínuas | robótica |
| Baseada em modelo | Dyna, MuZero, Dreamer | ambas | aprendem um modelo do mundo; alta eficiência amostral |

Marcos: Atari (DQN, 2015), AlphaGo/AlphaZero (busca em árvore + redes, 2016–17), OpenAI Five/AlphaStar, RLHF em LLMs. Desafios: eficiência amostral, recompensas esparsas, generalização, segurança e *reward hacking*.

## Implementação

[`dqn_cartpole.py`](dqn_cartpole.py) (Python/PyTorch + Gymnasium): DQN completo (replay buffer, rede alvo, $\varepsilon$-greedy, Huber) no CartPole — o retorno médio cresce de ~25 para ~400 em 250 episódios.

```bash
pip install torch gymnasium
python dqn_cartpole.py
```

## Referências

- Mnih et al., *Human-level control through deep reinforcement learning* (Nature, 2015).
- Hessel et al., *Rainbow: Combining Improvements in Deep RL* (2018).
- Sutton & Barto, caps. 9–11; Stable-Baselines3 (implementações de referência).
