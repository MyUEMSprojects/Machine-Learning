# Processos de Decisão de Markov e Equações de Bellman

Formalização matemática do aprendizado por reforço.

## MDP

Tupla $(\mathcal S,\mathcal A,P,R,\gamma)$: estados, ações, transições $P(s'|s,a)$, recompensa $R(s,a,s')$ e fator de desconto $\gamma\in[0,1]$. **Propriedade de Markov**: o futuro depende só do estado atual e da ação.

O agente escolhe uma **política** $\pi(a|s)$ para maximizar o **retorno** esperado

$$G_t=\sum_{k\ge0}\gamma^kR_{t+k+1}$$

($\gamma<1$: prefere recompensas próximas e garante que a soma converge.)

## Funções de valor

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$$

### Equações de Bellman (recursão)

Esperança: $\;V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\big[r+\gamma V^\pi(s')\big]$

**Otimalidade**: $\;V^*(s)=\max_a\sum_{s'}P(s'|s,a)\big[r+\gamma V^*(s')\big]$, $\quad Q^*(s,a)=\sum_{s'}P(s'|s,a)\big[r+\gamma\max_{a'}Q^*(s',a')\big]$

A política ótima é gulosa em relação a $Q^*$: $\pi^*(s)=\arg\max_aQ^*(s,a)$.

## Programação dinâmica (modelo $P,R$ conhecido)

| Algoritmo | Ideia |
|---|---|
| **Iteração de valor** | aplicar repetidamente o operador de otimalidade de Bellman (contração com fator $\gamma$ → converge a $V^*$) |
| **Iteração de política** | alternar **avaliação** de $\pi$ (resolver Bellman de expectativa) e **melhoria** (política gulosa); converge em poucas rodadas |

Custo por varredura $O(|\mathcal S|^2|\mathcal A|)$ — inviável para espaços enormes e exige conhecer o modelo. Sem o modelo: aprendizado por [diferença temporal / Q-learning](../tabular-q-learning/) (amostra transições) e, com muitos estados, aproximação de funções ([deep RL](../deep-rl/)).

## Implementação

[`mdp_bellman.cpp`](mdp_bellman.cpp) (C++): gridworld 4×4 com transições estocásticas; iteração de valor, iteração de política (chegam à mesma política) e efeito do fator de desconto.

```bash
g++ -std=c++17 -O2 mdp_bellman.cpp -o mdp_bellman && ./mdp_bellman
```

## Referências

- Sutton & Barto, *Reinforcement Learning: An Introduction*, caps. 3–4 (gratuito online).
- Puterman, *Markov Decision Processes*.
