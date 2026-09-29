# Aprendizado por Reforço

Agentes que aprendem por tentativa e erro: da formalização (MDP) e dos bandits ao deep RL e gradientes de política.

**Pré-requisitos:** módulos 00 (probabilidade) e 05 (para a parte profunda).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`bandits/`](bandits/) | ε-greedy, UCB, Thompson Sampling | ✅ | — |
| [`mdp-bellman/`](mdp-bellman/) | MDP, equações de Bellman, iteração de valor/política | ✅ | — |
| [`tabular-q-learning/`](tabular-q-learning/) | TD, Q-learning × SARSA (Cliff Walking) | ✅ | — |
| [`policy-gradient/`](policy-gradient/) | REINFORCE (CartPole do zero em C++), PPO | ✅ | ✅ |
| [`deep-rl/`](deep-rl/) | DQN: replay buffer, rede alvo | — | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
