# Q-Learning e SARSA (métodos tabulares)

Aprendizado por reforço **sem modelo** do ambiente: o agente aprende $Q(s,a)$ apenas por interação (tentativa e erro), guardando uma tabela.

## Diferença temporal (TD)

Combina Monte Carlo (aprende de experiência) e programação dinâmica (*bootstrapping*: atualiza estimativa com outra estimativa) — aprende a cada passo, sem esperar o fim do episódio.

### Q-learning (off-policy)

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]$$

Aprende $Q^*$ **independentemente** da política usada para explorar. O termo entre colchetes é o **erro TD**.

### SARSA (on-policy)

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\,Q(s',a')-Q(s,a)\Big]$$

com $a'$ a ação realmente escolhida pela política (com exploração) → aprende o valor da política que está seguindo, **incluindo** os erros de exploração.

## Exploração × aproveitamento

**$\varepsilon$-greedy**: com prob. $\varepsilon$ age aleatoriamente. Em geral decai $\varepsilon$ ao longo do tempo. Alternativas: softmax/Boltzmann, otimismo inicial, UCB, bônus de curiosidade.

## Hiperparâmetros

$\alpha$ (taxa de aprendizado), $\gamma$ (desconto), $\varepsilon$. Converge a $Q^*$ se todos os pares $(s,a)$ são visitados infinitamente e $\alpha$ decai apropriadamente.

## Limitações → deep RL

A tabela cresce com $|\mathcal S|\times|\mathcal A|$: inviável para estados contínuos/imagens. Solução: aproximar $Q$ por rede neural ([DQN](../deep-rl/)). Variantes: *Expected SARSA*, *Double Q-learning* (reduz viés de maximização), $n$-step, TD($\lambda$).

## Implementação

[`q_learning.cpp`](q_learning.cpp) (C++): **Cliff Walking** 4×12 comparando Q-learning e SARSA (média de 50 execuções); imprime o caminho guloso de cada um — Q-learning vai rente ao penhasco (ótimo), SARSA pela rota segura.

```bash
g++ -std=c++17 -O2 q_learning.cpp -o q_learning && ./q_learning
```

## Referências

- Sutton & Barto, *Reinforcement Learning: An Introduction*, cap. 6.
- Watkins & Dayan, *Q-learning* (1992).
