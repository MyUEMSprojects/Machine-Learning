# Multi-Armed Bandits

O problema **mais simples de RL**: sem estados, só $K$ ações ("braços") com recompensas de distribuição desconhecida. Em cada rodada escolhe-se um braço e observa-se a recompensa. O dilema central é **exploração × aproveitamento** (*exploration–exploitation*).

**Arrependimento** (*regret*): $\rho_T=T\mu^*-\sum_{t}\mu_{a_t}$ — quanto se deixou de ganhar por não jogar sempre o melhor braço. Bons algoritmos têm arrependimento **sublinear** (logarítmico em $T$).

## Estratégias

| Algoritmo | Regra | Comentário |
|---|---|---|
| **Greedy** | $a=\arg\max\hat\mu_a$ | pode travar num braço ruim (nunca explora) |
| **$\varepsilon$-greedy** | com prob. $\varepsilon$ aleatório | arrependimento linear se $\varepsilon$ fixo; decaia $\varepsilon$ |
| **UCB1** | $a=\arg\max\;\hat\mu_a+\sqrt{\tfrac{2\ln t}{n_a}}$ | *otimismo diante da incerteza*: bônus maior para braços pouco testados |
| **Thompson Sampling** | amostra $\theta_a\sim$ posterior e joga o argmax | bayesiano; para Bernoulli, posterior $\mathrm{Beta}(s_a+1,f_a+1)$; muito eficaz na prática |
| **Gradient bandit** | preferências $H_a$ + softmax, subida de gradiente | não estima valores |

Média incremental: $\hat\mu\leftarrow\hat\mu+\frac{1}{n}(r-\hat\mu)$ (com passo constante $\alpha$ se o ambiente é não estacionário).

## Extensões e aplicações

- **Bandits contextuais** (LinUCB): a recompensa depende de um contexto/feature (recomendação de notícias, anúncios).
- Testes A/B adaptativos, alocação em ensaios clínicos, seleção de hiperparâmetros (*Hyperband*), roteamento.
- RL completo adiciona **estados** e **consequências de longo prazo** das ações ([MDP](../mdp-bellman/)).

## Implementação

[`bandits.cpp`](bandits.cpp) (C++): testbed de 10 braços Bernoulli (500 execuções × 2000 passos) comparando aleatório, greedy, ε-greedy (0,01 e 0,1), UCB1 e Thompson; mede recompensa média, % de ação ótima e arrependimento.

```bash
g++ -std=c++17 -O2 bandits.cpp -o bandits && ./bandits
```

## Referências

- Sutton & Barto, cap. 2; Lattimore & Szepesvári, *Bandit Algorithms* (gratuito online).
- Auer et al., *Finite-time Analysis of the Multiarmed Bandit Problem* (UCB, 2002).
- Russo et al., *A Tutorial on Thompson Sampling* (2018).
