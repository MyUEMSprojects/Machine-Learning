# Modelos Gráficos Probabilísticos

Usam um grafo para representar a **estrutura de dependência** entre variáveis aleatórias, tornando distribuições conjuntas enormes tratáveis.

## Redes Bayesianas (grafo dirigido acíclico)

Cada nó $X_i$ tem uma tabela de probabilidade condicional $P(X_i\mid\mathrm{pais}(X_i))$ e a conjunta fatora:

$$P(X_1,\dots,X_n)=\prod_i P(X_i\mid\mathrm{pais}(X_i))$$

Com $n$ variáveis binárias, a conjunta tem $2^n-1$ parâmetros; a rede, só $\sum_i2^{|\mathrm{pais}_i|}$.

### Independências (d-separação)

Três estruturas básicas de trilha entre $A$ e $C$ via $B$:

| Estrutura | Nome | $A\perp C$? | Dado $B$? |
|---|---|---|---|
| $A\to B\to C$ | cadeia | não | **sim** |
| $A\leftarrow B\to C$ | causa comum | não | **sim** |
| $A\to B\leftarrow C$ | **colisor** | sim | **não** (observar $B$ ou descendente "abre" o caminho → *explaining away*) |

## Redes de Markov / MRF (grafo não dirigido)

Distribuição por potenciais nas cliques: $P(x)=\frac1Z\prod_c\psi_c(x_c)$. Exemplos: modelo de Ising, campos aleatórios condicionais (CRF), máquinas de Boltzmann. HMMs e Naive Bayes são casos particulares de redes bayesianas.

## Inferência

Calcular $P(\text{consulta}\mid\text{evidência})$:

- **Exata**: enumeração (exponencial), **eliminação de variáveis**, árvore de junção / *belief propagation* (exata em árvores). NP-difícil no caso geral.
- **Aproximada**: amostragem (rejeição, Gibbs, MCMC), inferência variacional, *loopy* BP.

## Aprendizado

- Parâmetros (estrutura conhecida): contagem/MLE, EM se há variáveis latentes.
- Estrutura: busca com score (BIC) ou testes de independência (PC).

## Implementação

[`rede_bayesiana.cpp`](rede_bayesiana.cpp) (C++): rede do **Alarme** (Pearl) com inferência exata por enumeração; demonstra *explaining away* e independência condicional.

```bash
g++ -std=c++17 -O2 rede_bayesiana.cpp -o rede_bayesiana && ./rede_bayesiana
```

(Bibliotecas: `pgmpy`, `pomegranate`, `PyMC` — ver [inferência bayesiana](../inferencia-bayesiana/).)

## Referências

- Koller & Friedman, *Probabilistic Graphical Models*.
- Pearl, *Probabilistic Reasoning in Intelligent Systems*.
