# Probabilidade

Modelos de ML lidam com incerteza: ruído nos dados, incerteza nos parâmetros, saídas probabilísticas.

## Conceitos-chave

- **Variável aleatória**, distribuição (PMF/PDF/CDF).
- **Esperança** $E[X]$ e **variância** $\mathrm{Var}[X]=E[X^2]-E[X]^2$.
- **Probabilidade condicional e independência**: $P(A|B)=P(A\cap B)/P(B)$.
- **Teorema de Bayes**: $P(\theta|D)=\dfrac{P(D|\theta)P(\theta)}{P(D)}$ (posterior ∝ verossimilhança × prior).
- **Distribuições comuns**: Bernoulli, Binomial, Categórica, Uniforme, Normal, Exponencial, Poisson.
- **Lei dos grandes números** e **Teorema Central do Limite**: médias de muitas amostras ficam normais.
- **Monte Carlo**: aproximar esperanças por médias de amostras, $E[f(X)]\approx\frac1N\sum f(x_i)$.

## Onde aparece

Naive Bayes, regressão logística (Bernoulli), modelos generativos, inferência bayesiana, dropout, SGD (amostragem), RL (esperança de retorno).

## Implementação

[`probabilidade.cpp`](probabilidade.cpp): π por Monte Carlo, Bayes num teste diagnóstico (analítico vs simulado — o resultado contraintuitivo de ~16%), TCL e momentos da normal.

```bash
g++ -std=c++17 -O2 probabilidade.cpp -o probabilidade && ./probabilidade
```

## Referências

- Bishop, *Pattern Recognition and Machine Learning*, cap. 1–2.
- Blitzstein & Hwang, *Introduction to Probability* (Harvard Stat 110).
