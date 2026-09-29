# Inferência Bayesiana

Trata os **parâmetros como variáveis aleatórias** e usa o teorema de Bayes para atualizar a crença com os dados, obtendo uma distribuição — não um único valor — para $\theta$:

$$\underbrace{p(\theta\mid D)}_{\text{posterior}}=\frac{\overbrace{p(D\mid\theta)}^{\text{verossimilhança}}\;\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(D)}_{\text{evidência}}},\qquad p(D)=\int p(D|\theta)p(\theta)\,d\theta$$

## MLE × MAP × Bayes completo

| | Resultado | Observação |
|---|---|---|
| MLE | $\arg\max p(D\mid\theta)$ | ignora prior |
| MAP | $\arg\max p(\theta\mid D)$ | = MLE + regularização (prior gaussiano = L2, Laplace = L1) |
| Bayes | distribuição $p(\theta\mid D)$ | quantifica incerteza; predição = média sobre o posterior |

**Distribuição preditiva**: $p(x^*\mid D)=\int p(x^*\mid\theta)\,p(\theta\mid D)\,d\theta$.

## Priors conjugados

Quando prior e posterior pertencem à mesma família há forma fechada. Ex.: Beta–Binomial: prior $\mathrm{Beta}(a,b)$ + $k$ sucessos em $n$ ⇒ posterior $\mathrm{Beta}(a+k,\,b+n-k)$ ($a,b$ agem como "pseudo-contagens"). Também: Normal–Normal, Gamma–Poisson, Dirichlet–Multinomial.

## Quando não há forma fechada

- **MCMC**: constrói uma cadeia de Markov cuja distribuição estacionária é o posterior.
  - **Metropolis–Hastings**: propõe $\theta'\sim q(\cdot|\theta)$ e aceita com prob. $\min\big(1,\frac{p(\theta'|D)q(\theta|\theta')}{p(\theta|D)q(\theta'|\theta)}\big)$. Só precisa do posterior **até a constante** $p(D)$.
  - **Gibbs**, **Hamiltonian Monte Carlo / NUTS** (usa gradientes; padrão do Stan e PyMC).
  - Cuidados: *burn-in*, autocorrelação, diagnósticos ($\hat R$, ESS).
- **Inferência variacional**: aproxima o posterior por uma família simples $q_\phi$ minimizando $\mathrm{KL}(q\|p)$ (maximiza o ELBO) — mais rápida, mais enviesada; base dos VAEs.

## Implementação

- [`inferencia_bayesiana.cpp`](inferencia_bayesiana.cpp) (C++): atualização Beta–Binomial e MCMC Metropolis–Hastings do zero com prior de Laplace.
- [`inferencia_bayesiana_python.py`](inferencia_bayesiana_python.py) (Python): regressão linear bayesiana com **PyMC** (NUTS) — requer `pip install pymc arviz`.

```bash
g++ -std=c++17 -O2 inferencia_bayesiana.cpp -o inferencia_bayesiana && ./inferencia_bayesiana
python inferencia_bayesiana_python.py
```

## Referências

- Gelman et al., *Bayesian Data Analysis*.
- McElreath, *Statistical Rethinking*.
- Murphy, *Probabilistic ML: Advanced Topics*.
