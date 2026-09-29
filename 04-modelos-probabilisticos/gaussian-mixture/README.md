# Misturas de Gaussianas (GMM) e o algoritmo EM

Modelo generativo de densidade: os dados vêm de $K$ gaussianas, cada ponto pertence a uma delas (variável latente $z$ não observada).

$$p(x)=\sum_{k=1}^K\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k),\qquad \pi_k\ge0,\ \sum_k\pi_k=1$$

Uso: clustering **suave** (probabilidades de pertencer a cada grupo), estimação de densidade, detecção de anomalias, base de HMMs com emissões contínuas. Generaliza o k-means (que é o caso $\Sigma_k=\sigma^2I$ com $\sigma\to0$ e atribuições duras).

## Algoritmo EM (Expectation–Maximization)

Maximizar $\sum_i\log\sum_k\pi_k\mathcal N(x_i|\mu_k,\Sigma_k)$ diretamente é difícil (log da soma). O EM alterna:

- **E**: responsabilidades $r_{ik}=P(z_i=k\mid x_i)=\dfrac{\pi_k\mathcal N(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i|\mu_j,\Sigma_j)}$
- **M**: com $N_k=\sum_i r_{ik}$:
  $\pi_k=\frac{N_k}{n},\quad \mu_k=\frac1{N_k}\sum_i r_{ik}x_i,\quad \Sigma_k=\frac1{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$

Cada iteração **nunca diminui** a log-verossimilhança; converge a um máximo **local** → use várias inicializações (ou k-means para inicializar).

## Cuidados

- **Singularidade**: uma gaussiana pode colapsar sobre um único ponto ($\det\Sigma\to0$, verossimilhança $\to\infty$); adicione regularização $\epsilon I$.
- **Escolha de $K$**: BIC $=p\ln n-2\ln\hat L$ (ou AIC); a verossimilhança sozinha sempre cresce com $K$.
- Tipos de covariância: esférica, diagonal, *tied*, completa (mais parâmetros → mais flexível).

O EM é um esquema geral para modelos com variáveis latentes (HMM/Baum–Welch, análise fatorial, etc.); pode ser visto como maximização de um limite inferior (ELBO) — base dos VAEs.

## Implementação

- [`gmm_em.cpp`](gmm_em.cpp) (C++): GMM 2D com covariância completa via EM, reinicializações e seleção de $K$ por BIC.
- [`gmm_sklearn.py`](gmm_sklearn.py) (Python): `GaussianMixture`, tipos de covariância, BIC, `predict_proba`.

```bash
g++ -std=c++17 -O2 gmm_em.cpp -o gmm_em && ./gmm_em
python gmm_sklearn.py
```

## Referências

- Bishop, *PRML*, cap. 9.
- Dempster, Laird & Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977).
