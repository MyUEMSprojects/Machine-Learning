# Regularização

Restringe a complexidade do modelo para reduzir variância (overfitting), aceitando um pouco de viés.

$$\min_\theta\; L(\theta)+\lambda\,\Omega(\theta)$$

## Penalidades

| Nome | $\Omega(\theta)$ | Efeito | Visão bayesiana |
|---|---|---|---|
| **Ridge (L2)** | $\lVert\theta\rVert_2^2$ | encolhe pesos suavemente, nunca zera; lida com colinearidade | prior gaussiano |
| **Lasso (L1)** | $\lVert\theta\rVert_1$ | zera pesos → **seleção de features** esparsa | prior de Laplace |
| **Elastic Net** | $\alpha L_1+(1-\alpha)L_2$ | combina ambos; estável com features correlacionadas | — |

Geometria: a região $\lVert\theta\rVert_1\le t$ é um losango com "quinas" nos eixos, onde a solução tende a cair → zeros. A bola L2 é redonda, sem quinas.

Forma fechada do Ridge: $\hat\theta=(X^\top X+\lambda I)^{-1}X^\top y$ (sempre invertível para $\lambda>0$).
Lasso não tem forma fechada: usa-se descida por coordenadas com *soft-thresholding* $S_\lambda(z)=\mathrm{sign}(z)\max(|z|-\lambda,0)$.

## Outras formas de regularizar

- **Early stopping**: parar quando o erro de validação sobe.
- **Dropout**, **weight decay**, **batch norm**, **data augmentation** (redes neurais).
- **Poda** de árvores, limite de profundidade, `min_samples_leaf`.
- **Mais dados**.

$\lambda$ é hiperparâmetro: escolha por [validação cruzada](../validacao-cruzada/). Padronize as features antes (a penalidade depende da escala).

## Implementação

[`regularizacao.cpp`](regularizacao.cpp): Ridge (forma fechada) e Lasso (descida por coordenadas) do zero em dados com 3 features relevantes e 17 irrelevantes. O Lasso recupera a esparsidade.

```bash
g++ -std=c++17 -O2 regularizacao.cpp -o regularizacao && ./regularizacao
```

## Referências

- Tibshirani, *Regression Shrinkage and Selection via the Lasso* (1996).
- Hastie et al., *ESL*, cap. 3.4.
