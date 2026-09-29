# Processos Gaussianos (GP)

Um GP é uma distribuição **sobre funções**: qualquer conjunto finito de valores $f(x_1),\dots,f(x_n)$ é conjuntamente gaussiano. É definido por uma média (geralmente 0) e um **kernel** $k(x,x')$ que codifica o que significa "duas entradas próximas terem saídas parecidas".

$$f\sim\mathcal{GP}(0,k),\qquad k_{\text{RBF}}(x,x')=\sigma_f^2\exp\!\Big(-\frac{(x-x')^2}{2\ell^2}\Big)$$

## Regressão com GP

Com observações $y=f(X)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma_n^2)$ e $K=k(X,X)+\sigma_n^2I$, a predição em $x_*$ é gaussiana (forma fechada!):

$$\mu_*=k_*^\top K^{-1}y,\qquad \sigma_*^2=k(x_*,x_*)-k_*^\top K^{-1}k_*$$

O ganho principal é a **incerteza calibrada**: baixa perto dos dados, alta onde não há dados (buracos/extrapolação).

## Hiperparâmetros

Comprimento de escala $\ell$ (suavidade), variância do sinal $\sigma_f^2$, ruído $\sigma_n^2$ — escolhidos maximizando a **log-verossimilhança marginal**

$$\log p(y|X)=-\tfrac12y^\top K^{-1}y-\tfrac12\log|K|-\tfrac n2\log2\pi$$

que já equilibra ajuste × complexidade (navalha de Occam automática). $\ell$ muito pequeno → overfit; muito grande → underfit.

## Kernels

RBF (infinitamente suave), Matérn (rugosidade ajustável), periódico, linear; kernels somam/multiplicam para compor estrutura (tendência + sazonalidade).

## Prós e contras

- ✅ Poucos dados, incerteza nativa, interpretável via kernel; base da **otimização bayesiana**.
- ❌ Custo $O(n^3)$ tempo e $O(n^2)$ memória (Cholesky) → aproximações esparsas/variacionais para $n$ grande; escolha do kernel importa.

## Implementação

- [`gp_regressao.cpp`](gp_regressao.cpp) (C++): GP do zero com Cholesky; média/variância preditivas e seleção de $\ell$ pela verossimilhança marginal, mostrando a incerteza aumentar no "buraco" e na extrapolação.
- [`gp_sklearn.py`](gp_sklearn.py) (Python): `GaussianProcessRegressor` com kernel composto.

```bash
g++ -std=c++17 -O2 gp_regressao.cpp -o gp_regressao && ./gp_regressao
python gp_sklearn.py
```

## Referências

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (gratuito em gaussianprocess.org/gpml).
- Görtler et al., *A Visual Exploration of Gaussian Processes* (Distill).
