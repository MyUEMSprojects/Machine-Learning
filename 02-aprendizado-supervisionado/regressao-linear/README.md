# Regressão Linear

O modelo supervisionado mais simples: prever um valor contínuo como combinação linear das features.

## Modelo

$$\hat y = w_0 + w_1x_1+\dots+w_dx_d = w^\top x$$

(com $x_0=1$ para o intercepto). Perda: **erro quadrático médio**

$$L(w)=\frac1n\sum_i (y_i-w^\top x_i)^2=\frac1n\lVert y-Xw\rVert^2$$

## Duas formas de resolver

1. **Equações normais** (forma fechada): $\nabla L=0\Rightarrow w=(X^\top X)^{-1}X^\top y$. Custo $O(nd^2+d^3)$; exige $X^\top X$ invertível (sem colinearidade perfeita).
2. **Gradiente descendente**: $w\leftarrow w-\eta\cdot\frac2n X^\top(Xw-y)$. Escala para $d$ grande; use features padronizadas. Como $L$ é convexa, converge ao mínimo global.

## Por que o MSE?

Assumindo $y=w^\top x+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$, maximizar a verossimilhança equivale a minimizar o MSE (ver [estatística](../../00-fundamentos-matematicos/estatistica/)).

## Suposições e limitações

Linearidade, erros independentes com variância constante, pouca multicolinearidade. Sensível a outliers (erro quadrático). Para não linearidade: features polinomiais; para muitos preditores: [regularização](../../01-fundamentos-ml/regularizacao/) (Ridge/Lasso).

## Implementação

- [`regressao_linear.cpp`](regressao_linear.cpp) (C++): equações normais e gradiente descendente do zero; ambos recuperam os coeficientes verdadeiros.
- [`regressao_linear_sklearn.py`](regressao_linear_sklearn.py) (Python): OLS, Ridge e Lasso do scikit-learn.

```bash
g++ -std=c++17 -O2 regressao_linear.cpp -o regressao_linear && ./regressao_linear
python regressao_linear_sklearn.py
```

## Referências

- Hastie et al., *ESL*, cap. 3.
- Curso Stanford CS229, notas sobre regressão linear.
