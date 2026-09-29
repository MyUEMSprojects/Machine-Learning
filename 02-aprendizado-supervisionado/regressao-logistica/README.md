# Regressão Logística

Apesar do nome, é um algoritmo de **classificação**: modela a probabilidade da classe com uma função linear passada por uma sigmoide.

## Modelo (binário)

$$P(y=1\mid x)=\sigma(w^\top x)=\frac{1}{1+e^{-w^\top x}}$$

A fronteira de decisão ($P=0.5$) é o hiperplano $w^\top x=0$ — classificador **linear**. $w^\top x$ é o *log-odds*: $\log\frac{p}{1-p}$.

## Perda: entropia cruzada (log loss)

$$L(w)=-\frac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$$

É a log-verossimilhança negativa de uma Bernoulli (ver [teoria da informação](../../00-fundamentos-matematicos/teoria-da-informacao/)). Convexa; sem forma fechada; otimiza-se por gradiente descendente (ou Newton/IRLS, L-BFGS). Gradiente elegante:

$$\nabla_w L=\frac1n X^\top(p-y)$$

## Multiclasse: softmax

$$P(y=k\mid x)=\frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}},\qquad \nabla_{w_k}L=\frac1n\sum_i (p_{ik}-\mathbb 1[y_i=k])\,x_i$$

Subtraia o máximo dos logits antes do `exp` (estabilidade numérica). A camada final de quase toda rede classificadora é exatamente isto.

## Prática

- Padronize as features; use regularização L2 (padrão) ou L1.
- Desbalanceamento: `class_weight`, ajuste do limiar de decisão.
- Coeficientes são interpretáveis (razão de chances $e^{w_j}$).

## Implementação

- [`regressao_logistica.cpp`](regressao_logistica.cpp) (C++): binária e softmax do zero com gradiente descendente.
- [`regressao_logistica_sklearn.py`](regressao_logistica_sklearn.py) (Python): efeito de `C` (regularização) por validação cruzada.

```bash
g++ -std=c++17 -O2 regressao_logistica.cpp -o regressao_logistica && ./regressao_logistica
python regressao_logistica_sklearn.py
```

## Referências

- Bishop, *PRML*, seção 4.3.
- Murphy, *Probabilistic Machine Learning: An Introduction*, cap. 10.
