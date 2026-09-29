# Gradient Boosting

Generalização do boosting para qualquer perda diferenciável: **descida do gradiente no espaço de funções**. É o estado da arte em dados tabulares (XGBoost, LightGBM, CatBoost).

## Algoritmo

1. Inicialize $F_0(x)=\arg\min_c\sum_i L(y_i,c)$ (para MSE: a média de $y$).
2. Para $m=1..M$:
   1. **Pseudo-resíduos**: $r_i=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}$ (para MSE, $r_i=y_i-F_{m-1}(x_i)$, o resíduo).
   2. Ajuste uma árvore rasa $h_m$ aos pares $(x_i,r_i)$.
   3. Atualize $F_m(x)=F_{m-1}(x)+\eta\,h_m(x)$, com **taxa de aprendizado (shrinkage)** $\eta$.

Cada árvore é um passo na direção de $-\nabla L$. Outras perdas: log loss (classificação), Huber (robusta), quantílica.

## Hiperparâmetros importantes

| Parâmetro | Efeito |
|---|---|
| `n_estimators` × `learning_rate` | compromisso: $\eta$ menor exige mais árvores, mas generaliza melhor |
| `max_depth` (3–8) | complexidade de interações |
| `subsample` (<1) | *stochastic* GB: reduz variância |
| `early stopping` | pare quando a validação piora |
| regularização L1/L2 nas folhas | XGBoost/LightGBM |

## Diferenças das implementações modernas

- **XGBoost**: expansão de 2ª ordem (gradiente + Hessiana), regularização explícita.
- **LightGBM**: histogramas, crescimento por folha (*leaf-wise*), muito rápido.
- **CatBoost**: tratamento nativo de categóricas, *ordered boosting*.

## Implementação

- [`gradient_boosting.cpp`](gradient_boosting.cpp) (C++): árvore de regressão + gradient boosting (MSE) do zero; imprime MSE de treino/teste por número de árvores. O erro de teste atinge o mínimo por volta de 50 árvores e depois sobe (overfit) enquanto o de treino segue caindo — é por isso que se usa *early stopping*.
- [`gradient_boosting_sklearn.py`](gradient_boosting_sklearn.py) (Python): scikit-learn, `HistGradientBoosting` e XGBoost opcional.

```bash
g++ -std=c++17 -O2 gradient_boosting.cpp -o gradient_boosting && ./gradient_boosting
python gradient_boosting_sklearn.py
```

## Referências

- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* (2001).
- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System* (2016).
