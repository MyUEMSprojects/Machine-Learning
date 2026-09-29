# AdaBoost

**Boosting**: constrói o ensemble *sequencialmente*, cada modelo fraco corrigindo os erros dos anteriores (ao contrário do bagging, que treina em paralelo e reduz variância; boosting reduz principalmente **viés**).

## Algoritmo (binário, $y\in\{-1,+1\}$)

1. Pesos iniciais $w_i=1/n$.
2. Para $t=1..T$:
   1. Treine o classificador fraco $h_t$ minimizando o erro **ponderado** $\varepsilon_t=\sum_i w_i\mathbb 1[h_t(x_i)\ne y_i]$.
   2. Peso do voto: $\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ (maior quanto melhor; $\alpha>0$ se $\varepsilon<0.5$).
   3. Atualize: $w_i\leftarrow w_i\,e^{-\alpha_t y_i h_t(x_i)}$ e normalize. Exemplos errados ganham peso.
3. Modelo final: $H(x)=\mathrm{sign}\big(\sum_t\alpha_t h_t(x)\big)$.

Equivale a minimizar a **perda exponencial** $e^{-yF(x)}$ por descida em espaço de funções — visão que leva ao [gradient boosting](../gradient-boosting/).

## Observações

- Learner fraco típico: *stump* (árvore de 1 corte), que sozinho é pouco melhor que o acaso.
- Sensível a ruído/outliers nos rótulos (eles ganham pesos enormes).
- Erro de treino cai exponencialmente; o erro de teste frequentemente continua caindo mesmo após o treino chegar a zero (efeito de margens).

## Implementação

- [`adaboost.cpp`](adaboost.cpp) (C++): AdaBoost com stumps do zero em círculos concêntricos, mostrando a acurácia crescer com as rodadas.
- [`adaboost_sklearn.py`](adaboost_sklearn.py) (Python): `AdaBoostClassifier`.

```bash
g++ -std=c++17 -O2 adaboost.cpp -o adaboost && ./adaboost
python adaboost_sklearn.py
```

## Referências

- Freund & Schapire, *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting* (1997).
- Hastie et al., *ESL*, cap. 10.
