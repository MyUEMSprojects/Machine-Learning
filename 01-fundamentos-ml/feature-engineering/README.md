# Feature Engineering

"Dados e features determinam o limite do desempenho; algoritmos apenas se aproximam dele."

## Escalonamento

Necessário para algoritmos baseados em distância (k-NN, SVM, k-means), gradiente (regressão, redes) e regularização. Desnecessário para árvores.

- **Padronização (z-score)**: $x'=(x-\mu)/\sigma$.
- **Min-max**: $x'=(x-x_{min})/(x_{max}-x_{min})$ — sensível a outliers.
- **Robust scaler**: usa mediana e IQR.
- **Regra de ouro**: `fit` só no treino; `transform` em treino, validação e teste.

## Variáveis categóricas

- **One-hot**: uma coluna binária por categoria (sem ordem implícita).
- **Ordinal**: inteiros, só se existe ordem real (P < M < G).
- **Target encoding / hashing**: para alta cardinalidade (cuidado com vazamento).

## Transformações e criação de features

- **Log / Box-Cox**: reduz assimetria de variáveis de cauda longa.
- **Polinomiais e interações**: dão não linearidade a modelos lineares.
- **Datas**: dia da semana, mês, sazonalidade cíclica (sin/cos).
- **Texto/imagem**: TF-IDF, embeddings (ver módulos 08 e 09).

## Dados faltantes e outliers

Imputação (média, mediana, kNN, modelo) + coluna indicadora de "faltante"; tratar outliers por *clipping* ou transformação, avaliando se são erros ou sinal.

## Seleção de features

Filtros (correlação, informação mútua), *wrappers* (RFE), *embedded* (L1/Lasso, importância de árvores). Ver [regularização](../regularizacao/).

## Implementação

[`feature_engineering.cpp`](feature_engineering.cpp): `StandardScaler` e `MinMaxScaler` (com `fit`/`transform` separados), one-hot, features polinomiais, `log1p` e imputação pela média.

```bash
g++ -std=c++17 -O2 feature_engineering.cpp -o feature_engineering && ./feature_engineering
```

## Referências

- Zheng & Casari, *Feature Engineering for Machine Learning*.
- Kuhn & Johnson, *Feature Engineering and Selection*.
