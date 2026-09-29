# Aprendizado Supervisionado

Algoritmos clássicos de regressão e classificação, do modelo linear aos ensembles de árvores (o estado da arte em dados tabulares).

**Pré-requisitos:** módulos 00 e 01.

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`regressao-linear/`](regressao-linear/) | mínimos quadrados: equações normais e gradiente descendente | ✅ | ✅ |
| [`regressao-logistica/`](regressao-logistica/) | classificação binária e softmax, entropia cruzada | ✅ | ✅ |
| [`knn/`](knn/) | vizinhos mais próximos (classificação e regressão) | ✅ | ✅ |
| [`naive-bayes/`](naive-bayes/) | Gaussiano e Multinomial, suavização de Laplace | ✅ | ✅ |
| [`arvores-de-decisao/`](arvores-de-decisao/) | CART, Gini/entropia, poda | ✅ | ✅ |
| [`svm/`](svm/) | margem máxima, perda hinge, truque do kernel (Pegasos) | ✅ | ✅ |
| [`ensembles/`](ensembles/) | bagging/Random Forest, AdaBoost, Gradient Boosting | ✅ | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
