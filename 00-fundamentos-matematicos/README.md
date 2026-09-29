# Fundamentos Matemáticos

A base matemática usada em todo o restante: álgebra linear, cálculo, probabilidade, estatística, otimização e teoria da informação. Cada tópico traz o conceito e uma implementação mínima em C++ para ver a matemática funcionando.

**Pré-requisitos:** nenhum (ensino médio + curiosidade).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`algebra-linear/`](algebra-linear/) | vetores, matrizes, sistemas lineares, autovalores/autovetores | ✅ | — |
| [`calculo/`](calculo/) | derivadas, gradientes, regra da cadeia, diferenciação automática | ✅ | — |
| [`probabilidade/`](probabilidade/) | variáveis aleatórias, Bayes, Monte Carlo, TCL | ✅ | — |
| [`estatistica/`](estatistica/) | estimadores, MLE, intervalos de confiança, bootstrap | ✅ | — |
| [`otimizacao/`](otimizacao/) | gradiente descendente, momentum, Adam, Newton | ✅ | — |
| [`teoria-da-informacao/`](teoria-da-informacao/) | entropia, entropia cruzada, KL, informação mútua | ✅ | — |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
