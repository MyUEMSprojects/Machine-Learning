# Modelos Probabilísticos

Modelos que representam incerteza explicitamente: misturas, sequências com estados ocultos, grafos de dependência, inferência bayesiana e processos gaussianos.

**Pré-requisitos:** módulos 00 (probabilidade, estatística) e 03.

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`gaussian-mixture/`](gaussian-mixture/) | GMM e o algoritmo EM | ✅ | ✅ |
| [`hmm/`](hmm/) | HMM: forward, Viterbi, Baum–Welch | ✅ | ✅ |
| [`modelos-graficos/`](modelos-graficos/) | redes bayesianas, d-separação, inferência exata | ✅ | — |
| [`inferencia-bayesiana/`](inferencia-bayesiana/) | priors conjugados, MCMC (Metropolis–Hastings), PyMC | ✅ | ✅ |
| [`processos-gaussianos/`](processos-gaussianos/) | regressão com incerteza calibrada | ✅ | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
