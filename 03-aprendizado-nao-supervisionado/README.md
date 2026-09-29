# Aprendizado Não Supervisionado

Encontrar estrutura em dados sem rótulos: grupos, dimensões principais, anomalias e regras de associação.

**Pré-requisitos:** módulos 00 e 01.

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`clustering/`](clustering/) | k-means (++), hierárquico, DBSCAN | ✅ | ✅ |
| [`reducao-dimensionalidade/`](reducao-dimensionalidade/) | PCA do zero; Kernel PCA, Isomap, t-SNE | ✅ | ✅ |
| [`deteccao-anomalias/`](deteccao-anomalias/) | Mahalanobis, Isolation Forest, LOF, One-Class SVM | ✅ | ✅ |
| [`regras-de-associacao/`](regras-de-associacao/) | Apriori: suporte, confiança, lift | ✅ | — |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
