# Fundamentos de Machine Learning

Conceitos gerais que valem para qualquer algoritmo: como formular o problema, medir, validar, evitar overfitting e preparar dados.

**Pré-requisitos:** módulo 00 (principalmente probabilidade e otimização).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`conceitos-gerais/`](conceitos-gerais/) | tipos de aprendizado, pipeline, glossário, vazamento de dados | — | ✅ |
| [`bias-variancia/`](bias-variancia/) | decomposição do erro, underfitting × overfitting | ✅ | — |
| [`metricas-avaliacao/`](metricas-avaliacao/) | regressão e classificação (F1, ROC AUC, log loss…) | ✅ | — |
| [`validacao-cruzada/`](validacao-cruzada/) | hold-out, k-fold, séries temporais, CV aninhada | ✅ | — |
| [`feature-engineering/`](feature-engineering/) | escalonamento, codificação, imputação, seleção | ✅ | — |
| [`regularizacao/`](regularizacao/) | Ridge, Lasso, Elastic Net, early stopping | ✅ | — |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
