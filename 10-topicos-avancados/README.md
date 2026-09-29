# Tópicos Avançados

Áreas de engenharia e pesquisa que complementam o núcleo: séries temporais, interpretabilidade, meta-learning, privacidade, otimização de caixa-preta e operação em produção.

**Pré-requisitos:** módulos 01–05.

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`series-temporais/`](series-temporais/) | ACF, Holt–Winters, AR, SARIMA, validação walk-forward | ✅ | ✅ |
| [`interpretabilidade-xai/`](interpretabilidade-xai/) | permutação, PDP, Shapley exato, LIME, SHAP | ✅ | ✅ |
| [`meta-learning/`](meta-learning/) | MAML e Reptile em senoides few-shot | — | ✅ |
| [`aprendizado-federado/`](aprendizado-federado/) | FedAvg, dados não-IID, privacidade diferencial | ✅ | ✅ |
| [`otimizacao-bayesiana/`](otimizacao-bayesiana/) | GP + Expected Improvement, Optuna/TPE | ✅ | ✅ |
| [`mlops/`](mlops/) | versionamento, portões de qualidade, deriva (PSI/KS), API | ✅ | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
