# Machine Learning

Repositório de estudo de Machine Learning: dos algoritmos fundamentais aos tópicos avançados, com explicações conceituais em Markdown e código.

## Convenções

- **C++** (C++17+): algoritmos "puros", implementados do zero (ex.: regressão linear, k-NN, árvores, backprop).
- **Python**: uso de bibliotecas prontas (scikit-learn, PyTorch, etc.) em tópicos difíceis de implementar em C++ (ex.: transformers, difusão, RL profundo).
- **Markdown**: explicações de conceitos (`.md` junto ao código ou em `README.md` do tópico).
- Cada módulo tem seu `README.md` com o índice de subtópicos; subtópicos podem ter subtópicos.
- Nomes de diretórios em `kebab-case`, módulos numerados pela ordem sugerida de estudo.
- Estrutura sugerida de um subtópico: veja [`_template/`](_template/).

## Módulos

| # | Módulo | Conteúdo |
|---|--------|----------|
| 00 | [fundamentos-matematicos](00-fundamentos-matematicos/) | álgebra linear, cálculo, probabilidade, estatística, otimização |
| 01 | [fundamentos-ml](01-fundamentos-ml/) | bias/variância, métricas, validação, feature engineering |
| 02 | [aprendizado-supervisionado](02-aprendizado-supervisionado/) | regressão, classificação, árvores, SVM, ensembles |
| 03 | [aprendizado-nao-supervisionado](03-aprendizado-nao-supervisionado/) | clustering, PCA, anomalias |
| 04 | [modelos-probabilisticos](04-modelos-probabilisticos/) | GMM, HMM, modelos gráficos, GPs |
| 05 | [redes-neurais](05-redes-neurais/) | MLP, backprop, CNN, RNN, Transformers |
| 06 | [deep-learning-avancado](06-deep-learning-avancado/) | VAE, GANs, difusão, GNNs |
| 07 | [aprendizado-por-reforco](07-aprendizado-por-reforco/) | MDP, Q-learning, policy gradient |
| 08 | [nlp](08-nlp/) | embeddings, modelos de linguagem, LLMs |
| 09 | [visao-computacional](09-visao-computacional/) | classificação, detecção, segmentação |
| 10 | [topicos-avancados](10-topicos-avancados/) | séries temporais, XAI, meta-learning, MLOps |
