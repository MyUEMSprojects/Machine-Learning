# Machine Learning

Repositório de **estudo** de Machine Learning: dos algoritmos fundamentais aos tópicos avançados, com explicações conceituais em Markdown (em português) e **código executável** para cada tópico.

- **C++17** — algoritmos "puros" implementados **do zero** (só biblioteca padrão): regressão, árvores, SVM, k-means, PCA, EM, HMM, backprop com autodiff, CNN, LSTM com BPTT, word2vec, DBSCAN, GP, FedAvg…
- **Python** — bibliotecas prontas (scikit-learn, PyTorch, Hugging Face, statsmodels…) e tópicos que seriam impraticáveis em C++ (VAE, GAN, difusão, GNN, mini-GPT, LoRA, PPO/DQN…).
- **Markdown** — cada tópico tem um `README.md` com intuição, matemática, prós/contras e referências.

## Trilha de estudo

| # | Módulo | Conteúdo |
|---|--------|----------|
| 00 | [fundamentos-matematicos](00-fundamentos-matematicos/) | álgebra linear, cálculo/autodiff, probabilidade, estatística, otimização, teoria da informação |
| 01 | [fundamentos-ml](01-fundamentos-ml/) | conceitos, viés-variância, métricas, validação cruzada, feature engineering, regularização |
| 02 | [aprendizado-supervisionado](02-aprendizado-supervisionado/) | regressão linear/logística, k-NN, Naive Bayes, árvores, SVM, ensembles (RF, AdaBoost, GBM) |
| 03 | [aprendizado-nao-supervisionado](03-aprendizado-nao-supervisionado/) | k-means, hierárquico, DBSCAN, PCA/t-SNE, detecção de anomalias, Apriori |
| 04 | [modelos-probabilisticos](04-modelos-probabilisticos/) | GMM/EM, HMM, redes bayesianas, inferência bayesiana/MCMC, processos gaussianos |
| 05 | [redes-neurais](05-redes-neurais/) | perceptron/MLP, backprop, otimizadores, CNN, RNN/LSTM, atenção e Transformers |
| 06 | [deep-learning-avancado](06-deep-learning-avancado/) | VAE, GANs, difusão, auto-supervisionado, transfer learning, GNNs |
| 07 | [aprendizado-por-reforco](07-aprendizado-por-reforco/) | bandits, MDP/Bellman, Q-learning, policy gradient/PPO, DQN |
| 08 | [nlp](08-nlp/) | TF-IDF/BPE, embeddings, modelos de linguagem, LLMs, fine-tuning/LoRA |
| 09 | [visao-computacional](09-visao-computacional/) | classificação, detecção de objetos, segmentação |
| 10 | [topicos-avancados](10-topicos-avancados/) | séries temporais, XAI, meta-learning, aprendizado federado, otimização bayesiana, MLOps |

Cada módulo tem um `README.md` com a lista de tópicos; dentro deles, subtópicos podem ter seus próprios subtópicos (ex.: [ensembles](02-aprendizado-supervisionado/ensembles/)).

## Como rodar

**C++** (arquivo único, sem dependências):

```bash
cd 02-aprendizado-supervisionado/svm
g++ -std=c++17 -O2 svm.cpp -o svm && ./svm
```

**Python:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # ou só o que o docstring do script indicar
python 02-aprendizado-supervisionado/svm/svm_sklearn.py
```

Alguns scripts baixam dados/modelos na primeira execução (MNIST, pesos do torchvision, DistilBERT/DistilGPT-2). Todos rodam em CPU.

## Estrutura de um tópico

```text
NN-modulo/topico/
├── README.md          # explicação: intuição, matemática, prós/contras, referências
├── topico.cpp         # implementação do zero (C++), com saída que valida o resultado
└── topico_lib.py      # uso de bibliotecas (Python)
```

Para adicionar um novo tópico, copie [`_template/`](_template/) e veja as convenções em [`CLAUDE.md`](CLAUDE.md).

## Referências gerais

- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* · Bishop, *Pattern Recognition and Machine Learning* · Murphy, *Probabilistic Machine Learning*
- Goodfellow, Bengio & Courville, *Deep Learning* · Sutton & Barto, *Reinforcement Learning: An Introduction*
- Jurafsky & Martin, *Speech and Language Processing* · Stanford CS229 / CS231n / CS224n
