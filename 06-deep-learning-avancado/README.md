# Deep Learning Avançado

Modelos generativos, aprendizado de representações sem rótulos e dados estruturados. Aqui as implementações são em Python/PyTorch (o custo de reimplementar tudo em C++ não compensa).

**Pré-requisitos:** módulo 05.

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`autoencoders-vae/`](autoencoders-vae/) | AE, VAE, ELBO, reparametrização | — | ✅ |
| [`gans/`](gans/) | jogo gerador × discriminador, colapso de modos | — | ✅ |
| [`difusao/`](difusao/) | DDPM: ruído direto e reverso, predição de ruído | — | ✅ |
| [`self-supervised/`](self-supervised/) | contrastivo (SimCLR/InfoNCE), preditivo, BYOL | — | ✅ |
| [`transfer-learning/`](transfer-learning/) | extrator congelado, fine-tuning, LR diferencial | — | ✅ |
| [`grafos-gnn/`](grafos-gnn/) | GCN, message passing, classificação de nós | — | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
