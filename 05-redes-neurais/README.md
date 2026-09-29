# Redes Neurais

Do neurônio à arquitetura Transformer: perceptron, MLP, backpropagation, otimizadores, CNNs, RNNs/LSTMs e atenção.

**Pré-requisitos:** módulos 00 (cálculo, álgebra linear, otimização) e 02 (regressão logística).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`perceptron-mlp/`](perceptron-mlp/) | neurônio, ativações, aproximação universal | ✅ | ✅ |
| [`backpropagation/`](backpropagation/) | regra da cadeia, autodiff reverso (mini-micrograd em C++) | ✅ | ✅ |
| [`otimizadores/`](otimizadores/) | SGD, momentum, RMSprop, Adam(W), schedulers | ✅ | ✅ |
| [`cnn/`](cnn/) | convolução, pooling, backward escrito à mão | ✅ | ✅ |
| [`rnn-lstm/`](rnn-lstm/) | BPTT, gradientes que somem, portões da LSTM | ✅ | ✅ |
| [`attention-transformers/`](attention-transformers/) | atenção, multi-head, posição, bloco Transformer, mini encoder | ✅ | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
