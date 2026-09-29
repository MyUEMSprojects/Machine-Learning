# Perceptron e Perceptron Multicamadas (MLP)

## Neurônio artificial

$$a=\phi(w^\top x+b)$$

Soma ponderada das entradas + não linearidade $\phi$ (função de ativação).

## Perceptron (Rosenblatt, 1958)

$\hat y=\mathbb 1[w^\top x+b>0]$. Regra de aprendizado: se errar, $w\leftarrow w+(y-\hat y)\,x$. **Teorema de convergência**: se os dados são linearmente separáveis, converge em tempo finito. Se não (ex.: **XOR**), nunca converge — limitação apontada por Minsky & Papert (1969) que provocou o 1º "inverno da IA".

## MLP

Empilhar camadas com ativações não lineares resolve o XOR e, com neurônios suficientes, aproxima qualquer função contínua (**teorema da aproximação universal**):

$$h=\phi(W_1x+b_1),\qquad \hat y=\sigma(W_2h+b_2)$$

Sem não linearidade, várias camadas lineares colapsam numa só.

### Ativações

| Função | Fórmula | Notas |
|---|---|---|
| Sigmoide | $1/(1+e^{-z})$ | saída em (0,1); satura → gradiente some; usar na saída binária |
| tanh | $\tanh z$ | centrada em zero; ainda satura |
| **ReLU** | $\max(0,z)$ | padrão em camadas ocultas; barata; "neurônios mortos" |
| Leaky ReLU / GELU / SiLU | variantes suaves | GELU nos Transformers |
| Softmax | $e^{z_k}/\sum e^{z_j}$ | saída multiclasse |

### Saída e perda

Regressão: linear + MSE. Binária: sigmoide + entropia cruzada binária. Multiclasse: softmax + entropia cruzada.

### Práticas

Inicialização escalada (Xavier/He) para manter a variância dos sinais; normalizar entradas; mini-lotes; regularização (weight decay, dropout, early stopping). O treino usa [backpropagation](../backpropagation/) + [otimizadores](../otimizadores/).

## Implementação

- [`perceptron_mlp.cpp`](perceptron_mlp.cpp) (C++): perceptron (AND ok, XOR falha), MLP com backprop escrito à mão (resolve XOR e "duas espirais").
- [`mlp_pytorch.py`](mlp_pytorch.py) (Python): MLP em PyTorch com dropout, Adam e loop de treino.

```bash
g++ -std=c++17 -O2 perceptron_mlp.cpp -o perceptron_mlp && ./perceptron_mlp
python mlp_pytorch.py
```

## Referências

- Goodfellow, Bengio & Courville, *Deep Learning*, cap. 6.
- Cybenko (1989) e Hornik (1991), teoremas de aproximação universal.
- Nielsen, *Neural Networks and Deep Learning* (online, gratuito).
