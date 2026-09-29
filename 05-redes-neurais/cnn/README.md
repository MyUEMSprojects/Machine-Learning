# Redes Neurais Convolucionais (CNN)

Arquitetura para dados com **estrutura espacial** (imagens, áudio, séries): explora **localidade** e **invariância a translação**.

## Camada convolucional

Um filtro (kernel) $K\in\mathbb R^{k\times k}$ desliza pela imagem:

$$(X*K)_{ij}=\sum_{u,v}K_{uv}\,X_{i+u,\,j+v}+b$$

(em DL é uma correlação cruzada). Cada filtro detecta um padrão (borda, textura…) em qualquer posição.

- **Compartilhamento de pesos**: o mesmo filtro em toda a imagem → muito menos parâmetros que uma camada densa.
- **Conexões locais**: cada saída olha só uma janela (*campo receptivo*), que cresce com a profundidade.
- **Hiperparâmetros**: nº de filtros, tamanho $k$, *stride* $s$, *padding* $p$. Tamanho de saída: $\lfloor\frac{n+2p-k}{s}\rfloor+1$.
- **Canais**: entrada $C_{in}\times H\times W$, cada filtro $C_{in}\times k\times k$, saída $C_{out}$ mapas de características.

## Pooling

Reduz resolução e dá invariância a pequenas translações: **max-pooling** (mantém o máximo; o gradiente flui só pelo vencedor) ou average-pooling. Arquiteturas modernas usam também convoluções com *stride*.

## Hierarquia de características

Camadas iniciais: bordas e cores → intermediárias: texturas/partes → finais: objetos. Terminar com camadas densas ou *global average pooling* + softmax.

## Marcos e técnicas

LeNet (1998) → AlexNet (2012, ReLU+dropout+GPU) → VGG (blocos 3×3 empilhados) → **ResNet** (conexões residuais $y=F(x)+x$ permitem 100+ camadas) → EfficientNet. Ferramentas: **BatchNorm**, dropout, *data augmentation*, *transfer learning* (ver [transfer learning](../../06-deep-learning-avancado/transfer-learning/)). Hoje ViTs competem/superam CNNs em muitas tarefas, mas CNNs seguem dominantes com poucos dados. Aplicações em [visão computacional](../../09-visao-computacional/).

## Implementação

- [`cnn.cpp`](cnn.cpp) (C++): convolução, ReLU, max-pool, densa+softmax e **backward de convolução e pooling escritos à mão**; filtro de Sobel; treina para distinguir linhas horizontal/vertical/diagonal em posições aleatórias e compara nº de parâmetros com uma MLP.
- [`cnn_pytorch.py`](cnn_pytorch.py) (Python): CNN com `Conv2d`, `BatchNorm2d`, `MaxPool2d` vs. MLP nos dígitos 8×8.

```bash
g++ -std=c++17 -O2 cnn.cpp -o cnn && ./cnn
python cnn_pytorch.py
```

## Referências

- LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (1998).
- He et al., *Deep Residual Learning for Image Recognition* (2015).
- Stanford CS231n; Dumoulin & Visin, *A guide to convolution arithmetic for deep learning*.
