# Visão Computacional

Entender imagens: classificar, localizar objetos e segmentar pixels — de HOG/Otsu clássicos a CNNs, R-CNN e U-Net.

**Pré-requisitos:** módulo 05 (CNNs).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`classificacao/`](classificacao/) | HOG + k-NN do zero; CNN no MNIST; ResNet pré-treinada | ✅ | ✅ |
| [`deteccao-objetos/`](deteccao-objetos/) | IoU, NMS, mAP do zero; Faster R-CNN | ✅ | ✅ |
| [`segmentacao/`](segmentacao/) | Otsu/morfologia/componentes conexas; U-Net | ✅ | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
