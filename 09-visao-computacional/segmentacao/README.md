# Segmentação de Imagens

Classificação **por pixel**: em vez de um rótulo por imagem, uma máscara com um rótulo para cada pixel.

## Tipos

| Tipo | Saída | Exemplo |
|---|---|---|
| **Semântica** | classe de cada pixel (todas as "pessoas" = mesma classe) | estrada, céu, tumor |
| **Instâncias** | separa objetos individuais da mesma classe | pessoa 1, pessoa 2 (Mask R-CNN) |
| **Panóptica** | semântica (*stuff*) + instâncias (*things*) | condução autônoma |
| Promptable | máscara a partir de cliques/caixas/texto | **SAM** (Segment Anything) |

## Métodos clássicos

- **Limiarização** (Otsu: maximiza a variância entre classes do histograma); limiar adaptativo local.
- **Morfologia matemática**: erosão, dilatação, abertura (remove ruído), fechamento (fecha buracos).
- **Componentes conexas**: rotula regiões contíguas → segmentação de instâncias simples.
- **Clustering** ([k-means](../../03-aprendizado-nao-supervisionado/clustering/) em cor/posição), *watershed*, crescimento de regiões, *graph cuts*, contornos ativos.
- Limites: sensíveis a iluminação, textura e ruído; sem noção semântica.

## Deep learning

- **FCN** (Fully Convolutional Network): troca camadas densas por convoluções → saída em mapa.
- **U-Net** (2015): encoder que reduz resolução (contexto) + decoder que a recupera; **skip connections** concatenam mapas do encoder ao decoder para preservar detalhes finos. Padrão em imagens médicas e base de [difusão](../../06-deep-learning-avancado/difusao/).
- **DeepLab** (convoluções dilatadas, ASPP), **Mask R-CNN** (detecção + máscara por instância), **SegFormer/Mask2Former** (Transformers), **SAM**.

### Perdas e métricas

- Perdas: entropia cruzada por pixel (com pesos por classe) + **Dice loss**, $1-\frac{2|P\cap G|}{|P|+|G|}$, que é robusta ao desbalanceamento (o fundo domina).
- Métricas: **IoU** (Jaccard) $=\frac{|P\cap G|}{|P\cup G|}$, **Dice/F1**, mIoU (média por classe), *pixel accuracy* (enganosa com fundo dominante).

## Implementação

- [`segmentacao_classica.cpp`](segmentacao_classica.cpp) (C++): Otsu, erosão/dilatação/abertura, componentes conexas por BFS e IoU/Dice do zero, numa imagem sintética ruidosa.
- [`unet_toy.py`](unet_toy.py) (Python/PyTorch): **U-Net** minúscula com *skip connections*, perda BCE+Dice, em formas sintéticas com iluminação variável; comparada ao Otsu do scikit-image.

```bash
g++ -std=c++17 -O2 segmentacao_classica.cpp -o segmentacao_classica && ./segmentacao_classica
pip install torch scikit-image numpy && python unet_toy.py
```

## Referências

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015).
- Long et al., *Fully Convolutional Networks for Semantic Segmentation* (2015); Kirillov et al., *Segment Anything* (2023).
- Otsu, *A Threshold Selection Method from Gray-Level Histograms* (1979).
