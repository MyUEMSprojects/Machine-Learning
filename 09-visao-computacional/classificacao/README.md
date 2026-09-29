# Classificação de Imagens

Atribuir um rótulo a uma imagem inteira (dígito, espécie, doença numa radiografia…). Foi a tarefa que detonou a revolução do deep learning (AlexNet no ImageNet, 2012).

## Representação de imagens

Imagem = tensor $H\times W\times C$ (pixels 0–255, canais RGB). Pré-processamento: redimensionar/recortar, converter para `float`, **normalizar** (média/desvio do dataset de pré-treino).

## Abordagem clássica

Extrair **características manuais** e usar um classificador simples ([SVM](../../02-aprendizado-supervisionado/svm/), [k-NN](../../02-aprendizado-supervisionado/knn/), [árvores](../../02-aprendizado-supervisionado/ensembles/)):

- **HOG**: histogramas da orientação dos gradientes por célula → captura forma/contorno; robusto a iluminação e pequenas translações.
- SIFT/SURF (pontos-chave invariantes a escala), LBP (textura), histogramas de cor.
- Limite: as características são projetadas por humanos.

## Deep learning

[CNNs](../../05-redes-neurais/cnn/) aprendem as características ponta-a-ponta (bordas → texturas → partes → objetos). Linha do tempo: LeNet → AlexNet → VGG → Inception → **ResNet** (residuais) → EfficientNet/ConvNeXt → **ViT** (Transformers com *patches*).

### Receita moderna

1. Comece por um **modelo pré-treinado** na ImageNet e faça [transfer learning](../../06-deep-learning-avancado/transfer-learning/) (extrator congelado → fine-tuning).
2. **Data augmentation** (flip, recorte aleatório, rotação, jitter de cor, Mixup/CutMix) reduz overfitting.
3. Regularização (dropout, weight decay, label smoothing), BatchNorm, AdamW/SGD com *cosine schedule*.
4. Métricas: acurácia top-1/top-5, matriz de confusão, F1 por classe ([métricas](../../01-fundamentos-ml/metricas-avaliacao/)).

Cuidados: vazamento entre treino/teste (imagens quase duplicadas), viés do dataset, mudança de distribuição (câmeras/hospitais diferentes), calibração das probabilidades.

## Implementação

- [`hog.cpp`](hog.cpp) (C++): **HOG do zero** + k-NN vs. pixels brutos em formas sintéticas com deslocamentos crescentes.
- [`classificacao_cnn.py`](classificacao_cnn.py) (Python/PyTorch): CNN no MNIST (~95% em 3 épocas com subconjunto de 20 mil imagens; passa de 98% com mais épocas e dados) e inferência com **ResNet-18 pré-treinada** (torchvision) em fotos de exemplo.

```bash
g++ -std=c++17 -O2 hog.cpp -o hog && ./hog
pip install torch torchvision scikit-learn pillow && python classificacao_cnn.py
```

## Referências

- Krizhevsky et al., *ImageNet Classification with Deep CNNs* (2012); He et al., *ResNet* (2015); Dosovitskiy et al., *ViT* (2020).
- Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection* (2005).
- Stanford CS231n.
