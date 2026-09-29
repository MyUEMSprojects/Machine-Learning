# Detecção de Objetos

Encontrar **o quê** e **onde**: para cada objeto, uma caixa delimitadora $(x_1,y_1,x_2,y_2)$, uma classe e um score.

## Blocos fundamentais

### IoU (Intersection over Union)
$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}\in[0,1]$$
Mede a sobreposição entre caixas. Uma detecção é considerada correta (TP) se IoU $\ge$ limiar (comum: 0,5) com um objeto real da mesma classe.

### Âncoras (*anchors*)
Detectores prevêm **deslocamentos** relativos a caixas de referência de várias escalas/proporções espalhadas pela imagem. Detectores *anchor-free* (FCOS, CenterNet, YOLOv8) predizem centro/tamanho diretamente.

### NMS (Non-Maximum Suppression)
O detector produz muitas caixas para o mesmo objeto. NMS: ordena por score, mantém a melhor e **suprime** as que a sobrepõem com IoU $>$ limiar (por classe); repete. Limiar baixo remove duplicatas mas pode apagar objetos próximos; Soft-NMS reduz score em vez de remover. Detectores estilo DETR (Transformers, *set prediction*) dispensam NMS.

### Perda
Classificação (cross-entropy/focal loss — lida com o desbalanceamento fundo×objeto) + regressão de caixa (L1/Smooth-L1, GIoU/CIoU).

## Famílias

| Tipo | Ideia | Exemplos |
|---|---|---|
| **Dois estágios** | 1º propõe regiões, 2º classifica/refina | R-CNN → Fast → **Faster R-CNN** (RPN), Mask R-CNN |
| **Um estágio** | prevê tudo em uma passada; rápido | **YOLO**, SSD, RetinaNet |
| **Baseados em Transformers** | consultas de objetos + atenção | DETR, Deformable DETR, RT-DETR |
| **Vocabulário aberto** | detecta classes descritas por texto | Grounding DINO, OWL-ViT |

## Métrica: AP e mAP

Para cada classe: ordene as detecções por score; percorra acumulando TP/FP (uma detecção duplicada de um objeto já casado é FP) para obter a **curva precisão×recall**; **AP** = área sob a curva (com precisão interpolada). **mAP** = média do AP entre as classes; COCO faz também a média sobre limiares de IoU 0,5:0,95 (mAP@[.5:.95]).

## Implementação

- [`iou_nms_map.cpp`](iou_nms_map.cpp) (C++): **IoU, NMS e AP/mAP do zero** num cenário sintético; mostra o mAP subir quando as duplicatas são suprimidas e como o limiar do NMS afeta o resultado.
- [`deteccao_torchvision.py`](deteccao_torchvision.py) (Python): Faster R-CNN (MobileNetV3) pré-treinado no COCO detectando objetos em uma foto; verificação de `box_iou` e `nms` do torchvision.

```bash
g++ -std=c++17 -O2 iou_nms_map.cpp -o iou_nms_map && ./iou_nms_map
pip install torch torchvision scikit-learn pillow && python deteccao_torchvision.py
```

## Referências

- Girshick et al., *Rich feature hierarchies* (R-CNN, 2014); Ren et al., *Faster R-CNN* (2015).
- Redmon et al., *You Only Look Once* (2016) e sucessores; Carion et al., *DETR* (2020).
- Lin et al., *Focal Loss for Dense Object Detection* (2017).
