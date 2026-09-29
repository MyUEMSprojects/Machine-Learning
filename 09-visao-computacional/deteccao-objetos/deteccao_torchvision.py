"""Detecção de objetos com um Faster R-CNN pré-treinado (COCO) do torchvision e verificação de NMS/IoU.

pip install torch torchvision scikit-learn pillow   (baixa os pesos, ~75 MB, na 1ª execução)
"""
import numpy as np
import torch
import torchvision
from sklearn.datasets import load_sample_image
from torchvision.ops import box_iou, nms

pesos = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
modelo = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=pesos, box_score_thresh=0.3).eval()
categorias = pesos.meta["categories"]

def carrega_imagem():
    """Foto do COCO com gatos e controles remotos (baixada); se offline, usa uma imagem do scikit-learn (sem objetos COCO)."""
    import urllib.request
    from pathlib import Path

    from PIL import Image

    destino = Path.home() / ".cache" / "coco_000000039769.jpg"
    try:
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve("http://images.cocodataset.org/val2017/000000039769.jpg", destino)
        return torch.tensor(np.array(Image.open(destino).convert("RGB")))
    except Exception:
        print("(sem internet: usando china.jpg, que provavelmente não tem objetos das classes COCO)")
        return torch.tensor(load_sample_image("china.jpg"))


img = carrega_imagem().permute(2, 0, 1).float() / 255  # (3, H, W) em [0,1]
with torch.no_grad():
    saida = modelo([img])[0]  # dict: boxes (x1,y1,x2,y2), labels, scores — o NMS já é aplicado internamente

print(f"imagem {tuple(img.shape)} -> {len(saida['boxes'])} detecções com score >= 0.3")
for b, l, s in zip(saida["boxes"], saida["labels"], saida["scores"]):
    print(f"  {categorias[l]:15s} score={s:.2f}  caixa={[round(v) for v in b.tolist()]}")

# NMS "na mão" com torchvision: duas caixas quase iguais + uma distante
caixas = torch.tensor([[10.0, 10, 110, 110], [12, 12, 112, 112], [200, 200, 300, 300]])
scores = torch.tensor([0.9, 0.8, 0.7])
print("\nIoU entre caixas:\n", box_iou(caixas, caixas).round(decimals=2))
print("índices mantidos pelo NMS (limiar 0.5):", nms(caixas, scores, 0.5).tolist(), "-> a 2ª (duplicata da 1ª) foi suprimida")
