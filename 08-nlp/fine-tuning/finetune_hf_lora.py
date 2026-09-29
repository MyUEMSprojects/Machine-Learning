"""Fine-tuning de um modelo pré-treinado (DistilBERT) para análise de sentimento, com LoRA via `peft`.

Usa um dataset minúsculo gerado no código (para rodar rápido em CPU). Compara fine-tuning completo e LoRA.

pip install torch transformers peft
"""
import random

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

random.seed(0)
torch.manual_seed(0)
NOME = "distilbert-base-uncased"

pos = ["great", "wonderful", "excellent", "amazing", "fantastic", "brilliant", "delightful", "superb"]
neg = ["terrible", "awful", "horrible", "boring", "dreadful", "disappointing", "dull", "worthless"]
modelos = ["The movie was {}.", "I found the book {}.", "What a {} experience!", "The service here is {}.", "This product looks {}."]


def gera(n):
    dados = []
    for _ in range(n):
        y = random.randint(0, 1)
        dados.append((random.choice(modelos).format(random.choice(pos if y else neg)), y))
    return dados


treino, teste = gera(64), gera(64)
tok = AutoTokenizer.from_pretrained(NOME)


def lote(dados):
    enc = tok([t for t, _ in dados], padding=True, return_tensors="pt")
    return enc, torch.tensor([y for _, y in dados])


def acuracia(m, dados):
    m.eval()
    enc, y = lote(dados)
    with torch.no_grad():
        return (m(**enc).logits.argmax(-1) == y).float().mean().item()


def roda(modo):
    base = AutoModelForSequenceClassification.from_pretrained(NOME, num_labels=2)
    if modo == "lora":
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
        m = get_peft_model(base, cfg)  # congela o modelo; injeta A,B nas projeções de atenção q e v; a cabeça de classificação segue treinável
    else:
        m = base
    treinaveis = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-4 if modo == "lora" else 3e-5)
    for ep in range(1, 6):
        m.train()
        random.shuffle(treino)
        for i in range(0, len(treino), 16):
            enc, y = lote(treino[i : i + 16])
            loss = m(**enc, labels=y).loss
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"{modo:9s} treináveis={treinaveis:>11,} ({100 * treinaveis / total:5.2f}% do total)  acc treino={acuracia(m, treino):.3f}  acc teste={acuracia(m, teste):.3f}")


print(f"modelo: {NOME}")
roda("completo")
roda("lora")
