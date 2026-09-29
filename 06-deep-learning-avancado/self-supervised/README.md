# Aprendizado Auto-supervisionado (SSL)

Aprender representações úteis de **dados sem rótulos**, criando a supervisão a partir dos próprios dados (*pretext task*). Depois, as representações são reaproveitadas em tarefas com poucos rótulos. É a base de LLMs e modelos de fundação.

## Famílias

### 1. Preditivo / generativo (mascarar e prever)
- **Modelagem de linguagem causal** (GPT): prever o próximo token.
- **Masked LM** (BERT): prever tokens mascarados.
- **MAE** (visão): reconstruir patches mascarados da imagem.

### 2. Contrastivo
Aproximar **pares positivos** (duas visões aumentadas da mesma amostra) e afastar **negativos**. Perda **InfoNCE / NT-Xent** para embeddings normalizados:

$$\ell_i=-\log\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i,z_k)/\tau)}$$

com $\tau$ (temperatura) pequeno. Maximiza um limite inferior da informação mútua entre visões. Exemplos: **SimCLR** (negativos = outros exemplos do lote → lotes grandes), **MoCo** (fila de negativos + encoder momentum), **CLIP** (contraste imagem↔texto).

### 3. Sem negativos (auto-destilação)
**BYOL**, **DINO**: um encoder "aluno" prevê a saída de um "professor" (média móvel do aluno); evitam colapso com assimetria/centralização.

## Detalhes que importam

- **Aumentações** definem a invariância aprendida (recorte, cor, rotação, ruído…) — a escolha é o "conhecimento prévio" do método.
- **Cabeça de projeção**: aplica-se a perda após uma MLP pequena e descarta-se para uso downstream.
- **Avaliação**: *linear probing* (classificador linear sobre features congeladas), *few-shot*, fine-tuning.
- **Colapso**: todas as saídas iguais minimizam a distância dos positivos — negativos, assimetria ou regularização de variância (VICReg) evitam.

## Implementação

[`simclr_digits.py`](simclr_digits.py) (Python/PyTorch): SimCLR minúsculo nos dígitos 8×8 com aumentações próprias, perda NT-Xent e avaliação por sonda linear com 10 e 50 rótulos, comparando pixels brutos × encoder aleatório × encoder pré-treinado sem rótulos (ganho modesto mas consistente — em imagens minúsculas 8×8 há pouco a aprender além dos pixels).

```bash
python simclr_digits.py
```

## Referências

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, 2020).
- Radford et al., *CLIP* (2021); He et al., *Masked Autoencoders Are Scalable Vision Learners* (2021).
- Grill et al., *BYOL* (2020); Caron et al., *DINO* (2021).
