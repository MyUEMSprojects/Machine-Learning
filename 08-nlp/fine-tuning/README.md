# Fine-tuning de Modelos de Linguagem

Adaptar um modelo pré-treinado a uma tarefa/domínio/estilo específico (ver [transfer learning](../../06-deep-learning-avancado/transfer-learning/)). Em NLP, é o passo entre o modelo base e um assistente útil.

## Tipos

| Tipo | Dados | Objetivo |
|---|---|---|
| **Classificação/extração** (BERT-like) | texto + rótulo | trocar a cabeça e treinar (sentimento, NER, QA) |
| **SFT** (*supervised fine-tuning*) | pares instrução→resposta | ensinar o formato/seguimento de instruções |
| **Adaptação de domínio** | texto do domínio (jurídico, médico, código) | continuar o pré-treino |
| **Alinhamento** | comparações de preferência | RLHF, **DPO**, RLAIF |

## Fine-tuning completo × eficiente em parâmetros (PEFT)

Ajustar todos os parâmetros de um LLM exige memória para pesos + gradientes + estados do Adam (~16 bytes/parâmetro em precisão mista) e gera uma cópia completa por tarefa. **PEFT** treina uma fração mínima:

- **LoRA** (Hu et al., 2021): congela $W_0\in\mathbb R^{d\times k}$ e aprende uma atualização de **baixo posto**

$$W=W_0+\frac{\alpha}{r}BA,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d,k)$$

  $B$ inicia em zero (o modelo começa idêntico ao original). Parâmetros treináveis: $r(d+k)$ em vez de $dk$ (frequentemente <1%). Após o treino, funde-se $BA$ em $W_0$ → **sem custo extra na inferência**; adaptadores minúsculos são trocáveis por tarefa. Hipótese: a mudança de pesos durante a adaptação tem baixo posto intrínseco.
- **QLoRA**: LoRA sobre o modelo base quantizado em 4 bits → fine-tuning de LLMs grandes em uma GPU.
- **Adapters**, **prefix/prompt tuning**, **IA³**, BitFit (só vieses).

## Boas práticas

- LR bem menor que no pré-treino ($10^{-5}$–$10^{-4}$ completo; $10^{-4}$–$10^{-3}$ LoRA) com *warmup* e decaimento.
- Poucas épocas (1–3): overfitting e **esquecimento catastrófico** aparecem rápido; misture dados gerais se necessário.
- **Qualidade > quantidade** dos dados (LIMA: ~1000 exemplos bem escolhidos já ensinam o estilo).
- Avalie em dados retidos e em capacidades gerais; use o mesmo *chat template*/tokenização do modelo.
- Antes de fine-tunar, teste *prompting* e **RAG** — muitas vezes bastam.

## Implementação

- [`lora_from_scratch.py`](lora_from_scratch.py) (Python/PyTorch): `LoRALinear` do zero (com `merge`) numa MLP adaptada a um domínio novo; compara *sem adaptação*, *só última camada*, LoRA (r=1/4/8) e *fine-tuning completo* em acurácia, % de parâmetros treináveis e esquecimento.
- [`finetune_hf_lora.py`](finetune_hf_lora.py) (Python/Transformers+PEFT): DistilBERT para sentimento, fine-tuning completo vs LoRA (`peft`); baixa o modelo na 1ª execução.

```bash
python lora_from_scratch.py
pip install torch transformers peft && python finetune_hf_lora.py
```

## Referências

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021); Dettmers et al., *QLoRA* (2023).
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022); Rafailov et al., *DPO* (2023).
- Documentação Hugging Face PEFT / TRL.
