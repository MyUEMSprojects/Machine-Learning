# Transfer Learning

Reaproveitar o conhecimento de um modelo treinado numa tarefa/dado de **origem** (grande) para uma tarefa de **destino** (poucos dados ou pouco poder computacional). Paradigma dominante em visão e NLP: ninguém treina um ResNet ou um LLM do zero para cada problema.

## Por que funciona

As primeiras camadas aprendem características **gerais** (bordas, texturas; em texto, sintaxe e semântica); só as últimas são específicas da tarefa. Essas representações são reutilizáveis.

## Estratégias

| Estratégia | O que fazer | Quando |
|---|---|---|
| **Feature extraction** | congelar o backbone, treinar só uma nova cabeça (ou um classificador linear/SVM nas features) | poucos dados; destino parecido com a origem |
| **Fine-tuning completo** | treinar tudo com LR pequeno (e *warmup*) | dados razoáveis; destino um pouco diferente |
| **Fine-tuning parcial / gradual** | descongelar camadas de cima para baixo; **LR diferencial** (menor nas camadas antigas) | meio-termo; reduz esquecimento |
| **PEFT** (LoRA, adapters, prompt tuning) | treinar poucos parâmetros extras, congelando o modelo | LLMs grandes — ver [fine-tuning](../../08-nlp/fine-tuning/) |

Cuidados: **esquecimento catastrófico** (perder o conhecimento original), *domain shift* entre origem e destino, usar o mesmo pré-processamento/normalização do pré-treino, congelar `BatchNorm` em batches pequenos.

## Relação com outros tópicos

[Aprendizado auto-supervisionado](../self-supervised/) produz os backbones pré-treinados (em dados sem rótulos); *few-shot* e *meta-learning* levam a ideia ao extremo.

## Implementação

[`transfer_learning.py`](transfer_learning.py) (Python/PyTorch): pré-treina uma MLP nos dígitos 0–4 com deslocamentos aleatórios (aprendendo invariância a translação) e transfere para as classes novas 5–9 com só 10 exemplos por classe, testando em imagens deslocadas. Compara **do zero**, **extrator congelado** e **fine-tuning com LR diferencial** (média de 5 sementes); inclui a receita para `torchvision`.

> Lição extra: em dígitos "limpos", treinar do zero com poucos exemplos já funciona bem e transferir não ajuda. Transfer learning só compensa quando a origem ensina algo que o destino não conseguiria aprender sozinho.

```bash
python transfer_learning.py
```

## Referências

- Yosinski et al., *How transferable are features in deep neural networks?* (2014).
- Howard & Ruder, *Universal Language Model Fine-tuning (ULMFiT)* (2018).
- Stanford CS231n, notas de Transfer Learning.
