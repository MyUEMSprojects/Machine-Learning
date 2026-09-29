# Modelos de Linguagem de Grande Escala (LLMs)

LLMs são [Transformers](../../05-redes-neurais/attention-transformers/) (quase sempre *decoder-only*) treinados para **prever o próximo token** ([modelos de linguagem](../modelos-de-linguagem/)) em trilhões de tokens, com bilhões de parâmetros. Em escala suficiente surgem capacidades como seguir instruções, raciocínio, código e aprendizado em contexto.

## Ciclo de vida

1. **Pré-treino** (auto-supervisionado, caríssimo): prever o próximo token em web/livros/código → modelo base que "completa texto". Perda = entropia cruzada.
2. **Ajuste supervisionado (SFT)**: demonstrações de instruções→respostas ([fine-tuning](../fine-tuning/)).
3. **Alinhamento** com preferências: **RLHF** (modelo de recompensa + PPO — ver [gradiente de política](../../07-aprendizado-por-reforco/policy-gradient/)), DPO, RLAIF.
4. **Inferência**: geração autorregressiva token a token.

## Leis de escala

A perda cai como lei de potência com parâmetros $N$, dados $D$ e computação $C$ (Kaplan 2020). **Chinchilla** (Hoffmann 2022): para um orçamento fixo, $N$ e $D$ devem crescer juntos (~20 tokens por parâmetro) — modelos anteriores eram subtreinados.

## Geração (decodificação)

Em cada passo o modelo dá logits $z$ sobre o vocabulário; $p=\mathrm{softmax}(z/T)$.

| Estratégia | Descrição | Efeito |
|---|---|---|
| **Greedy** | argmax | determinístico, repetitivo |
| **Temperatura $T$** | reescala logits | $T<1$ conservador, $T>1$ criativo/caótico |
| **Top-$k$** | só os $k$ mais prováveis | corta a cauda ruim; tamanho fixo |
| **Top-$p$ (nucleus)** | menor conjunto com massa $\ge p$ | tamanho adaptativo à incerteza |
| **Beam search** | mantém as $b$ melhores sequências | bom para tradução/resumo; genérico em chat |

**KV-cache**: guarda as chaves/valores dos tokens já processados para não recalcular a atenção passada — custo por token passa de $O(n^2)$ para $O(n)$ (memória cresce com o contexto). Outras otimizações: quantização (int8/int4), FlashAttention, *speculative decoding*, *continuous batching*.

## Técnicas de uso

- **Prompting**: zero/few-shot, *chain-of-thought*.
- **RAG**: recuperar documentos por [embeddings](../embeddings/) e colocá-los no contexto (reduz alucinação, dá dados atualizados).
- **Agentes/ferramentas**: o modelo chama funções (busca, código, APIs).
- **Avaliação**: perplexidade (base), benchmarks (MMLU, HumanEval…), avaliação por humanos ou por outro LLM.

## Limitações

Alucinações, viés, janela de contexto finita, raciocínio frágil, tokenização (aritmética/contagem de letras), custo. Interpretabilidade é um campo ativo ([XAI](../../10-topicos-avancados/interpretabilidade-xai/)).

## Implementação

- [`sampling.cpp`](sampling.cpp) (C++): greedy, temperatura, top-$k$, top-$p$ e beam search do zero; mostra o efeito na distribuição e a diferença entre tamanho fixo (top-k) e adaptativo (top-p).
- [`mini_gpt.py`](mini_gpt.py) (Python/PyTorch): **GPT minúsculo do zero** (atenção causal, pré-LN, weight tying, AdamW + OneCycle) treinado nos READMEs do repo, com geração top-$k$/temperatura.
- [`hf_generate.py`](hf_generate.py) (Python/Transformers): usar um LLM pré-treinado (`distilgpt2`): distribuição do próximo token, entropia, decodificações e perplexidade.

```bash
g++ -std=c++17 -O2 sampling.cpp -o sampling && ./sampling
python mini_gpt.py
python hf_generate.py            # baixa o modelo na 1ª execução
```

## Referências

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2); Brown et al., *GPT-3* (2020).
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022).
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (top-p, 2020).
- Karpathy, *nanoGPT* e *Let's build GPT*.
