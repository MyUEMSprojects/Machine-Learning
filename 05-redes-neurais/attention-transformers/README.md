# Atenção e Transformers

## Atenção como busca suave

Cada posição gera uma **query** $q$; cada posição da sequência expõe **key** $k$ e **value** $v$. A saída é uma média dos values ponderada pela similaridade query–key:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

- $\sqrt{d_k}$ evita que os produtos internos cresçam com a dimensão e saturem o softmax.
- **Self-attention**: $Q,K,V$ vêm da mesma sequência ($Q=XW_Q$, $K=XW_K$, $V=XW_V$) → cada token agrega informação de todos os outros, com **caminho de comprimento 1** entre quaisquer duas posições (RNNs precisam de $O(T)$ passos) e **paralelizável** no tempo.
- **Máscara causal**: em modelos autorregressivos (GPT) a posição $i$ só vê $j\le i$ (pesos futuros = $-\infty$ antes do softmax).
- **Multi-head**: $h$ atenções em subespaços de dimensão $d/h$, concatenadas e projetadas — cabeças diferentes capturam relações diferentes (sintaxe, correferência…).
- Custo: $O(n^2d)$ tempo e $O(n^2)$ memória → gargalo para contextos longos (FlashAttention, atenção esparsa/linear, KV-cache na inferência).

## Posição

Self-attention é **equivariante a permutações**: não sabe a ordem. Injeta-se posição por **embeddings posicionais** (sinusoidais $\sin/\cos(p/10000^{2i/d})$, aprendidos, relativos, **RoPE**, ALiBi).

## Bloco Transformer

$$x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x)),\qquad x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$$

com FFN $=W_2\,\mathrm{GELU}(W_1x)$ (dimensão oculta $\approx4d$), **conexões residuais** e **LayerNorm** (aqui *pré-LN*, mais estável). Empilhe $L$ blocos.

## Arquiteturas

| Tipo | Atenção | Exemplos | Tarefas |
|---|---|---|---|
| Encoder-only | bidirecional | BERT | classificação, embeddings |
| Decoder-only | causal | GPT, Llama | geração de texto |
| Encoder–decoder | + atenção cruzada | T5, tradução original | seq2seq |
| ViT | patches de imagem como tokens | ViT | visão |

Detalhes de LLMs em [NLP/LLMs](../../08-nlp/llms/).

## Implementação

- [`attention.cpp`](attention.cpp) (C++, só forward): atenção escalonada, multi-head, máscara causal, LayerNorm, GELU, bloco Transformer completo e **verificações numéricas** (busca em dicionário, máscara, equivariância a permutação e como a codificação posicional a quebra).
- [`transformer_mini.py`](transformer_mini.py) (Python): MHA escrita à mão em PyTorch + mini encoder treinado para **inverter sequências**; imprime o mapa de atenção aprendido (anti-diagonal).

```bash
g++ -std=c++17 -O2 attention.cpp -o attention && ./attention
python transformer_mini.py
```

## Referências

- Vaswani et al., *Attention Is All You Need* (2017).
- Alammar, *The Illustrated Transformer*; Karpathy, *Let's build GPT: from scratch*.
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic).
