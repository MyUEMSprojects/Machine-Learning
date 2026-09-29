# Modelos de Linguagem

Um **modelo de linguagem** atribui probabilidade a sequências de tokens. Pela regra da cadeia:

$$P(w_1,\dots,w_n)=\prod_{i=1}^{n}P(w_i\mid w_1,\dots,w_{i-1})$$

Prever o próximo token é a tarefa de treino de GPT, Llama e companhia ([LLMs](../llms/)).

## Modelos n-gram

Aproximam o histórico pelos últimos $n-1$ tokens (hipótese de Markov): $P(w_i|w_{i-n+1:i-1})$, estimado por contagens (MLE): $\frac{C(w_{i-2}w_{i-1}w_i)}{C(w_{i-2}w_{i-1})}$.

**Problema da esparsidade**: a maioria dos n-gramas nunca aparece no treino → probabilidade 0 → perplexidade infinita. Soluções:

| Técnica | Ideia |
|---|---|
| **Add-$k$ (Laplace)** | soma $k$ a todas as contagens |
| **Interpolação** | $\lambda_3P_{tri}+\lambda_2P_{bi}+\lambda_1P_{uni}$ |
| **Backoff** (Katz, *stupid backoff*) | usa ordem menor quando a maior não tem evidência |
| **Kneser–Ney** | melhor método clássico; usa a "diversidade de contextos" das palavras |

Limites: contexto curto, sem generalização entre palavras parecidas, tabelas gigantes.

## Perplexidade

$$\mathrm{PP}=\exp\Big(-\frac1N\sum_i\log P(w_i\mid \text{contexto})\Big)=2^{H}$$

Interpretação: "fator de ramificação efetivo" — o número de escolhas igualmente prováveis que o modelo enxerga a cada passo. Menor é melhor; só é comparável com o mesmo vocabulário/tokenização. Equivale à exponencial da entropia cruzada ([teoria da informação](../../00-fundamentos-matematicos/teoria-da-informacao/)); *bits por caractere* = $\log_2\mathrm{PP}$ em nível de caractere.

## Modelos neurais

- **Bengio et al. (2003)**: MLP sobre embeddings das $n-1$ palavras anteriores — generaliza entre palavras similares ([embeddings](../embeddings/)).
- **RNN/LSTM**: contexto ilimitado em princípio, estado oculto carrega o histórico ([RNNs](../../05-redes-neurais/rnn-lstm/)).
- **Transformers**: atenção sobre todo o contexto, treino paralelo ([atenção](../../05-redes-neurais/attention-transformers/)) → [LLMs](../llms/).

## Geração

Amostrar $w_i\sim P(\cdot|\text{contexto})$ repetidamente. A **temperatura** $T$ reescala logits ($p\propto e^{z/T}$): $T\to0$ é guloso/repetitivo, $T$ alto é caótico (ver [amostragem](../llms/)).

## Implementação

- [`ngram.cpp`](ngram.cpp) (C++): unigrama/bigrama/trigrama do zero, add-$k$, interpolação, MLE sem suavização, **perplexidade** e geração de texto em corpus sintético (o trigrama supera o bigrama porque captura a dependência verbo→objeto).
- [`lm_lstm_char.py`](lm_lstm_char.py) (Python/PyTorch): LSTM de 2 camadas em nível de caractere, treinada nos próprios READMEs do repositório; reporta perplexidade e bits/char e gera texto com diferentes temperaturas. Com um corpus tão pequeno (~120 mil caracteres) a perplexidade de validação começa a **subir** depois de ~600 passos enquanto a de treino continua caindo — *overfitting* clássico; por isso o treino é curto.

```bash
g++ -std=c++17 -O2 ngram.cpp -o ngram && ./ngram
python lm_lstm_char.py
```

## Referências

- Jurafsky & Martin, *Speech and Language Processing*, cap. 3 (n-gramas) e 9–10 (gratuito online).
- Bengio et al., *A Neural Probabilistic Language Model* (2003).
- Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*.
