# Embeddings

Representações **densas e de baixa dimensão** (dezenas a milhares de números) em que a **geometria captura significado**: palavras/frases parecidas ficam próximas.

## Hipótese distribucional

"Você conhece uma palavra pela companhia que ela mantém" (Firth, 1957): palavras que aparecem em contextos semelhantes têm significados semelhantes. Todo método abaixo explora isso.

## word2vec (Mikolov et al., 2013)

**Skip-gram**: dada uma palavra central $w$, prever palavras de contexto $c$ na janela. Com **negative sampling**, treina-se um classificador logístico que separa pares reais $(w,c)$ de pares falsos $(w,c')$ com $c'$ amostrado de $P_n(c)\propto f(c)^{3/4}$:

$$\log\sigma(u_c^\top v_w)+\sum_{k=1}^{K}\mathbb E_{c'\sim P_n}\big[\log\sigma(-u_{c'}^\top v_w)\big]$$

Cada palavra tem um vetor $v$ (entrada) e $u$ (contexto); usa-se $v$ (ou a média). **CBOW** faz o inverso (contexto → palavra). Resultado famoso: **analogias** lineares, $\vec{rei}-\vec{homem}+\vec{mulher}\approx\vec{rainha}$.

Outros: **GloVe** (fatora estatísticas globais de co-ocorrência), **fastText** (soma vetores de n-gramas de caracteres → lida com palavras raras/OOV e morfologia). Levy & Goldberg mostraram que o SGNS equivale a fatorar implicitamente a matriz de **PMI deslocada**, e `PPMI + SVD` funciona quase tão bem.

## Limitações → embeddings contextuais

Um vetor por palavra não distingue "banco" (assento) de "banco" (instituição) e reflete vieses do corpus. **ELMo/BERT/GPT** produzem vetores **dependentes do contexto** (a saída de um [Transformer](../../05-redes-neurais/attention-transformers/)). **Sentence embeddings** (Sentence-BERT, E5, OpenAI/Cohere embeddings) mapeiam frases/documentos inteiros; treinados de forma contrastiva ([self-supervised](../../06-deep-learning-avancado/self-supervised/)).

## Usos

Busca semântica e **RAG** (recuperar por similaridade do cosseno em um banco vetorial: FAISS, pgvector), clusterização, classificação com poucos rótulos, recomendação (embeddings de itens/usuários), detecção de duplicatas.

Medida de similaridade: **cosseno** $\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}$ (ou produto interno com vetores normalizados).

## Implementação

- [`word2vec.cpp`](word2vec.cpp) (C++): **skip-gram com negative sampling do zero** (SGD manual) em corpus sintético; imprime vizinhos mais próximos e a similaridade dentro/entre categorias.
- [`embeddings_gensim.py`](embeddings_gensim.py) (Python): `gensim.Word2Vec`, alternativa **PPMI + SVD** e (opcional) embeddings de sentenças pré-treinados.

```bash
g++ -std=c++17 -O2 word2vec.cpp -o word2vec && ./word2vec
pip install gensim scikit-learn && python embeddings_gensim.py
```

## Referências

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013) e *Distributed Representations of Words and Phrases* (2013).
- Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (2014).
- Reimers & Gurevych, *Sentence-BERT* (2019).
