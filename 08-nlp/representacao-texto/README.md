# Representação de Texto

Modelos precisam de números: como transformar texto em vetores?

## Pipeline clássico

1. **Tokenização**: dividir em unidades (palavras, subpalavras, caracteres, bytes).
2. Normalização: minúsculas, remoção de pontuação/acentos, *stopwords*, stemming/lematização (menos usados hoje).
3. **Vetorização**.

## Bag-of-Words (BoW)

Vetor de contagens sobre um vocabulário fixo; ignora a ordem. Esparso, alta dimensão. **N-gramas** (bigramas…) recuperam um pouco de contexto local.

## TF-IDF

$$\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t),\qquad \mathrm{idf}(t)=\ln\frac{1+N}{1+\mathrm{df}(t)}+1$$

Pondera termos frequentes no documento e **raros** na coleção (palavras como "o", "de" recebem peso baixo). Com normalização L2, o produto interno entre dois vetores é a **similaridade do cosseno**: $\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$. Base de buscadores e de bons *baselines* de classificação (com regressão logística/SVM linear, ou [Naive Bayes](../../02-aprendizado-supervisionado/naive-bayes/)).

**BM25**: evolução do TF-IDF para *ranking*, com saturação da frequência do termo ($k_1$) e normalização de tamanho ($b$); ainda é o padrão em busca lexical e complementa busca vetorial (RAG híbrido).

**Limite**: sem semântica — "gato" e "cachorro" são tão diferentes quanto "gato" e "rede" → [embeddings](../embeddings/).

## Tokenização por subpalavras

Palavras inteiras geram vocabulário enorme e palavras fora do vocabulário (OOV); caracteres geram sequências longas. **Subpalavras** são o meio-termo.

- **BPE (Byte-Pair Encoding)**: começa com caracteres/bytes e **funde iterativamente o par adjacente mais frequente** até atingir o tamanho de vocabulário desejado. GPT/Llama usam BPE sobre bytes (nunca há token desconhecido).
- **WordPiece** (BERT): funde pares que maximizam a verossimilhança do corpus; prefixo `##` marca continuação.
- **Unigram/SentencePiece** (T5, Llama): parte de um vocabulário grande e remove tokens que menos prejudicam a verossimilhança; independente de espaços.

Consequências práticas: custo/contexto são medidos em **tokens**, não palavras; idiomas com menos dados (ex.: português) gastam mais tokens por palavra; números e código são tokenizados de formas estranhas (fonte de erros aritméticos de LLMs).

## Implementação

- [`tfidf.cpp`](tfidf.cpp) (C++): tokenização, BoW, TF-IDF, cosseno e ranking por BM25 vs. TF-IDF.
- [`bpe.cpp`](bpe.cpp) (C++): treinamento de BPE do zero (10 merges) e tokenização de palavras novas.
- [`representacao_sklearn.py`](representacao_sklearn.py) (Python): `CountVectorizer`/`TfidfVectorizer` com n-gramas, classificação de sentimento e tokenizadores reais (GPT-2 BPE, DistilBERT WordPiece) via `transformers`.

```bash
g++ -std=c++17 -O2 tfidf.cpp -o tfidf && ./tfidf
g++ -std=c++17 -O2 bpe.cpp -o bpe && ./bpe
python representacao_sklearn.py
```

## Referências

- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (gratuito).
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE, 2016).
- Karpathy, *Let's build the GPT Tokenizer* (vídeo).
