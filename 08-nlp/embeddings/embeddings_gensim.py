"""Embeddings de palavras: word2vec (gensim) e PPMI + SVD (NumPy), no mesmo corpus sintético do exemplo em C++.
Também mostra como usar embeddings de sentenças pré-treinados (sentence-transformers), se disponível.

pip install gensim scikit-learn numpy  (opcional: sentence-transformers)
"""
import random
from collections import Counter

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

random.seed(0)
animais = ["gato", "cachorro", "cavalo", "vaca", "ovelha"]
comidas = ["pao", "queijo", "maca", "arroz", "bolo"]
veiculos = ["carro", "onibus", "trem", "aviao", "barco"]
sents = []
for _ in range(6000):
    sents.append(["o", random.choice(animais), random.choice(["corre", "dorme", "late", "pasta"]), "no", random.choice(["campo", "quintal", "pasto"])])
    sents.append(["eu", "como", random.choice(comidas), "no", random.choice(["cafe", "almoco", "jantar"])])
    sents.append(["o", random.choice(veiculos), random.choice(["anda", "para", "passa"]), "na", random.choice(["rua", "estrada", "pista"])])

# --- word2vec skip-gram com negative sampling ---
w2v = Word2Vec(sents, vector_size=16, window=2, min_count=1, sg=1, negative=5, epochs=5, seed=0, workers=1)
for q in ["gato", "pao", "carro"]:
    print(f"word2vec vizinhos de {q:6s}:", [(w, round(s, 2)) for w, s in w2v.wv.most_similar(q, topn=3)])

# --- alternativa por contagem: matriz de co-ocorrência -> PPMI -> SVD (equivale implicitamente ao SGNS, Levy & Goldberg 2014) ---
vocab = sorted({w for s in sents for w in s})
idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))
for s in sents:
    for i, w in enumerate(s):
        for j in range(max(0, i - 2), min(len(s), i + 3)):
            if i != j:
                C[idx[w], idx[s[j]]] += 1
total, pw, pc = C.sum(), C.sum(1, keepdims=True), C.sum(0, keepdims=True)
with np.errstate(divide="ignore"):
    pmi = np.log(C * total / (pw * pc))
ppmi = np.maximum(pmi, 0)
E = TruncatedSVD(8, random_state=0).fit_transform(ppmi)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
for q in ["gato", "pao", "carro"]:
    sim = E @ E[idx[q]]
    viz = [(vocab[i], round(float(sim[i]), 2)) for i in np.argsort(-sim)[1:4]]
    print(f"PPMI+SVD vizinhos de {q:6s}:", viz)

try:  # embeddings de SENTENÇAS pré-treinados (baixa ~90 MB na 1ª vez)
    from sentence_transformers import SentenceTransformer, util

    m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    a, b, c = m.encode(["O gato dorme no sofá.", "Um felino descansa na poltrona.", "A bolsa de valores caiu hoje."])
    print("similaridade  gato~felino:", float(util.cos_sim(a, b)), " gato~bolsa:", float(util.cos_sim(a, c)))
except Exception:
    print("(sentence-transformers não instalado — pulando exemplo de embeddings de sentenças)")
