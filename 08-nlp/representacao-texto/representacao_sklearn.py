"""Representação de texto com scikit-learn e Hugging Face tokenizers: BoW, TF-IDF, n-gramas e tokenizadores BPE/WordPiece.

pip install scikit-learn transformers   (os tokenizadores são baixados na 1ª execução)
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

textos = [
    "ótimo filme, adorei a história", "filme maravilhoso e emocionante", "atuação excelente, recomendo muito",
    "história envolvente e final incrível", "amei cada minuto, obra-prima",
    "filme horrível, perdi meu tempo", "roteiro fraco e atuação péssima", "muito chato, não recomendo",
    "final decepcionante e previsível", "detestei, uma completa perda de tempo",
]
rotulos = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

bow = CountVectorizer()
X = bow.fit_transform(textos)
print("BoW:", X.shape, "| exemplo de vocabulário:", list(bow.vocabulary_)[:6])

tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
Xt = tfidf.fit_transform(textos)
print("TF-IDF com bigramas:", Xt.shape)
print("similaridade cosseno (doc 0 vs 1, positivos):", cosine_similarity(Xt[0], Xt[1])[0, 0].round(3),
      "| (doc 0 vs 5, pos vs neg):", cosine_similarity(Xt[0], Xt[5])[0, 0].round(3))

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression())
print("acurácia CV(5) classificador TF-IDF + logística:", cross_val_score(clf, textos, rotulos, cv=5).mean().round(2))

try:
    from transformers import AutoTokenizer

    frase = "Aprendizado de máquina é fascinante!"
    for nome in ["gpt2", "distilbert-base-uncased"]:
        tok = AutoTokenizer.from_pretrained(nome)
        print(f"{nome:26s} ->", tok.tokenize(frase))
except Exception as e:  # sem internet / transformers ausente
    print("(pulando tokenizadores HF:", type(e).__name__, ")")
