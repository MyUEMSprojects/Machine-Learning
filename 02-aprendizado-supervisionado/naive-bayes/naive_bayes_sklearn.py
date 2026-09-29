"""Naive Bayes com scikit-learn: MultinomialNB + TF-IDF para classificação de texto (20 newsgroups).

Baixa o dataset na primeira execução (precisa de internet).
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.misc"]
tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))

m = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB(alpha=0.1)).fit(tr.data, tr.target)
print(f"acurácia teste: {accuracy_score(te.target, m.predict(te.data)):.3f}")
