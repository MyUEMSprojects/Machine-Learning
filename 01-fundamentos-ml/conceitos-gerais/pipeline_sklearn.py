"""Pipeline completo de ML com scikit-learn: dados -> split -> pré-processamento -> modelo -> avaliação.

Uso: python pipeline_sklearn.py   (requer numpy, scikit-learn)
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

# 1) separar treino/teste ANTES de qualquer ajuste (evita vazamento de dados)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# 2) Pipeline garante que o scaler só "vê" o treino
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo.fit(X_tr, y_tr)

# 3) avaliar no teste (dados nunca vistos)
print(f"acurácia treino: {modelo.score(X_tr, y_tr):.3f}")
print(f"acurácia teste : {modelo.score(X_te, y_te):.3f}")
print(classification_report(y_te, modelo.predict(X_te)))
