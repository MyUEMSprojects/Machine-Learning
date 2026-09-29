"""Regressão logística com scikit-learn: probabilidades e efeito da regularização C."""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
for C in [0.001, 0.01, 0.1, 1, 10]:  # C = 1/lambda: menor C => mais regularização
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    print(f"C={C:<6} acurácia CV(5) = {cross_val_score(m, X, y, cv=5).mean():.3f}")
