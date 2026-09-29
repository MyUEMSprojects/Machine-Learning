"""Detecção de anomalias com scikit-learn: IsolationForest, LOF, One-Class SVM e EllipticEnvelope."""
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(0)
normais = np.vstack([rng.normal([0, 0], 0.6, (300, 2)), rng.normal([4, 4], 0.9, (300, 2))])  # 2 modos
anomalias = rng.uniform(-3, 7, (30, 2))
X = np.vstack([normais, anomalias])
y = np.r_[np.zeros(len(normais)), np.ones(len(anomalias))]  # 1 = anomalia

modelos = {
    "IsolationForest": IsolationForest(random_state=0).fit(X),
    "LOF": LocalOutlierFactor(n_neighbors=30, novelty=True).fit(X),
    "OneClassSVM": OneClassSVM(gamma="scale", nu=0.05).fit(X),
    "EllipticEnvelope": EllipticEnvelope(random_state=0).fit(X),  # assume 1 gaussiana -> falha em 2 modos
}
for nome, m in modelos.items():
    score = -m.score_samples(X) if hasattr(m, "score_samples") else -m.decision_function(X)
    print(f"{nome:17s} AUC = {roc_auc_score(y, score):.3f}")
