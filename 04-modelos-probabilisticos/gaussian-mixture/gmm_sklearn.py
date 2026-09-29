"""GMM com scikit-learn: tipos de covariância, seleção de K por BIC e clustering suave."""
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(1)
X = np.vstack([
    rng.normal([-3, 0], 0.6, (300, 2)),
    rng.multivariate_normal([3, 2], [[2.25, 1.0], [1.0, 0.7]], 300),
    rng.normal([0, -4], 0.8, (300, 2)),
])

print("K | BIC (covariância completa)")
for k in range(1, 7):
    print(k, round(GaussianMixture(k, n_init=5, random_state=0).fit(X).bic(X), 1))

for tipo in ["spherical", "diag", "tied", "full"]:
    g = GaussianMixture(3, covariance_type=tipo, n_init=5, random_state=0).fit(X)
    print(f"covariância {tipo:9s} BIC = {g.bic(X):.1f}")

g = GaussianMixture(3, n_init=5, random_state=0).fit(X)
print("\npesos:", g.weights_.round(3))
print("P(componente | x=(-1.5, -2)) =", g.predict_proba([[-1.5, -2.0]]).round(3), " <- probabilidades, não rótulo duro")
