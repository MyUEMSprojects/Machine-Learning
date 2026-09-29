"""Interpretabilidade com bibliotecas: importância por permutação, PDP/ICE (scikit-learn) e SHAP (TreeExplainer).

pip install scikit-learn shap
"""
import numpy as np
import shap
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split

data = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, random_state=0)
modelo = GradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
print(f"R² teste = {modelo.score(X_te, y_te):.3f}")

# 1) importância por permutação NO CONJUNTO DE TESTE (mede o que o modelo realmente usa para generalizar)
pi = permutation_importance(modelo, X_te, y_te, n_repeats=20, random_state=0)
ordem = np.argsort(-pi.importances_mean)
print("\nimportância por permutação (queda de R²):")
for i in ordem[:5]:
    print(f"  {data.feature_names[i]:4s} {pi.importances_mean[i]:.3f} ± {pi.importances_std[i]:.3f}")

# 2) dependência parcial da feature mais importante
j = int(ordem[0])
pd_res = partial_dependence(modelo, X_tr, [j], grid_resolution=6)
print(f"\nPDP de '{data.feature_names[j]}':", np.round(pd_res["average"][0], 1))

# 3) SHAP: contribuição de cada feature PARA CADA predição (exato e rápido para árvores)
expl = shap.TreeExplainer(modelo)
sv = expl.shap_values(X_te)
i = 0
print(f"\nSHAP da amostra 0: predição={modelo.predict(X_te[:1])[0]:.1f}, base={float(np.ravel(expl.expected_value)[0]):.1f}")
print("  soma(SHAP)+base =", round(float(sv[i].sum() + np.ravel(expl.expected_value)[0]), 1), "(propriedade de eficiência)")
print("  maiores contribuições:", [(data.feature_names[k], round(float(sv[i][k]), 1)) for k in np.argsort(-np.abs(sv[i]))[:3]])
print("\nimportância global SHAP (média |φ|):", {data.feature_names[k]: round(float(np.abs(sv[:, k]).mean()), 1) for k in np.argsort(-np.abs(sv).mean(0))[:4]})
