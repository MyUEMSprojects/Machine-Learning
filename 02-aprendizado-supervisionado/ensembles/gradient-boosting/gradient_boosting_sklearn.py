"""Gradient Boosting: scikit-learn (HistGradientBoosting) e, se instalado, XGBoost/LightGBM.

pip install numpy scikit-learn  (opcional: xgboost lightgbm)
"""
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)  # baixa o dataset na 1ª vez
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

modelos = {
    "RandomForest": RandomForestRegressor(200, n_jobs=-1, random_state=0),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=0),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, early_stopping=True, random_state=0),
}
try:
    from xgboost import XGBRegressor
    modelos["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8)
except ImportError:
    pass

for nome, m in modelos.items():
    print(f"{nome:22s} R² teste = {m.fit(X_tr, y_tr).score(X_te, y_te):.3f}")
