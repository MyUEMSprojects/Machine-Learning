"""Regressão linear com scikit-learn: OLS, Ridge e Lasso no dataset diabetes."""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

for nome, m in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=0.1))]:
    pipe = make_pipeline(StandardScaler(), m).fit(X_tr, y_tr)
    print(f"{nome:6s} R² teste = {pipe.score(X_te, y_te):.3f}")
