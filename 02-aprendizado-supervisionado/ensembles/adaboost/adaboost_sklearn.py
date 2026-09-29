"""AdaBoost com scikit-learn: desempenho em função do número de estimadores fracos (stumps)."""
from sklearn.datasets import make_circles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_circles(n_samples=600, noise=0.15, factor=0.5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

print("stump sozinho:", round(DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr).score(X_te, y_te), 3))
for n in [1, 10, 50, 200]:
    m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n, random_state=0).fit(X_tr, y_tr)
    print(f"AdaBoost n={n:<4} acc teste={m.score(X_te, y_te):.3f}")
