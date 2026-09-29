"""Árvore de decisão com scikit-learn: visualização em texto, importância de features e poda (ccp_alpha)."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

arv = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
print(export_text(arv, feature_names=iris.feature_names))
print("importâncias:", {n: round(float(v), 3) for n, v in zip(iris.feature_names, arv.feature_importances_)})

for alpha in [0.0, 0.01, 0.05]:  # poda por custo-complexidade
    a = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0).fit(X_tr, y_tr)
    print(f"ccp_alpha={alpha:<5} folhas={a.get_n_leaves():<3} acc teste={a.score(X_te, y_te):.3f}")
