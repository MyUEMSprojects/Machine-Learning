"""SVM com scikit-learn: kernels linear/RBF em círculos concêntricos e busca de hiperparâmetros C e gamma."""
from sklearn.datasets import make_circles
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "poly", "rbf"]:
    m = SVC(kernel=kernel).fit(X_tr, y_tr)
    print(f"kernel={kernel:<7} acc teste={m.score(X_te, y_te):.3f}  vetores de suporte={m.n_support_.sum()}")

busca = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]},
    cv=5,
).fit(X_tr, y_tr)
print("melhores hiperparâmetros:", busca.best_params_, f"acc teste={busca.score(X_te, y_te):.3f}")
