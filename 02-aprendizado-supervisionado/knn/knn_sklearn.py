"""k-NN com scikit-learn: escolha de k por validação cruzada e importância do escalonamento."""
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)  # features em escalas muito diferentes
for k in [1, 5, 15]:
    sem = cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean()
    com = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), X, y, cv=5).mean()
    print(f"k={k:<3} sem escalonar={sem:.3f}  padronizado={com:.3f}")
