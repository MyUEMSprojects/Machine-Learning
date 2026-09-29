"""Redução de dimensionalidade com scikit-learn: PCA, Kernel PCA, t-SNE e Isomap nos dígitos (64D -> 2D).

Gera o arquivo reducao_digitos.png com as projeções (requer matplotlib).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)

pca = PCA().fit(X)
acum = pca.explained_variance_ratio_.cumsum()
print("componentes para 90% da variância:", int((acum < 0.90).sum() + 1), "de", X.shape[1])

metodos = {
    "PCA": PCA(2),
    "Kernel PCA (RBF)": KernelPCA(2, kernel="rbf", gamma=0.001),
    "Isomap": Isomap(n_components=2),
    "t-SNE": TSNE(2, random_state=0),
}
fig, eixos = plt.subplots(1, 4, figsize=(20, 5))
for ax, (nome, m) in zip(eixos, metodos.items()):
    Z = m.fit_transform(X)
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6)
    ax.set_title(nome)
    acc = cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
    print(f"{nome:18s} k-NN em 2D: acurácia CV = {acc:.3f}")
fig.savefig("reducao_digitos.png", dpi=110, bbox_inches="tight")
print("figura salva em reducao_digitos.png")
