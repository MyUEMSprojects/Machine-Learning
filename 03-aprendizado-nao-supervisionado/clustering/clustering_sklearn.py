"""Comparação de algoritmos de clustering do scikit-learn em datasets com formatos diferentes (ARI)."""
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

datasets = {
    "blobs": make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=0),
    "luas": make_moons(n_samples=400, noise=0.07, random_state=0),
}
for nome, (X, y) in datasets.items():
    X = StandardScaler().fit_transform(X)
    k = len(set(y))
    algs = {
        "KMeans": KMeans(k, n_init=10, random_state=0),
        "Hierárquico (ward)": AgglomerativeClustering(k),
        "Hierárquico (single)": AgglomerativeClustering(k, linkage="single"),
        "DBSCAN": DBSCAN(eps=0.3),
        "GMM": GaussianMixture(k, random_state=0),
    }
    print(f"--- {nome} (ARI: 1 = perfeito, 0 = aleatório)")
    for n, a in algs.items():
        pred = a.fit_predict(X)
        print(f"  {n:22s} ARI = {adjusted_rand_score(y, pred):.3f}")
