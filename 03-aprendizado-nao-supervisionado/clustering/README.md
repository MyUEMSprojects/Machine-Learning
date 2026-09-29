# Clustering

Agrupar amostras semelhantes **sem rótulos**. Não há resposta "certa": a qualidade depende da métrica de distância, da forma dos grupos e do objetivo.

## k-means

Minimiza a **inércia** $\sum_i\lVert x_i-\mu_{c(i)}\rVert^2$ alternando:
1. **Atribuição**: cada ponto vai ao centro mais próximo.
2. **Atualização**: cada centro vira a média do seu cluster.

Converge (a um mínimo local). Boas práticas: inicialização **k-means++** (espalha os centros), várias reinicializações (`n_init`), padronizar features. Assume clusters esféricos, de tamanho parecido; $k$ é dado — escolha por **cotovelo** (inércia × $k$) ou **silhueta** ($s=\frac{b-a}{\max(a,b)}\in[-1,1]$). Custo $O(nkdI)$.

## Hierárquico aglomerativo

Começa com cada ponto como um cluster e funde repetidamente os dois mais próximos, produzindo um **dendrograma**; corte na altura desejada para escolher $k$. A **ligação** define a distância entre clusters:

| Ligação | Distância | Efeito |
|---|---|---|
| single | mínima | encadeamento, acha formas alongadas, sensível a ruído |
| complete | máxima | clusters compactos |
| average | média | meio-termo |
| ward | aumento da variância | parecido com k-means |

Custo $O(n^2)$–$O(n^3)$: só para $n$ moderado.

## DBSCAN

Baseado em **densidade**: dois parâmetros, $\varepsilon$ (raio) e `min_pts`. Ponto *núcleo* tem ≥ `min_pts` vizinhos em $\varepsilon$; clusters são componentes conexas de pontos núcleo (+ bordas); o resto é **ruído**. Não precisa de $k$, acha formas arbitrárias e detecta outliers; sofre com densidades variadas (veja HDBSCAN) e alta dimensão.

## Outros

[Misturas gaussianas](../../04-modelos-probabilisticos/gaussian-mixture/) (clustering suave, probabilístico), spectral clustering, mean-shift, HDBSCAN.

## Avaliação

Interna (sem rótulos): silhueta, Davies–Bouldin, inércia. Externa (com rótulos de referência): Adjusted Rand Index (ARI), informação mútua normalizada.

## Implementação

- [`kmeans.cpp`](kmeans.cpp): Lloyd + k-means++ + múltiplas inicializações; varredura de $k$ com inércia e silhueta.
- [`dbscan.cpp`](dbscan.cpp): DBSCAN em "duas luas" com ruído.
- [`hierarquico.cpp`](hierarquico.cpp): aglomerativo com ligações single/complete/average e corte do dendrograma.
- [`clustering_sklearn.py`](clustering_sklearn.py): comparação de algoritmos em blobs vs. luas.

```bash
g++ -std=c++17 -O2 kmeans.cpp -o kmeans && ./kmeans
g++ -std=c++17 -O2 dbscan.cpp -o dbscan && ./dbscan
g++ -std=c++17 -O2 hierarquico.cpp -o hierarquico && ./hierarquico
python clustering_sklearn.py
```

## Referências

- Arthur & Vassilvitskii, *k-means++: The Advantages of Careful Seeding* (2007).
- Ester et al., *A Density-Based Algorithm for Discovering Clusters* (DBSCAN, 1996).
