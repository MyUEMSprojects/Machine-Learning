# k-Vizinhos Mais Próximos (k-NN)

Algoritmo **não paramétrico** e "preguiçoso" (*lazy*): não treina nada, apenas memoriza os dados. Para classificar $x$, acha os $k$ pontos de treino mais próximos e vota (classificação) ou faz a média (regressão).

## Algoritmo

1. Calcule a distância de $x$ a todos os pontos de treino (euclidiana: $\lVert a-b\rVert_2$; também Manhattan, Minkowski, cosseno).
2. Pegue os $k$ menores.
3. Classe = mais votada (opcionalmente ponderada por $1/d$); regressão = média.

## Hiperparâmetros e comportamento

- **$k$ pequeno** → fronteira irregular, variância alta (overfit; $k=1$ tem erro de treino 0).
- **$k$ grande** → fronteira suave, viés alto. Escolha por [validação cruzada](../../01-fundamentos-ml/validacao-cruzada/). Use $k$ ímpar em binário para evitar empates.
- **Escalonamento é obrigatório**: features de escala grande dominam a distância.
- **Maldição da dimensionalidade**: em alta dimensão, todas as distâncias ficam parecidas e o vizinho "mais próximo" perde significado; reduza dimensão antes ([PCA](../../03-aprendizado-nao-supervisionado/reducao-dimensionalidade/)).

## Complexidade

Treino $O(1)$; predição $O(nd)$ por consulta em força bruta. Aceleração: KD-tree / Ball-tree (baixa dimensão), busca aproximada (HNSW, FAISS) em alta dimensão.

## Implementação

- [`knn.cpp`](knn.cpp) (C++): classificação e regressão do zero; varre $k$ em "duas luas" mostrando o trade-off viés-variância.
- [`knn_sklearn.py`](knn_sklearn.py) (Python): efeito do escalonamento no dataset Wine.

```bash
g++ -std=c++17 -O2 knn.cpp -o knn && ./knn
python knn_sklearn.py
```

## Referências

- Cover & Hart, *Nearest neighbor pattern classification* (1967).
- Hastie et al., *ESL*, cap. 13.
