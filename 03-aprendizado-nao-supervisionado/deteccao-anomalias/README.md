# Detecção de Anomalias (Outliers)

Achar observações raras que diferem da maioria: fraude, falhas de máquina, intrusão, erros de dados. Tipicamente há **poucos ou nenhum exemplo rotulado** de anomalia, e as classes são extremamente desbalanceadas.

## Abordagens

| Família | Método | Ideia |
|---|---|---|
| Estatística | z-score, **Mahalanobis** | quão longe da média, considerando a covariância: $d^2=(x-\mu)^\top\Sigma^{-1}(x-\mu)$ (~$\chi^2_d$ se gaussiano) |
| Distância/densidade | **LOF** | densidade local do ponto vs. a de seus vizinhos |
| Isolamento | **Isolation Forest** | cortes aleatórios isolam anomalias com poucos passos |
| Fronteira | **One-Class SVM** | fronteira que envolve os dados normais |
| Reconstrução | autoencoder, PCA | erro de reconstrução alto = anômalo |

## Isolation Forest

Constrói árvores com feature e limiar **totalmente aleatórios** em subamostras. Anomalias, poucas e diferentes, são separadas perto da raiz (caminho curto). Score:

$$s(x)=2^{-E[h(x)]/c(\psi)}$$

onde $h$ é o comprimento do caminho e $c(\psi)$ o caminho médio numa BST com $\psi$ pontos. $s\to1$: anomalia; $s\approx0.5$: normal. Linear no número de amostras, sem cálculo de distâncias, funciona bem em alta dimensão.

## Considerações

- Avalie com **ROC AUC / PR AUC**, não acurácia.
- Defina o limiar pela taxa de contaminação esperada ou pelo custo de falsos alarmes.
- **Novelty** (treino só com dados normais) vs. **outlier detection** (treino já contém anomalias).
- Séries temporais: veja [séries temporais](../../10-topicos-avancados/series-temporais/).

## Implementação

- [`deteccao_anomalias.cpp`](deteccao_anomalias.cpp) (C++): Mahalanobis e Isolation Forest do zero; mostra que só Mahalanobis destaca o ponto "contra a correlação" em dados correlacionados.
- [`deteccao_anomalias_sklearn.py`](deteccao_anomalias_sklearn.py) (Python): IsolationForest, LOF, One-Class SVM, EllipticEnvelope e AUC.

```bash
g++ -std=c++17 -O2 deteccao_anomalias.cpp -o deteccao_anomalias && ./deteccao_anomalias
python deteccao_anomalias_sklearn.py
```

## Referências

- Liu, Ting & Zhou, *Isolation Forest* (2008).
- Breunig et al., *LOF: Identifying Density-Based Local Outliers* (2000).
- Chandola et al., *Anomaly Detection: A Survey* (2009).
