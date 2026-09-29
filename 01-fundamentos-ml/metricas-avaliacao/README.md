# Métricas de Avaliação

A métrica deve refletir o **custo real** dos erros no problema — escolha-a antes de treinar.

## Regressão

| Métrica | Fórmula | Notas |
|---|---|---|
| MSE | $\frac1n\sum(y-\hat y)^2$ | penaliza erros grandes; diferenciável |
| RMSE | $\sqrt{\text{MSE}}$ | mesma unidade de $y$ |
| MAE | $\frac1n\sum\lvert y-\hat y\rvert$ | robusta a outliers |
| $R^2$ | $1-\frac{SS_{res}}{SS_{tot}}$ | fração da variância explicada (0 = prever a média) |

## Classificação

Matriz de confusão: TP, FP, TN, FN.

- **Acurácia** $=\frac{TP+TN}{n}$ — enganosa com classes desbalanceadas.
- **Precisão** $=\frac{TP}{TP+FP}$ — dos que previ positivos, quantos eram? (importante quando falso positivo é caro: spam)
- **Recall** $=\frac{TP}{TP+FN}$ — dos positivos reais, quantos achei? (importante quando falso negativo é caro: câncer)
- **F1** — média harmônica de precisão e recall.
- **ROC AUC**: área sob a curva TPR×FPR variando o limiar. Interpretação: probabilidade de um positivo ter score maior que um negativo. Independe do limiar.
- **PR AUC**: melhor que ROC com desbalanceamento forte.
- **Log loss** (entropia cruzada): avalia a qualidade das *probabilidades*, não só da decisão.
- Multiclasse: média *macro* (igual por classe) vs. *micro* (global) vs. *weighted*.

## Outras

Ranking/busca (NDCG, MAP), geração de texto (BLEU, perplexidade), clustering (silhouette, ARI), detecção (IoU, mAP).

## Implementação

[`metricas.cpp`](metricas.cpp): todas as métricas acima do zero, incluindo AUC por comparação de pares, e a demonstração de que "sempre prever 0" dá 95% de acurácia num problema desbalanceado.

```bash
g++ -std=c++17 -O2 metricas.cpp -o metricas && ./metricas
```

## Referências

- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets* (2015).
- Documentação `sklearn.metrics`.
