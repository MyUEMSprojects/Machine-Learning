# Estatística

Inferir propriedades de uma população a partir de amostras — exatamente o problema de aprender de dados.

## Conceitos-chave

- **Estatísticas descritivas**: média, mediana, variância, desvio padrão, covariância, correlação.
- **Estimador**: função dos dados que estima um parâmetro. Viés e variância do estimador (ex.: variância amostral divide por $n-1$).
- **Máxima verossimilhança (MLE)**: $\hat\theta=\arg\max_\theta \prod_i p(x_i|\theta)$. Perdas comuns em ML são MLE disfarçada (MSE = MLE gaussiana; entropia cruzada = MLE categórica).
- **MAP**: MLE com prior; equivale a regularização.
- **Intervalos de confiança e testes de hipótese**: quantificam incerteza.
- **Bootstrap**: reamostragem com reposição para estimar incerteza sem fórmulas.
- **Correlação ≠ causalidade**; correlação mede só dependência linear.

## Implementação

[`estatistica.cpp`](estatistica.cpp): estatísticas descritivas, MLE gaussiano, IC 95% da média e bootstrap da mediana.

```bash
g++ -std=c++17 -O2 estatistica.cpp -o estatistica && ./estatistica
```

## Referências

- Wasserman, *All of Statistics*.
- Efron & Hastie, *Computer Age Statistical Inference*.
