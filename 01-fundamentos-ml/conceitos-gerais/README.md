# Conceitos Gerais de Machine Learning

## O que é aprender?

Um programa aprende de uma experiência $E$ em relação a uma tarefa $T$ e medida de desempenho $P$ se seu desempenho em $T$, medido por $P$, melhora com $E$ (Mitchell, 1997).

Formalmente: dado um conjunto de dados $\{(x_i,y_i)\}$ amostrado de uma distribuição desconhecida, achar $f_\theta$ que **generalize** para dados novos.

## Paradigmas

| Paradigma | Dados | Objetivo | Exemplos |
|---|---|---|---|
| **Supervisionado** | $(x,y)$ rotulados | prever $y$ | regressão, classificação |
| **Não supervisionado** | só $x$ | achar estrutura | clustering, PCA |
| **Semi/auto-supervisionado** | poucos rótulos / rótulos derivados dos dados | representações | BERT, SimCLR |
| **Por reforço** | recompensas de interação | política ótima | jogos, robótica |

Tarefas supervisionadas: **regressão** (saída contínua) e **classificação** (saída discreta).

## Ingredientes de um algoritmo de ML

1. **Modelo** (espaço de hipóteses): $f_\theta(x)$.
2. **Função de perda** $L(f_\theta(x),y)$: mede o erro.
3. **Otimização**: como achar $\theta$ que minimiza o risco empírico $\frac1N\sum_i L(f_\theta(x_i),y_i)$.

## Pipeline típico

1. Definir o problema e a métrica de sucesso.
2. Coletar/limpar dados; análise exploratória.
3. **Separar treino / validação / teste** (o teste só é usado uma vez, no fim).
4. Pré-processar e criar features (ajustando **só no treino**).
5. Treinar, ajustar hiperparâmetros na validação.
6. Avaliar no teste, monitorar em produção.

## Glossário

- **Parâmetros** (aprendidos) vs. **hiperparâmetros** (definidos por você: $\eta$, profundidade da árvore, $k$).
- **Overfitting**: decora o treino, falha em dados novos. **Underfitting**: simples demais.
- **Vazamento de dados (data leakage)**: informação do teste contaminando o treino → métricas otimistas e falsas.
- **No free lunch**: nenhum algoritmo é melhor em todos os problemas.
- **Viés indutivo**: suposições que permitem generalizar (ex.: k-NN assume vizinhos parecidos).

## Implementação

[`pipeline_sklearn.py`](pipeline_sklearn.py) (Python): pipeline completo com `train_test_split` estratificado, `StandardScaler` + `LogisticRegression` e relatório de classificação.

```bash
pip install numpy scikit-learn
python pipeline_sklearn.py
```

## Referências

- Mitchell, *Machine Learning*; Géron, *Hands-On Machine Learning*.
- Hastie et al., *The Elements of Statistical Learning*, cap. 1–2.
