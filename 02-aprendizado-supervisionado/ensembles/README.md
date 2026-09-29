# Ensembles

Combinar vários modelos fracos/instáveis para obter um modelo forte. Duas estratégias complementares:

| | Bagging / Random Forest | Boosting |
|---|---|---|
| Treino | paralelo, independente | sequencial, cada um corrige o anterior |
| Learner base | árvores profundas (viés baixo, variância alta) | árvores rasas (viés alto) |
| Reduz principalmente | **variância** | **viés** |
| Risco de overfit | baixo | maior (ajuste `learning_rate`, early stopping) |

Existe ainda **stacking** (um meta-modelo aprende a combinar as previsões dos modelos base) e **voting** (média/voto simples).

## Subtópicos

- [`bagging-random-forest/`](bagging-random-forest/): bootstrap, decorrelação de árvores, OOB.
- [`adaboost/`](adaboost/): reponderação de exemplos, perda exponencial.
- [`gradient-boosting/`](gradient-boosting/): boosting como descida do gradiente; base de XGBoost/LightGBM.

Pré-requisito: [árvores de decisão](../arvores-de-decisao/).
