# Bagging e Random Forest

## Bagging (Bootstrap AGGregatING)

1. Gere $B$ amostras *bootstrap* (sorteio com reposição, mesmo tamanho $n$).
2. Treine um modelo em cada uma (árvores completas: baixo viés, alta variância).
3. Combine: voto majoritário (classificação) ou média (regressão).

Média de $B$ modelos com variância $\sigma^2$ e correlação $\rho$: $\;\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$. Bagging reduz a variância sem aumentar o viés — mas o ganho é limitado pela **correlação** entre os modelos.

## Random Forest

Bagging de árvores **+ decorrelação**: em cada nó, só um subconjunto aleatório de $m$ features é considerado para o corte (típico: $m=\sqrt d$ classificação, $d/3$ regressão). Árvores ficam menos parecidas ($\rho$ menor) → média melhor.

## OOB (out-of-bag)

Cada bootstrap deixa ~36,8% ($e^{-1}$) das amostras de fora. Cada amostra pode ser avaliada só pelas árvores que não a viram → estimativa honesta do erro **sem** conjunto de validação.

## Prós e contras

- ✅ Poucos hiperparâmetros, robusto, paralelizável, importância de features, quase sem overfit ao aumentar $B$.
- ❌ Menos interpretável que uma árvore; modelo grande; em geral perde para gradient boosting bem ajustado em dados tabulares; não extrapola em regressão.
- **Extra-Trees**: também sorteia os limiares → ainda mais aleatório e rápido.

## Implementação

- [`random_forest.cpp`](random_forest.cpp) (C++): CART + bootstrap + sorteio de features + voto + OOB, em dados com 8 features de ruído.
- [`random_forest_sklearn.py`](random_forest_sklearn.py) (Python): `RandomForestClassifier`, `ExtraTrees`, OOB e importâncias.

```bash
g++ -std=c++17 -O2 random_forest.cpp -o random_forest && ./random_forest
python random_forest_sklearn.py
```

## Referências

- Breiman, *Random Forests* (2001); *Bagging Predictors* (1996).
- Hastie et al., *ESL*, cap. 15.
