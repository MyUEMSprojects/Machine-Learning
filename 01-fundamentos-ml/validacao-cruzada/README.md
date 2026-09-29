# Validação Cruzada

Como estimar o desempenho em dados novos e escolher hiperparâmetros **sem tocar no conjunto de teste**.

## Hold-out

Divide em treino/validação/teste (ex.: 60/20/20). Simples, mas a estimativa é ruidosa e desperdiça dados.

## k-fold

1. Embaralhe e divida os dados em $k$ partes (folds).
2. Para cada fold $i$: treine nos outros $k-1$, valide no fold $i$.
3. Média dos $k$ erros = estimativa do erro de generalização.

Tipicamente $k=5$ ou $10$. Custo: $k$ treinos.

## Variantes

- **Estratificada**: preserva proporção de classes em cada fold (classificação desbalanceada).
- **Leave-One-Out**: $k=n$; baixo viés, alta variância e caro.
- **Repeated k-fold**: repete com várias partições.
- **Group k-fold**: amostras do mesmo grupo (paciente, usuário) ficam no mesmo fold.
- **Time series split**: sempre treinar no passado e validar no futuro; **nunca embaralhar** séries temporais.
- **CV aninhada**: laço interno escolhe hiperparâmetros, externo estima desempenho (evita otimismo).

## Armadilhas

- Pré-processamento (scaler, seleção de features) deve ser ajustado *dentro* de cada fold → use `Pipeline`.
- Escolher hiperparâmetros muitas vezes na mesma CV "vaza" informação; reserve um teste final.

## Implementação

[`validacao_cruzada.cpp`](validacao_cruzada.cpp): k-fold implementado do zero para selecionar o grau de um polinômio. O erro de treino cai monotonicamente; o erro de CV forma um "U" e aponta o grau adequado.

```bash
g++ -std=c++17 -O2 validacao_cruzada.cpp -o validacao_cruzada && ./validacao_cruzada
```

## Referências

- Hastie et al., *ESL*, cap. 7.
- Cawley & Talbot, *On Over-fitting in Model Selection* (2010).
