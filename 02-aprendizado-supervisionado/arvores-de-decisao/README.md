# Árvores de Decisão

Particionam recursivamente o espaço de features com perguntas do tipo "$x_j\le t$?", formando regras interpretáveis. Algoritmo **CART** (Breiman, 1984).

## Construção (guloso, top-down)

1. Em cada nó, teste todos os pares (feature $j$, limiar $t$).
2. Escolha o corte que mais reduz a **impureza** ponderada dos filhos.
3. Recurse até: nó puro, profundidade máxima, ou poucas amostras.
4. Folha prediz a classe majoritária (ou a média, em regressão).

### Impureza

- **Gini**: $G=1-\sum_k p_k^2$
- **Entropia**: $H=-\sum_k p_k\log p_k$ (ganho de informação = redução de entropia; ver [teoria da informação](../../00-fundamentos-matematicos/teoria-da-informacao/))
- Regressão: variância / MSE dentro do nó.

Ordenar por feature e varrer os cortes mantendo contagens incrementais torna a busca $O(d\,n\log n)$ por nó.

## Prós e contras

- ✅ Interpretável, sem escalonamento, lida com features mistas e interações não lineares.
- ❌ **Overfit fácil** (uma árvore completa decora o treino); instável (pequena mudança nos dados muda a árvore); fronteiras alinhadas aos eixos.
- Controle: `max_depth`, `min_samples_leaf`, **poda** por custo-complexidade.
- A alta variância de uma árvore é exatamente o que [ensembles](../ensembles/) resolvem.

## Implementação

- [`arvore_decisao.cpp`](arvore_decisao.cpp) (C++): CART com Gini do zero e impressão da árvore. Numa região retangular com rótulos ruidosos, profundidade 1 é insuficiente, 2 recupera a regra verdadeira e profundidade ilimitada faz overfit.

> Curiosidade: o **XOR** é um caso clássico em que a busca gulosa falha — nenhum corte único reduz a impureza, então a árvore não "enxerga" a interação.
- [`arvore_decisao_sklearn.py`](arvore_decisao_sklearn.py) (Python): `export_text`, importância de features e poda.

```bash
g++ -std=c++17 -O2 arvore_decisao.cpp -o arvore_decisao && ./arvore_decisao
python arvore_decisao_sklearn.py
```

## Referências

- Breiman et al., *Classification and Regression Trees* (1984).
- Hastie et al., *ESL*, cap. 9.2.
