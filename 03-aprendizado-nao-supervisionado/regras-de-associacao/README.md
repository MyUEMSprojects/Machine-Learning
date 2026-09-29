# Regras de Associação

Descobrir padrões "quem compra A também compra B" em bases de transações (*market basket analysis*), recomendações simples, co-ocorrência de sintomas, logs de uso.

## Métricas

Para uma regra $A\Rightarrow B$ em $N$ transações:

- **Suporte**: $\mathrm{sup}(A\cup B)=\frac{\#\{t\supseteq A\cup B\}}{N}$ — quão frequente.
- **Confiança**: $\mathrm{conf}(A\Rightarrow B)=\frac{\mathrm{sup}(A\cup B)}{\mathrm{sup}(A)}=P(B\mid A)$.
- **Lift**: $\frac{\mathrm{conf}(A\Rightarrow B)}{\mathrm{sup}(B)}$ — $>1$ associação positiva, $=1$ independência, $<1$ negativa. Confiança sozinha engana quando $B$ já é muito comum.

## Apriori

Propriedade **anti-monotônica**: todo subconjunto de um itemset frequente é frequente (e, contrapositivamente, superconjunto de um infrequente é infrequente). Algoritmo:

1. Encontre itens frequentes (tamanho 1).
2. Gere candidatos de tamanho $k+1$ juntando itemsets frequentes de tamanho $k$; **pode** os que têm subconjunto infrequente.
3. Conte suporte; mantenha os frequentes; repita até não haver mais.
4. Para cada itemset frequente, gere partições $A\Rightarrow B$ e filtre por confiança mínima.

O custo é a explosão combinatória de candidatos. **FP-Growth** evita gerar candidatos usando uma árvore de prefixos (FP-tree) e é o padrão na prática.

## Cuidados

Associação ≠ causalidade; muitos padrões espúrios com suporte baixo; itens muito frequentes dominam. Para ordem temporal use mineração de sequências.

## Implementação

[`apriori.cpp`](apriori.cpp) (C++): Apriori com poda e geração de regras com suporte, confiança e lift em uma cesta de compras de exemplo (fraldas → cerveja).

```bash
g++ -std=c++17 -O2 apriori.cpp -o apriori && ./apriori
```

(Em Python: `mlxtend.frequent_patterns.apriori`/`fpgrowth` + `association_rules`.)

## Referências

- Agrawal & Srikant, *Fast Algorithms for Mining Association Rules* (1994).
- Han et al., *Mining Frequent Patterns without Candidate Generation* (FP-Growth, 2000).
