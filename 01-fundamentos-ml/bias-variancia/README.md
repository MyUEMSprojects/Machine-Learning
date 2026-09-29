# Trade-off Viés–Variância

## Ideia

O erro esperado de um modelo em dados novos se decompõe (para perda quadrática) em:

$$E[(y-\hat f(x))^2]=\underbrace{(E[\hat f(x)]-f(x))^2}_{\text{viés}^2}+\underbrace{\mathrm{Var}[\hat f(x)]}_{\text{variância}}+\underbrace{\sigma^2}_{\text{ruído irredutível}}$$

- **Viés**: erro sistemático por suposições simplistas (modelo não consegue representar $f$) → *underfitting*.
- **Variância**: sensibilidade ao conjunto de treino específico (o modelo "persegue" o ruído) → *overfitting*.
- **Ruído irredutível**: limite inferior do erro; nenhum modelo passa dele.

Aumentar a complexidade reduz viés e aumenta variância. O ótimo está no meio. Mais dados reduzem variância; regularização troca um pouco de viés por menos variância.

> Nota: em deep learning moderno observa-se *double descent* (modelos superparametrizados voltam a melhorar), que complementa essa visão clássica.

## Diagnóstico prático (curvas de aprendizado)

| Erro treino | Erro validação | Diagnóstico | Remédio |
|---|---|---|---|
| alto | alto (≈ treino) | viés alto | modelo mais complexo, mais features |
| baixo | bem maior | variância alta | mais dados, regularização, modelo mais simples |

## Implementação

[`bias_variancia.cpp`](bias_variancia.cpp): sorteia 300 conjuntos de treino de $y=\sin(2\pi x)+\varepsilon$, ajusta polinômios de graus 0, 1, 3, 5, 9 e estima empiricamente viés² e variância. Grau 0/1 → viés alto; grau 9 → variância alta.

```bash
g++ -std=c++17 -O2 bias_variancia.cpp -o bias_variancia && ./bias_variancia
```

## Referências

- Hastie et al., *ESL*, seção 7.3.
- Belkin et al., *Reconciling modern machine learning practice and the bias-variance trade-off* (2019).
