# Álgebra Linear

A linguagem de ML: dados são vetores/matrizes, modelos são transformações lineares (e não lineares por cima delas).

## Conceitos-chave

- **Vetor**: ponto/direção em $\mathbb{R}^n$. Produto interno $\langle a,b\rangle=\sum a_i b_i$ mede alinhamento; norma $\|a\|=\sqrt{\langle a,a\rangle}$.
- **Matriz**: transformação linear $x\mapsto Ax$. Multiplicação $C=AB$ compõe transformações.
- **Sistema linear** $Ax=b$: resolvido por eliminação de Gauss / decomposição LU. Base da regressão linear (equações normais).
- **Autovalores/autovetores**: $Av=\lambda v$. Direções que a matriz apenas estica. Base do PCA (autovetores da covariância).
- **Matriz simétrica**: autovalores reais, autovetores ortogonais (teorema espectral).
- **SVD**: $A=U\Sigma V^\top$, generaliza autovalores para qualquer matriz (compressão, PCA, recomendação).
- **Posto (rank), traço, determinante**: traço = soma dos autovalores; determinante = produto.

## Por que importa em ML

| Conceito | Onde aparece |
|---|---|
| Produto matriz-vetor | camadas de redes neurais ($Wx+b$) |
| Solução de $Ax=b$ | regressão linear (mínimos quadrados) |
| Autovetores | PCA, spectral clustering, PageRank |
| SVD | redução de dimensão, LSA, low-rank adaptation |

## Implementação

[`algebra_linear.cpp`](algebra_linear.cpp) (C++): transposta, matmul, `solve` (Gauss com pivoteamento parcial), *power iteration* (autovalor dominante) e Jacobi (todos os autovalores de matriz simétrica).

```bash
g++ -std=c++17 -O2 algebra_linear.cpp -o algebra_linear && ./algebra_linear
```

## Referências

- Gilbert Strang, *Linear Algebra and Its Applications*; curso MIT 18.06.
- 3Blue1Brown, *Essence of Linear Algebra*.
