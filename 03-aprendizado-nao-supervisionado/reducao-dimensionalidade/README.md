# Redução de Dimensionalidade

Representar dados de alta dimensão com poucas dimensões preservando o que importa — para visualizar, comprimir, remover ruído/redundância e combater a maldição da dimensionalidade.

## PCA (Análise de Componentes Principais)

Encontra as direções ortogonais de **máxima variância**.

1. Centralize os dados: $\tilde X=X-\bar x$.
2. Covariância $\Sigma=\frac{1}{n-1}\tilde X^\top\tilde X$.
3. Autovetores $v_k$ de $\Sigma$ (componentes principais) ordenados pelos autovalores $\lambda_k$ (= variância em cada direção).
4. Projete: $z=V_k^\top(x-\bar x)$. Reconstrução: $\hat x=\bar x+V_kz$.

- **Variância explicada** da componente $k$: $\lambda_k/\sum_j\lambda_j$. Escolha $k$ para atingir, p.ex., 90–95%.
- Equivalente a minimizar o erro de reconstrução quadrático; na prática calcula-se via **SVD** de $\tilde X$ (mais estável).
- Linear; sensível à escala → padronize antes. Componentes podem ser difíceis de interpretar.

## Métodos não lineares (aprendizado de variedades)

| Método | Ideia | Uso |
|---|---|---|
| **Kernel PCA** | PCA no espaço de features do kernel | estruturas não lineares |
| **Isomap / LLE** | preservam distâncias geodésicas / vizinhança local | variedades suaves |
| **t-SNE** | preserva vizinhanças locais minimizando KL entre distribuições de probabilidade em alta e baixa dimensão | **visualização** (clusters); distâncias globais e tamanhos dos grupos *não* são confiáveis |
| **UMAP** | grafo de vizinhos + otimização topológica | visualização, mais rápido e preserva mais estrutura global que t-SNE |
| **Autoencoders** | rede que comprime e reconstrói | ver [autoencoders/VAE](../../06-deep-learning-avancado/autoencoders-vae/) |

## Implementação

- [`pca.cpp`](pca.cpp) (C++): PCA completo do zero (covariância + autovetores por Jacobi), variância explicada e erro de reconstrução. Dados com 6 features geradas por 2 fatores latentes: 2 componentes capturam quase tudo.
- [`reducao_sklearn.py`](reducao_sklearn.py) (Python): PCA, Kernel PCA, Isomap e t-SNE nos dígitos 8×8 (64D→2D), com figura.

```bash
g++ -std=c++17 -O2 pca.cpp -o pca && ./pca
python reducao_sklearn.py
```

## Referências

- Jolliffe, *Principal Component Analysis*.
- van der Maaten & Hinton, *Visualizing Data using t-SNE* (2008); Wattenberg et al., *How to Use t-SNE Effectively* (Distill).
- McInnes et al., *UMAP* (2018).
