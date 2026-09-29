# Otimização

Aprender = resolver $\min_\theta L(\theta)$.

## Conceitos-chave

- **Gradiente descendente**: $\theta\leftarrow\theta-\eta\nabla L(\theta)$. A taxa de aprendizado $\eta$ é crítica: alta diverge, baixa é lenta.
- **Convexidade**: função convexa tem um único mínimo global (regressão linear/logística). Redes neurais são não convexas — mínimos locais, platôs, pontos de sela.
- **Momentum**: acumula velocidade, atravessa vales estreitos mais rápido.
- **Adam**: taxa adaptativa por parâmetro usando 1º e 2º momentos do gradiente, com correção de viés.
- **Método de Newton**: usa a Hessiana, $\theta\leftarrow\theta-H^{-1}\nabla L$. Convergência quadrática, mas custo $O(n^3)$ — impraticável para redes grandes (quasi-Newton como L-BFGS ajudam).
- **SGD / mini-batch**: gradiente estimado em subconjuntos; ruído ajuda a escapar de mínimos ruins.
- **Otimização com restrições**: multiplicadores de Lagrange (base do SVM).

## Implementação

[`otimizacao.cpp`](otimizacao.cpp): GD, momentum, Adam e Newton na função de Rosenbrock (vale banana, difícil para métodos de 1ª ordem). Newton converge em poucas iterações; GD simples fica devagar.

```bash
g++ -std=c++17 -O2 otimizacao.cpp -o otimizacao && ./otimizacao
```

## Referências

- Boyd & Vandenberghe, *Convex Optimization*.
- Ruder, *An overview of gradient descent optimization algorithms* (2016).
