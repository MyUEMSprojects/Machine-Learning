# Cálculo

Treinar um modelo é minimizar uma função de perda; o cálculo diz **em que direção** mudar os parâmetros.

## Conceitos-chave

- **Derivada**: taxa de variação instantânea, $f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$.
- **Gradiente** $\nabla f=(\partial_1 f,\dots,\partial_n f)$: aponta para a direção de maior crescimento. Descemos em $-\nabla f$.
- **Regra da cadeia**: $\frac{d}{dx}f(g(x))=f'(g(x))\,g'(x)$. É o coração do backpropagation.
- **Jacobiana / Hessiana**: derivadas de 1ª ordem de funções vetoriais / de 2ª ordem (curvatura).
- **Diferenciação automática**: calcula derivadas *exatas* decompondo o programa em operações elementares.
  - *Forward mode* (números duais): eficiente com poucas entradas.
  - *Reverse mode* (backprop): eficiente com muitos parâmetros e uma saída (a perda).
- **Gradient checking**: comparar gradiente analítico com diferença central para depurar implementações.

## Implementação

[`calculo.cpp`](calculo.cpp): derivada numérica (diferença central), gradiente numérico, autodiff *forward* com `struct Dual`, e comparação das três abordagens em $f(x,y)=x^2y+\sin(xy)$.

```bash
g++ -std=c++17 -O2 calculo.cpp -o calculo && ./calculo
```

## Referências

- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (2018).
- Stewart, *Cálculo*.
