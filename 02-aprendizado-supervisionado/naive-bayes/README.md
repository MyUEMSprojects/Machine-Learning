# Naive Bayes

Classificador **generativo** baseado no teorema de Bayes com a suposição "ingênua" de que as features são condicionalmente independentes dada a classe.

$$P(c\mid x)\propto P(c)\prod_{j=1}^{d}P(x_j\mid c)\quad\Rightarrow\quad \hat c=\arg\max_c\;\log P(c)+\sum_j\log P(x_j\mid c)$$

Use log-probabilidades (produtos de muitos números pequenos dão *underflow*).

## Variantes (mudam $P(x_j\mid c)$)

| Variante | Feature | Modelo |
|---|---|---|
| **Gaussian NB** | contínua | $\mathcal N(\mu_{cj},\sigma^2_{cj})$ estimados por MLE |
| **Multinomial NB** | contagem (texto) | frequência de palavras por classe |
| **Bernoulli NB** | binária (presença) | Bernoulli por palavra |

**Suavização de Laplace** ($\alpha=1$): soma $\alpha$ às contagens para que uma palavra nunca vista numa classe não zere toda a probabilidade.

## Por que funciona apesar da suposição falsa?

As probabilidades ficam mal calibradas, mas o *ranking* das classes frequentemente continua correto. Treino é uma única passada $O(nd)$; ótimo baseline para texto (spam, sentimento) e com poucos dados.

## Limitações

Não captura interação entre features; probabilidades extremas (mal calibradas). Correlação forte entre features "conta duas vezes" a evidência.

## Implementação

- [`naive_bayes.cpp`](naive_bayes.cpp) (C++): Gaussian NB em blobs 2D e Multinomial NB para um mini filtro de spam.
- [`naive_bayes_sklearn.py`](naive_bayes_sklearn.py) (Python): TF-IDF + `MultinomialNB` no 20 Newsgroups.

```bash
g++ -std=c++17 -O2 naive_bayes.cpp -o naive_bayes && ./naive_bayes
python naive_bayes_sklearn.py
```

## Referências

- Manning et al., *Introduction to Information Retrieval*, cap. 13.
- Murphy, *Probabilistic ML: An Introduction*, cap. 9.
