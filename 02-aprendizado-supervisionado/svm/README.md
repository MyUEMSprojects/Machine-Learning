# Máquinas de Vetores de Suporte (SVM)

Classificador que busca o hiperplano de **margem máxima** — o que separa as classes com a maior folga possível.

## SVM de margem suave (soft margin)

$$\min_{w,b}\;\frac{1}{2}\lVert w\rVert^2+C\sum_i\max\big(0,\,1-y_i(w^\top x_i+b)\big),\qquad y_i\in\{-1,+1\}$$

- $\lVert w\rVert^2$: maximizar a margem $2/\lVert w\rVert$.
- **Perda hinge** $\max(0,1-y f(x))$: só penaliza pontos dentro da margem ou errados.
- $C$: compromisso entre margem larga (regularização forte, $C$ pequeno) e poucos erros no treino.
- **Vetores de suporte**: os pontos sobre/dentro da margem; só eles definem o modelo.

## Truque do kernel

A forma dual depende dos dados só por produtos internos $x_i^\top x_j$; troque por $k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$ e obtenha uma fronteira não linear sem calcular $\phi$:

- **Linear**: $x^\top z$
- **Polinomial**: $(\gamma x^\top z+r)^d$
- **RBF (gaussiano)**: $\exp(-\gamma\lVert x-z\rVert^2)$ — espaço de features infinito; $\gamma$ grande → fronteiras locais (overfit).

Predição: $f(x)=\sum_i\alpha_i y_i\,k(x_i,x)+b$, com $\alpha_i\neq0$ apenas nos vetores de suporte.

## Prática

Padronize as features. Ajuste $C$ e $\gamma$ por [validação cruzada](../../01-fundamentos-ml/validacao-cruzada/). Treino kernelizado é $O(n^2)$–$O(n^3)$: bom até dezenas de milhares de amostras. Para dados enormes/texto esparso use SVM linear (SGD/LIBLINEAR).

## Implementação

- [`svm.cpp`](svm.cpp) (C++): **Pegasos** (SGD na perda hinge), versão linear e **kernelizada com RBF**, resolvendo círculos concêntricos que o linear não separa. (Implementação didática via SGD, não o solver SMO exato.)
- [`svm_sklearn.py`](svm_sklearn.py) (Python): `SVC` com vários kernels e `GridSearchCV` (usa libsvm).

```bash
g++ -std=c++17 -O2 svm.cpp -o svm && ./svm
python svm_sklearn.py
```

## Referências

- Cortes & Vapnik, *Support-Vector Networks* (1995).
- Shalev-Shwartz et al., *Pegasos: Primal Estimated sub-GrAdient SOlver for SVM* (2007).
