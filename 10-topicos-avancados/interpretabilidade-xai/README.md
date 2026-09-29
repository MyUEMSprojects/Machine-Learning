# Interpretabilidade e IA Explicável (XAI)

Entender **por que** um modelo prevê o que prevê: depurar, ganhar confiança, atender regulação (LGPD/GDPR, saúde, crédito), detectar vieses e vazamentos de dados.

## Intrínseca × pós-hoc

- **Intrínseca (modelos transparentes)**: regressão linear/logística (coeficientes), árvores rasas, regras, GAMs/EBM. Há um compromisso (parcial) com acurácia.
- **Pós-hoc**: explica modelos opacos (ensembles, redes) depois de treinados. Pode ser **global** (comportamento geral) ou **local** (uma predição).

## Métodos

| Método | Escopo | Ideia | Cuidados |
|---|---|---|---|
| **Importância de árvore** (impureza) | global | redução de impureza por feature | enviesada p/ features de alta cardinalidade |
| **Importância por permutação** | global | queda do desempenho ao embaralhar a feature | subestima features correlacionadas; use dados de teste |
| **PDP / ICE** | global / por instância | efeito médio (ou individual) de variar uma feature marginalizando as demais | assume independência entre features |
| **LIME** | local | ajusta um modelo linear ponderado numa vizinhança perturbada da instância | instável, depende do kernel/amostragem |
| **SHAP** | local + global | **valores de Shapley** da teoria dos jogos: contribuição justa de cada feature | custo exponencial (exato só p/ poucas features; TreeSHAP é polinomial) |
| **Gradientes / Grad-CAM / Integrated Gradients** | local (redes) | sensibilidade da saída aos pixels/tokens | mapas de saliência podem enganar (sanity checks) |
| **Contrafactuais** | local | menor mudança na entrada que altera a decisão ("recurso") | plausibilidade/viabilidade |
| **Probing / interpretabilidade mecanicista** | redes | o que as ativações codificam; circuitos | campo de pesquisa ativo (LLMs) |

### Valores de Shapley

Contribuição da feature $j$ para a predição de $x$, média sobre todas as ordens de "revelar" features:

$$\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{j\})-v(S)\Big]$$

com $v(S)=\mathbb E[f(x_S,X_{\bar S})]$. Propriedades: **eficiência** ($\sum_j\phi_j=f(x)-\mathbb E[f(X)]$), simetria, dummy (feature irrelevante ⇒ 0), aditividade. Interações são divididas entre as features envolvidas.

## Armadilhas

Explicação ≠ causalidade (mostra o que o **modelo** usa, não o que causa o resultado); features correlacionadas dividem/mascaram importância; explicações podem ser manipuladas; "interpretável" depende do público.

## Implementação

- [`xai.cpp`](xai.cpp) (C++): importância por permutação, PDP, **Shapley exato por enumeração** (verificando a propriedade de eficiência) e **LIME** simplificado (regressão linear local ponderada) sobre uma função caixa-preta com interação.
- [`xai_sklearn_shap.py`](xai_sklearn_shap.py) (Python): `permutation_importance`, `partial_dependence` e **SHAP** (`TreeExplainer`) num Gradient Boosting.

```bash
g++ -std=c++17 -O2 xai.cpp -o xai && ./xai
pip install scikit-learn shap && python xai_sklearn_shap.py
```

## Referências

- Molnar, *Interpretable Machine Learning* (christophm.github.io/interpretable-ml-book, gratuito).
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, 2017); Ribeiro et al., *"Why Should I Trust You?"* (LIME, 2016).
- Olah et al., *Zoom In: An Introduction to Circuits* (Distill).
