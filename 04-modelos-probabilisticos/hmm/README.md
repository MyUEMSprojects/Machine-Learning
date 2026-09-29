# Modelos Ocultos de Markov (HMM)

Modelo para sequências em que uma **cadeia de Markov de estados ocultos** gera observações visíveis.

- Estados $z_t\in\{1..N\}$ com $P(z_{t+1}|z_t)=A$ (Markov de 1ª ordem).
- Observações com $P(x_t|z_t)=B$ (discretas ou contínuas, p.ex. gaussianas/GMM).
- Parâmetros $\lambda=(\pi,A,B)$.

Aplicações clássicas: reconhecimento de fala, bioinformática (genes, proteínas), POS tagging, segmentação de atividades, séries com regimes.

## Os três problemas

| Problema | Pergunta | Algoritmo |
|---|---|---|
| **Avaliação** | $P(x_{1:T}\mid\lambda)$? | **Forward** — $O(TN^2)$ (em vez de $N^T$) |
| **Decodificação** | melhor sequência de estados? | **Viterbi** (programação dinâmica) |
| **Aprendizado** | achar $\lambda$ | **Baum–Welch** (EM), usando forward–backward |

### Forward
$\alpha_t(j)=\Big[\sum_i\alpha_{t-1}(i)A_{ij}\Big]B_j(x_t)$, e $P(x)=\sum_j\alpha_T(j)$.

### Viterbi
$\delta_t(j)=\max_i\delta_{t-1}(i)A_{ij}\cdot B_j(x_t)$ guardando os ponteiros; recupere o caminho de trás para frente. Trabalhe em **log** (ou escalone) para evitar *underflow*.

### Baum–Welch (EM)
- E: $\gamma_t(i)=P(z_t=i|x)$ e $\xi_t(i,j)=P(z_t=i,z_{t+1}=j|x)$ via forward–backward.
- M: $\hat A_{ij}=\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(i)}$, $\hat B_j(m)=\frac{\sum_{t:x_t=m}\gamma_t(j)}{\sum_t\gamma_t(j)}$, $\hat\pi_i=\gamma_1(i)$.

Converge a máximo local; os estados aprendidos podem vir **permutados** em relação aos verdadeiros.

## Implementação

- [`hmm.cpp`](hmm.cpp) (C++): forward/backward com escalonamento, Viterbi em log-espaço e Baum–Welch. Simula um HMM "clima → atividades", decodifica e reaprende os parâmetros só das observações.
- [`hmm_python.py`](hmm_python.py) (Python): Viterbi em NumPy e, opcionalmente, `hmmlearn`.

```bash
g++ -std=c++17 -O2 hmm.cpp -o hmm && ./hmm
python hmm_python.py
```

## Referências

- Rabiner, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition* (1989).
- Bishop, *PRML*, cap. 13.
