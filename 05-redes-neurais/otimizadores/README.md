# Otimizadores para Redes Neurais

Como usar o gradiente $g_t=\nabla L(\theta_t)$ (estimado em mini-lotes) para atualizar os parâmetros. A superfície de perda é não convexa, ruidosa, com vales estreitos e regiões planas.

## Família

| Otimizador | Atualização (resumo) | Ideia |
|---|---|---|
| **SGD** | $\theta\leftarrow\theta-\eta g$ | simples; sensível a $\eta$ e escala |
| **Momentum** | $v\leftarrow\beta v+g;\ \theta\leftarrow\theta-\eta v$ | acumula velocidade, amortece oscilações em vales |
| **Nesterov** | gradiente calculado no ponto "à frente" | correção antecipada |
| **AdaGrad** | divide por $\sqrt{\sum g^2}$ | passo maior para parâmetros raros; $\eta$ decai demais |
| **RMSprop** | divide por $\sqrt{\text{média móvel de }g^2}$ | passo adaptativo por parâmetro |
| **Adam** | momentum ($m$) + RMSprop ($v$) + correção de viés | padrão; funciona bem "de fábrica" |
| **AdamW** | Adam com *weight decay* desacoplado | padrão em Transformers |

Adam: $m\leftarrow\beta_1m+(1-\beta_1)g,\; v\leftarrow\beta_2v+(1-\beta_2)g^2,\; \hat m=\frac{m}{1-\beta_1^t},\;\hat v=\frac{v}{1-\beta_2^t},\; \theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$. Defaults: $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$.

## Taxa de aprendizado é o hiperparâmetro mais importante

- **Schedulers**: decaimento em degraus, cosseno, **warmup** linear (essencial em Transformers), *one-cycle*, `ReduceLROnPlateau`.
- Regra prática: busque $\eta$ em escala logarítmica (*LR range test*).
- **Batch size**: maior = gradiente menos ruidoso, passos mais caros; ao aumentar o batch, aumente $\eta$ (regra linear).
- **Gradient clipping** evita explosões (RNNs, Transformers).

## SGD vs. Adam

Adam converge rápido e é robusto; SGD+momentum bem ajustado às vezes generaliza melhor (CNNs de visão). Otimizadores mais novos: Lion, Adafactor, Shampoo/Muon.

## Implementação

- [`otimizadores.cpp`](otimizadores.cpp) (C++): SGD, Momentum, RMSprop, Adam e AdamW do zero, numa regressão logística com features de escalas muito diferentes.
- [`otimizadores_pytorch.py`](otimizadores_pytorch.py) (Python): otimizadores e *scheduler* de cosseno do `torch.optim`.

```bash
g++ -std=c++17 -O2 otimizadores.cpp -o otimizadores && ./otimizadores
python otimizadores_pytorch.py
```

## Referências

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015); Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW).
- Ruder, *An overview of gradient descent optimization algorithms*.
