# Redes Recorrentes: RNN, LSTM e GRU

Modelos para **sequências** (texto, áudio, séries temporais): compartilham parâmetros no tempo e carregam um **estado oculto** $h_t$ que resume o passado.

## RNN simples

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b),\qquad \hat y=f(h_T)\ \text{(ou uma saída por passo)}$$

Treino por **BPTT** (*backpropagation through time*): desenrola a rede em $T$ passos e aplica backprop no grafo resultante; os gradientes dos pesos são **somados** sobre os passos.

### O problema dos gradientes

O gradiente até o passo $k$ contém o produto $\prod_{j}W_h^\top\,\mathrm{diag}(\tanh'(a_j))$: com autovalores $<1$ **some** (a rede não aprende dependências longas), com $>1$ **explode** (mitigado por *gradient clipping*).

## LSTM (Hochreiter & Schmidhuber, 1997)

Adiciona um **estado de célula** $c_t$ (uma "esteira" com atualização aditiva) controlado por portões:

$$\begin{aligned}
i_t&=\sigma(W_i[x_t,h_{t-1}]+b_i)&&\text{entrada}\\
f_t&=\sigma(W_f[x_t,h_{t-1}]+b_f)&&\text{esquecimento}\\
g_t&=\tanh(W_g[x_t,h_{t-1}]+b_g)&&\text{candidato}\\
o_t&=\sigma(W_o[x_t,h_{t-1}]+b_o)&&\text{saída}\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)
\end{aligned}$$

Como $\partial c_t/\partial c_{t-1}=f_t$ (sem multiplicação por matriz e tanh'), o gradiente atravessa muitos passos quando $f\approx1$. Truque: inicializar o viés do *forget gate* em ~1.

## GRU

Versão simplificada (portões de *reset* e *update*, sem célula separada): menos parâmetros, desempenho similar na prática.

## Extensões e situação atual

Bidirecional, empilhadas (*stacked*), seq2seq com codificador-decodificador (tradução), atenção de Bahdanau (precursora dos Transformers). Limitações: processamento **sequencial** (difícil de paralelizar) e memória finita. **Transformers** ([atenção](../attention-transformers/)) substituíram RNNs em NLP; RNNs/LSTMs e SSMs (Mamba) ainda aparecem em séries temporais, dispositivos embarcados e modelos de sequência longa eficientes.

## Implementação

- [`rnn_lstm.cpp`](rnn_lstm.cpp) (C++): RNN e LSTM do zero com **BPTT escrito à mão**, Adam e clipping, na tarefa "lembrar o sinal de $x_0$" com $T=5,20,30$. Nesta tarefa simples e com ruído pequeno, ambos aprendem até $T\approx40$ e falham em $T=60$ (acaso ≈ 0,5) — a vantagem teórica da LSTM só aparece de forma consistente em tarefas mais difíceis (ruído forte, várias dependências, texto), e depende de sementes e hiperparâmetros. No script PyTorch (mesma tarefa, $T$ até 80, 2000 passos) a RNN simples chegou a resolver tudo enquanto GRU/LSTM ficaram num platô inicial — um lembrete de que "LSTM > RNN" não é lei: portões podem demorar mais para "decolar". Experimente variar `T`, o ruído, a inicialização e o número de passos.
- [`rnn_pytorch.py`](rnn_pytorch.py) (Python): `nn.RNN`, `nn.GRU`, `nn.LSTM` na mesma tarefa.

```bash
g++ -std=c++17 -O2 rnn_lstm.cpp -o rnn_lstm && ./rnn_lstm
python rnn_pytorch.py
```

## Referências

- Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997).
- Olah, *Understanding LSTM Networks* (colah.github.io).
- Pascanu et al., *On the difficulty of training recurrent neural networks* (2013).
