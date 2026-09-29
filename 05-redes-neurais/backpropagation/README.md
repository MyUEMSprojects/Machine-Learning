# Backpropagation

Algoritmo que calcula $\nabla_\theta L$ para **todos** os parâmetros de uma rede em um custo próximo ao de um único forward — a regra da cadeia aplicada de forma eficiente sobre o **grafo computacional**.

## Ideia

Uma rede é uma composição $L=\ell(f_n(\dots f_2(f_1(x))))$. Com $a_k=f_k(a_{k-1};\theta_k)$:

$$\frac{\partial L}{\partial\theta_k}=\underbrace{\frac{\partial L}{\partial a_k}}_{\delta_k}\frac{\partial a_k}{\partial\theta_k},\qquad \delta_{k-1}=\frac{\partial L}{\partial a_{k-1}}=\delta_k\frac{\partial a_k}{\partial a_{k-1}}$$

1. **Forward**: calcule e **guarde** as ativações intermediárias.
2. **Backward**: do fim para o início, propague $\delta$ e acumule gradientes dos parâmetros.

Para uma camada densa $z=Wa+b$, $a'=\phi(z)$: $\;\delta_z=\delta_{a'}\odot\phi'(z)$, $\;\nabla_W=\delta_z\,a^\top$, $\;\nabla_b=\delta_z$, $\;\delta_a=W^\top\delta_z$.

Com sigmoide/softmax + entropia cruzada, $\delta_{z}=\hat y-y$ (simplificação elegante).

## Diferenciação automática (modo reverso)

Frameworks (PyTorch, JAX, TensorFlow) implementam o backprop genericamente: cada operação registra como propagar gradientes; o backward percorre o grafo em ordem topológica invertida, **acumulando** ($+=$) gradientes quando um valor é usado mais de uma vez. Custo de memória: guardar ativações (por isso *gradient checkpointing*).

## Problemas clássicos

- **Gradientes que somem** (sigmoide/tanh saturadas, redes profundas/RNNs): mitigados por ReLU, inicialização He/Xavier, normalização (BatchNorm/LayerNorm), conexões residuais, LSTM.
- **Gradientes que explodem**: *gradient clipping*.

## Depuração: gradient checking

Compare o gradiente analítico com $\frac{L(\theta+h)-L(\theta-h)}{2h}$. Diferença relativa $\lesssim10^{-7}$ indica implementação correta.

## Implementação

- [`backprop_autodiff.cpp`](backprop_autodiff.cpp) (C++): mini-framework de **autodiff reverso** (estilo micrograd): `Value`, grafo, ordenação topológica, `tanh/exp/log`; treina um MLP e faz *gradient check*.
- [`backprop_pytorch.py`](backprop_pytorch.py) (Python): backward manual em NumPy comparado ao `autograd` do PyTorch.

```bash
g++ -std=c++17 -O2 backprop_autodiff.cpp -o backprop_autodiff && ./backprop_autodiff
python backprop_pytorch.py
```

## Referências

- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* (1986).
- Karpathy, *micrograd* e vídeo "The spelled-out intro to neural networks and backpropagation".
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*.
