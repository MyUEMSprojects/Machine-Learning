# Otimização Bayesiana

Otimizar uma função $f(x)$ **cara de avaliar** (treinar uma rede, rodar um experimento, uma simulação) e **sem gradiente** ("caixa-preta") com **poucas** avaliações. Uso mais comum: ajuste de **hiperparâmetros** (ver [validação cruzada](../../01-fundamentos-ml/validacao-cruzada/)), mas também design de experimentos, química/materiais, engenharia.

## Por que não grade/aleatória?

Busca em grade explode exponencialmente; busca aleatória é surpreendentemente boa (Bergstra & Bengio, 2012) mas **não aprende** com as avaliações anteriores. A otimização bayesiana usa o histórico para decidir **onde avaliar a seguir**.

## Algoritmo

1. Avalie $f$ em alguns pontos iniciais.
2. **Modelo substituto** (*surrogate*): ajuste um modelo probabilístico $p(f\mid\text{dados})$ — tipicamente um [Processo Gaussiano](../../04-modelos-probabilisticos/processos-gaussianos/), que dá média $\mu(x)$ e incerteza $\sigma(x)$.
3. **Função de aquisição** $a(x)$: pontua quão promissor é avaliar $x$, equilibrando **exploração** (alta incerteza) e **aproveitamento** (média boa) — o mesmo dilema dos [bandits](../../07-aprendizado-por-reforco/bandits/). Maximize $a(x)$ (barato, pode usar gradiente/grade).
4. Avalie $f$ no ponto escolhido, adicione aos dados, repita até esgotar o orçamento.

### Funções de aquisição (minimização; $f^*$ = melhor valor até agora)

- **Expected Improvement**: $\mathrm{EI}(x)=\mathbb E[\max(f^*-f(x),0)]=(f^*-\mu)\Phi(z)+\sigma\varphi(z)$, com $z=\frac{f^*-\mu}{\sigma}$ — a mais usada.
- **UCB/LCB**: $\mu(x)-\kappa\sigma(x)$ (otimismo diante da incerteza; $\kappa$ controla exploração).
- Probability of Improvement, Thompson sampling, Entropy Search / Knowledge Gradient.

## Variantes práticas

| Método | Modelo substituto | Observação |
|---|---|---|
| **GP-BO** (BoTorch, Ax, scikit-optimize) | Processo Gaussiano | ótimo para poucas dimensões (~<20) e contínuas |
| **TPE** (Optuna, Hyperopt) | densidades $p(x\mid\text{bom})/p(x\mid\text{ruim})$ | lida bem com espaços condicionais/categóricos, mais escalável |
| **SMAC** | Random Forest | espaços mistos |
| **Hyperband / ASHA / BOHB** | – | **early stopping** de configurações ruins + BO |
| Bandit/evolutivos (CMA-ES) | – | alternativas sem gradiente |

Dicas: escala **logarítmica** para taxa de aprendizado/regularização; defina limites sensatos; paralelize com lotes de pontos; limitações: alta dimensionalidade e ruído forte.

## Implementação

- [`bayes_opt.cpp`](bayes_opt.cpp) (C++): **GP (Cholesky) + Expected Improvement** do zero na função de Forrester; compara com a busca aleatória em 50 repetições com o mesmo orçamento.
- [`otimizacao_optuna.py`](otimizacao_optuna.py) (Python): **Optuna** (TPE) vs. busca aleatória no ajuste de um Gradient Boosting.

```bash
g++ -std=c++17 -O2 bayes_opt.cpp -o bayes_opt && ./bayes_opt
pip install optuna scikit-learn && python otimizacao_optuna.py
```

## Referências

- Shahriari et al., *Taking the Human Out of the Loop: A Review of Bayesian Optimization* (2016).
- Snoek et al., *Practical Bayesian Optimization of Machine Learning Algorithms* (2012); Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* (2012).
- Documentação Optuna, BoTorch.
