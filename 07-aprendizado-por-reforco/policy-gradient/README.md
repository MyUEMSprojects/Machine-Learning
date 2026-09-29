# Métodos de Gradiente de Política

Em vez de aprender valores $Q$ e derivar a política, **otimizam-se diretamente os parâmetros $\theta$ da política** $\pi_\theta(a|s)$ por subida de gradiente no retorno esperado $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G(\tau)]$.

Vantagens: ações **contínuas** (política gaussiana), políticas **estocásticas** (exploração natural), integração simples com redes profundas. Desvantagem: alta variância do gradiente e baixa eficiência amostral.

## Teorema do gradiente de política

$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\;\hat A_t\Big]$$

Aumenta a probabilidade de ações com **vantagem** positiva (melhores que o esperado) e reduz as demais. Usa o truque $\nabla p=p\nabla\log p$: não é preciso derivar o ambiente.

## Algoritmos

| Algoritmo | Vantagem $\hat A_t$ | Ideia |
|---|---|---|
| **REINFORCE** | retorno $G_t$ (menos *baseline*) | Monte Carlo; atualiza no fim do episódio; alta variância |
| **Actor-Critic** | $r+\gamma V(s')-V(s)$ (erro TD) | um **crítico** $V_\phi$ estima valores → menos variância, mas algum viés |
| **A2C/A3C** | $n$-step / GAE | atores paralelos |
| **TRPO / PPO** | GAE($\lambda$) | limitam o tamanho do passo de política: PPO usa o **clipping** $\min(\rho_tA_t,\ \mathrm{clip}(\rho_t,1\pm\epsilon)A_t)$, $\rho_t=\pi_\theta/\pi_{\theta_{old}}$ |
| **DDPG / TD3 / SAC** | crítico $Q$ | *off-policy* para ações contínuas (SAC maximiza entropia) |

**Baseline**: subtrair $b(s)$ (ex.: $V(s)$ ou a média dos retornos) não introduz viés e reduz muito a variância. **GAE($\lambda$)** interpola entre TD (viés alto, variância baixa) e Monte Carlo.

**PPO** é o "cavalo de batalha" de RL moderno: estável, simples, base de RLHF (alinhamento de LLMs com feedback humano; ver [LLMs](../../08-nlp/llms/)).

## Implementação

- [`reinforce.cpp`](reinforce.cpp) (C++): **CartPole com física do zero** + REINFORCE com baseline e política sigmoide linear; o comprimento dos episódios sobe de ~20 para ~400 (máx. 500).
- [`ppo_cartpole.py`](ppo_cartpole.py) (Python/PyTorch + Gymnasium): PPO completo (ator-crítico, GAE, clipping, bônus de entropia); resolve o CartPole (retorno 500) em ~40 mil passos.

```bash
g++ -std=c++17 -O2 reinforce.cpp -o reinforce && ./reinforce
pip install torch gymnasium && python ppo_cartpole.py
```

## Referências

- Sutton & Barto, cap. 13; Williams, *REINFORCE* (1992).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) e *GAE* (2016).
- OpenAI *Spinning Up in Deep RL* (spinningup.openai.com).
