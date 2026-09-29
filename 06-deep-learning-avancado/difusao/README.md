# Modelos de Difusão

Estado da arte em geração de imagens, áudio e vídeo (Stable Diffusion, DALL·E 2/3, Sora…). Ideia: aprender a **remover ruído** gradualmente.

## Processo direto (fixo): destruir a estrutura

Adiciona ruído gaussiano em $T$ passos com agenda $\beta_t$:

$$q(x_t|x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_tI\big)$$

Com $\alpha_t=1-\beta_t$ e $\bar\alpha_t=\prod_{s\le t}\alpha_s$, existe forma fechada para saltar direto a qualquer $t$:

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad\varepsilon\sim\mathcal N(0,I)$$

Para $t=T$ grande, $x_T\approx\mathcal N(0,I)$.

## Processo reverso (aprendido): criar estrutura

Rede $\varepsilon_\theta(x_t,t)$ **prevê o ruído** adicionado. Objetivo simples (DDPM):

$$L=\mathbb E_{x_0,t,\varepsilon}\Big[\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2\Big]$$

Amostragem: parta de $x_T\sim\mathcal N(0,I)$ e para $t=T..1$:

$$x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\Big)+\sigma_tz,\quad z\sim\mathcal N(0,I)$$

Equivale a *score matching*: $\varepsilon_\theta\propto-\nabla_{x}\log p_t(x)$ (o "score"), e a amostragem é uma discretização de uma SDE/ODE reversa.

## Ingredientes de sistemas reais

- **Arquitetura**: U-Net (ou Transformer — DiT) com embedding do passo $t$.
- **Difusão latente** (Stable Diffusion): difunde no espaço latente de um [VAE](../autoencoders-vae/), muito mais barato.
- **Condicionamento**: texto via atenção cruzada (encoder tipo CLIP/T5); **classifier-free guidance** interpola predições condicional/incondicional para aumentar a fidelidade ao prompt.
- **Amostradores rápidos**: DDIM (determinístico), DPM-Solver, destilação/consistency models → de 1000 para 1–50 passos.

## Comparação

| | GAN | VAE | Difusão |
|---|---|---|---|
| Qualidade | alta | média (borrada) | **muito alta** |
| Cobertura de modos | pode colapsar | boa | boa |
| Treino | instável | estável | estável |
| Geração | 1 passo | 1 passo | muitos passos (lento) |

## Implementação

[`ddpm_2d.py`](ddpm_2d.py) (Python/PyTorch): DDPM completo do zero em "duas luas": agenda de ruído, forma fechada de $q(x_t|x_0)$, MLP com embedding sinusoidal de $t$, treino por predição de ruído e amostragem reversa. Compara a distância das amostras à variedade real.

```bash
python ddpm_2d.py
```

## Referências

- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020).
- Song et al., *Score-Based Generative Modeling through SDEs* (2021).
- Weng, *What are Diffusion Models?* (lilianweng.github.io).
