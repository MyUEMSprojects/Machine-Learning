# GANs (Redes Generativas Adversariais)

Dois modelos treinados em **jogo de soma zero**:

- **Gerador** $G(z)$: transforma ruído $z\sim\mathcal N(0,I)$ em amostras falsas.
- **Discriminador** $D(x)$: estima a probabilidade de $x$ ser real.

$$\min_G\max_D\;\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z}[\log(1-D(G(z)))]$$

No ótimo de $D$, o objetivo de $G$ equivale a minimizar a divergência de **Jensen–Shannon** entre $p_{data}$ e $p_G$; o equilíbrio é $p_G=p_{data}$, $D\equiv\tfrac12$.

## Treino na prática

Alterne: (1) um passo em $D$ (real→1, falso→0); (2) um passo em $G$ com a **perda não saturante** $-\log D(G(z))$ (gradientes fortes quando $D$ vence no início).

### Dificuldades

- **Instabilidade**: o problema é uma busca de ponto de sela, não uma minimização; as perdas não indicam qualidade.
- **Colapso de modos** (*mode collapse*): $G$ produz poucos tipos de amostras.
- **Gradiente que some** se $D$ ficar bom demais.

### Remédios e variantes

| Técnica | Ideia |
|---|---|
| **WGAN / WGAN-GP** | distância de Wasserstein (D vira "crítico" Lipschitz, gradient penalty) → treino mais estável, perda com significado |
| Spectral norm, TTUR, *label smoothing*, *R1 penalty* | estabilizam D |
| **DCGAN** | arquitetura convolucional com BatchNorm |
| **cGAN / Pix2Pix / CycleGAN** | condicionadas em rótulo/imagem (tradução de imagens) |
| **StyleGAN** | controle de estilo em múltiplas escalas; rostos fotorrealistas |

Hoje modelos de [difusão](../difusao/) dominam a geração de imagens (treino estável, melhor cobertura), mas GANs seguem úteis quando se precisa de geração **rápida** (um forward).

## Implementação

[`gan.py`](gan.py) (Python/PyTorch): GAN com perda não saturante numa mistura de 8 gaussianas em círculo; imprime qualidade das amostras e nº de modos cobertos ao longo do treino. Mude a semente para ver colapso de modos.

```bash
python gan.py
```

## Referências

- Goodfellow et al., *Generative Adversarial Nets* (2014).
- Arjovsky et al., *Wasserstein GAN* (2017); Karras et al., *StyleGAN* (2019).
