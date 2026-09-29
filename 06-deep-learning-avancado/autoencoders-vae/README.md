# Autoencoders e VAEs

## Autoencoder (AE)

Rede que **comprime e reconstrói**: encoder $z=f_\phi(x)$ (gargalo de baixa dimensão) e decoder $\hat x=g_\theta(z)$, treinados para minimizar $\lVert x-\hat x\rVert^2$. O gargalo força uma representação compacta — um "PCA não linear" (com ativações lineares, recupera o subespaço do PCA).

Variantes: **denoising** (reconstrói o original a partir de entrada ruidosa), **esparso** (penaliza ativações), **convolucional**, **contrativo**. Usos: compressão, pré-treino, detecção de anomalias (erro de reconstrução alto), remoção de ruído.

Limitação: o espaço latente é irregular — amostrar um $z$ aleatório não gera dados plausíveis.

## Autoencoder Variacional (VAE)

Modelo **generativo** probabilístico com variável latente $z\sim\mathcal N(0,I)$ e $p_\theta(x|z)$. O encoder produz uma distribuição $q_\phi(z|x)=\mathcal N(\mu_\phi(x),\mathrm{diag}\,\sigma_\phi^2(x))$. Maximiza-se o **ELBO** (limite inferior da log-verossimilhança):

$$\log p(x)\ \ge\ \underbrace{\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstrução}}-\underbrace{D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{regularização do latente}}$$

- **KL** em forma fechada para gaussianas: $-\tfrac12\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$.
- **Truque da reparametrização**: $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$ torna a amostragem diferenciável.
- O termo KL organiza o espaço latente (contínuo, sem buracos) → **amostrar $z\sim\mathcal N(0,I)$ e decodificar gera dados novos**; interpolar entre latentes gera transições suaves.
- **β-VAE**: peso $\beta>1$ no KL incentiva latentes desemaranhados (*disentangled*) ao custo de reconstrução.

Problema típico: saídas borradas (a verossimilhança gaussiana/Bernoulli faz média) e *posterior collapse* (o decoder ignora $z$). Sucessores: VQ-VAE (latentes discretos, usado em geração de imagens/áudio), e VAEs como compressor de latentes em modelos de [difusão](../difusao/).

## Implementação

[`autoencoder_vae.py`](autoencoder_vae.py) (Python/PyTorch): AE vs. PCA (mesma dimensão latente), VAE com reparametrização e ELBO, geração por amostragem do prior e interpolação latente — tudo nos dígitos 8×8.

```bash
python autoencoder_vae.py
```

## Referências

- Kingma & Welling, *Auto-Encoding Variational Bayes* (2014).
- Doersch, *Tutorial on Variational Autoencoders* (2016).
- Higgins et al., *β-VAE* (2017).
