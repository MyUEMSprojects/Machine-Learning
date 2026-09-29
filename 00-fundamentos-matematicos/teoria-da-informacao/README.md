# Teoria da Informação

Mede incerteza e "distância" entre distribuições — origem das funções de perda de classificação.

## Conceitos-chave

- **Entropia**: $H(p)=-\sum_i p_i\log p_i$. Incerteza média (bits se $\log_2$). Máxima na distribuição uniforme.
- **Entropia cruzada**: $H(p,q)=-\sum_i p_i\log q_i$. Custo de codificar dados de $p$ usando um código otimizado para $q$. É a *loss* padrão de classificação.
- **Divergência KL**: $D_{KL}(p\|q)=H(p,q)-H(p)\ge0$. Não é simétrica, não é distância. Minimizar entropia cruzada = minimizar KL (pois $H(p)$ é constante) = MLE.
- **Informação mútua**: $I(X;Y)=D_{KL}(p(x,y)\|p(x)p(y))$. Quanto saber $Y$ reduz a incerteza sobre $X$; 0 se independentes.
- **Ganho de informação**: redução de entropia — critério de divisão em árvores de decisão.

## Onde aparece

Loss de classificação, árvores de decisão, VAEs (termo KL), t-SNE (minimiza KL), seleção de features, RL (regularização por entropia, PPO/TRPO).

## Implementação

[`teoria_informacao.cpp`](teoria_informacao.cpp): entropia, entropia cruzada, KL (mostrando assimetria) e informação mútua (independente vs. dependente).

```bash
g++ -std=c++17 -O2 teoria_informacao.cpp -o teoria_informacao && ./teoria_informacao
```

## Referências

- Cover & Thomas, *Elements of Information Theory*.
- Colah, *Visual Information Theory*.
