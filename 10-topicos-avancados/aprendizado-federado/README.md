# Aprendizado Federado

Treinar um modelo **sem centralizar os dados**: os dados ficam nos dispositivos/organizações (celulares, hospitais, bancos) e apenas **atualizações do modelo** são enviadas a um servidor. Motivações: privacidade, regulação (LGPD/GDPR, HIPAA), custo de transferência, soberania de dados.

## FedAvg (McMahan et al., 2017)

Repita por rodadas:

1. O servidor envia o modelo global $w_t$ a um subconjunto de $C$ clientes.
2. Cada cliente $k$ treina localmente por $E$ épocas (SGD) nos seus dados, obtendo $w_{t+1}^{k}$.
3. O servidor faz a **média ponderada** pelo tamanho dos dados: $\;w_{t+1}=\sum_k\frac{n_k}{n}\,w^{k}_{t+1}$.

$E=1$ com lote completo equivale ao **FedSGD** (SGD centralizado). Mais épocas locais reduzem a comunicação (o gargalo), mas aumentam o desvio entre clientes.

## O desafio: dados não-IID

Na prática, cada cliente tem uma distribuição diferente (usuários com hábitos distintos; hospitais com populações distintas). Efeitos: **client drift** (cada modelo local vai para seu ótimo local, a média pode piorar ou oscilar), convergência lenta. Remédios: **FedProx** (termo de proximidade $\frac{\mu}{2}\lVert w-w_t\rVert^2$), **SCAFFOLD** (variáveis de controle), momentum no servidor (FedAvgM/FedAdam), personalização (fine-tuning local, meta-learning — ver [meta-learning](../meta-learning/)).

## Cenários

*Cross-device* (milhões de dispositivos, poucos participam por rodada, instáveis) × *cross-silo* (poucas organizações confiáveis e estáveis). Federado **horizontal** (mesmas features, usuários diferentes) × **vertical** (mesmos usuários, features diferentes em organizações distintas).

## Privacidade: federado ≠ automaticamente privado

Atualizações de gradiente/pesos podem vazar informação (ataques de inversão de gradiente, inferência de pertencimento). Defesas complementares:

- **Privacidade diferencial (DP)**: recortar a norma de cada atualização ($\lVert\Delta\rVert\le C$) e somar ruído gaussiano; garante que a presença de um indivíduo altera pouco a saída ($\varepsilon,\delta$). *Trade-off* privacidade × utilidade.
- **Agregação segura** (criptografia/MPC): o servidor só vê a **soma** das atualizações.
- **Criptografia homomórfica**, ambientes confiáveis (TEE).

Outros problemas: comunicação (compressão, quantização, esparsificação), clientes maliciosos (envenenamento; agregação robusta: mediana, Krum), equidade entre clientes.

## Implementação

- [`fedavg.cpp`](fedavg.cpp) (C++): **FedAvg do zero** com regressão logística; compara centralizado × só local × FedAvg com 1/5/20 épocas locais em clientes **IID** e **não-IID**.
- [`fedavg_pytorch.py`](fedavg_pytorch.py) (Python/PyTorch): FedAvg em dígitos com clientes que veem só 2 classes; **recorte de norma + ruído gaussiano** (DP simples) mostrando o custo de utilidade.

```bash
g++ -std=c++17 -O2 fedavg.cpp -o fedavg && ./fedavg
python fedavg_pytorch.py
```

Frameworks de produção: **Flower**, TensorFlow Federated, NVIDIA FLARE, PySyft, Opacus (DP).

## Referências

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, 2017).
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx, 2020); Kairouz et al., *Advances and Open Problems in Federated Learning* (2021).
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*.
