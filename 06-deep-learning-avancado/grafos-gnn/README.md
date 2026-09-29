# Redes Neurais em Grafos (GNNs)

Aprendizado em dados **relacionais**: redes sociais, moléculas, grafos de conhecimento, tráfego, recomendação. Um grafo $G=(V,E)$ tem features de nó $X\in\mathbb R^{n\times d}$ e adjacência $A$. Diferente de imagens/sequências, não há ordem nem vizinhança de tamanho fixo → a rede deve ser **invariante/equivariante a permutação de nós**.

## Troca de mensagens (message passing)

Cada camada atualiza o vetor de um nó agregando o de seus vizinhos:

$$h_v^{(l+1)}=\phi\Big(h_v^{(l)},\ \bigoplus_{u\in\mathcal N(v)}\psi\big(h_u^{(l)},h_v^{(l)}\big)\Big)$$

com $\bigoplus$ simétrica (soma, média, máximo). Com $L$ camadas, cada nó "enxerga" vizinhos a até $L$ saltos.

## GCN (Kipf & Welling, 2017)

$$H^{(l+1)}=\sigma\big(\hat A\,H^{(l)}W^{(l)}\big),\qquad \hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$$

Média ponderada normalizada de vizinhos (com *self-loop*) + transformação linear + não linearidade. Interpretação: **suavização** das features sobre o grafo (nós vizinhos ficam parecidos) — ótima quando o grafo é *homofílico* (vizinhos tendem a ter o mesmo rótulo).

## Outras arquiteturas

| Modelo | Ideia |
|---|---|
| **GraphSAGE** | amostra vizinhos e agrega (média/LSTM/pool); escala para grafos grandes, indutivo |
| **GAT** | **atenção** sobre vizinhos: pesos $\alpha_{vu}$ aprendidos |
| **GIN** | soma + MLP; poder discriminativo igual ao teste WL de isomorfismo |
| **MPNN** | framework geral (química quântica) |
| **Graph Transformers** | atenção global + codificações estruturais |

## Tarefas

Classificação de **nós** (semi-supervisionada), de **arestas** / predição de links (recomendação), de **grafos inteiros** (propriedades moleculares; requer *readout*/pooling global).

## Problemas conhecidos

*Over-smoothing* (muitas camadas tornam todos os nós iguais), *over-squashing* (informação de longo alcance espremida), heterofilia, escalabilidade (amostragem, cluster-GCN). Bibliotecas: **PyTorch Geometric**, **DGL**.

## Implementação

[`gcn_sbm.py`](gcn_sbm.py) (Python/PyTorch): GCN com matriz de adjacência densa normalizada, em um grafo sintético (*stochastic block model*, 3 comunidades) com features fracas e 5 rótulos por classe; compara com uma MLP que ignora o grafo.

```bash
python gcn_sbm.py
```

## Referências

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2017).
- Hamilton, *Graph Representation Learning* (livro gratuito); Veličković et al., *Graph Attention Networks* (2018).
- Bronstein et al., *Geometric Deep Learning* (2021).
