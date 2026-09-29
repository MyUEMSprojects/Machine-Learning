# Meta-Learning ("aprender a aprender")

Em vez de treinar um modelo para **uma** tarefa, treina-se para que ele **aprenda novas tarefas rapidamente**, com pouquíssimos exemplos (*few-shot learning*). Cenário típico: reconhecer uma classe nova com 1–5 imagens; adaptar um robô a um novo ambiente; personalizar um modelo por usuário.

## Formulação

Há uma **distribuição de tarefas** $p(\mathcal T)$. Cada tarefa tem um conjunto de **suporte** (poucos exemplos, para adaptar) e um de **consulta** (para avaliar a adaptação). Meta-treino: repetidamente sorteia-se uma tarefa, adapta-se no suporte e otimiza-se o desempenho na consulta. Meta-teste: tarefas **novas** (nunca vistas).

## Famílias

| Família | Ideia | Exemplos |
|---|---|---|
| **Baseada em otimização** | aprende uma **inicialização** (ou otimizador) da qual poucos passos de gradiente bastam | **MAML**, Reptile, Meta-SGD |
| **Baseada em métrica** | aprende um espaço de embeddings onde classificar = comparar com exemplos do suporte | Redes Siamesas, **Prototypical Networks**, Matching Networks |
| **Baseada em modelo / memória** | uma rede (RNN/Transformer) lê o suporte e prediz diretamente | Meta-Networks, **aprendizado em contexto de LLMs** (*in-context learning*) |

### MAML (Finn et al., 2017)

Para cada tarefa $i$: adaptação interna $\theta_i'=\theta-\alpha\nabla_\theta\mathcal L_i^{sup}(\theta)$. Objetivo externo:

$$\min_\theta\sum_i\mathcal L_i^{qry}\big(\theta_i'\big)=\sum_i\mathcal L^{qry}_i\big(\theta-\alpha\nabla_\theta\mathcal L^{sup}_i(\theta)\big)$$

O gradiente atravessa o passo interno (envolve a Hessiana → **segunda ordem**); a aproximação de primeira ordem (FOMAML) ignora esse termo. Encontra um $\theta$ "sensível": pequenos passos em qualquer tarefa reduzem muito a perda.

### Reptile (Nichol et al., 2018)

Só primeira ordem e muito simples: treina $k$ passos numa tarefa obtendo $\tilde\theta$ e move $\theta\leftarrow\theta+\varepsilon(\tilde\theta-\theta)$. Equivale a maximizar o produto interno entre gradientes de minibatches da mesma tarefa.

## Relação com outros temas

- [Transfer learning](../../06-deep-learning-avancado/transfer-learning/): aproveita **uma** tarefa de origem; meta-learning otimiza explicitamente a *capacidade de adaptação* entre **muitas** tarefas.
- **Few-shot em LLMs**: exemplos no prompt fazem o papel do conjunto de suporte, sem atualizar pesos ([LLMs](../../08-nlp/llms/)).
- **Aprendizado federado personalizado** ([federado](../aprendizado-federado/)) usa MAML/Reptile para adaptar o modelo global a cada cliente.
- [Otimização bayesiana](../otimizacao-bayesiana/): outra forma de "aprender a escolher bem" com poucas avaliações.

Desafios: custo de memória/computação do MAML de segunda ordem, definir uma boa distribuição de tarefas, ganho menor quando as tarefas de teste diferem muito das de meta-treino.

## Implementação

[`maml_reptile_sine.py`](maml_reptile_sine.py) (Python/PyTorch): **MAML** (segunda ordem com `create_graph=True`) e **Reptile** do zero em regressão de senoides com $K=5$ exemplos por tarefa; compara com inicialização aleatória e pré-treino conjunto em tarefas novas após 0/1/5/10 passos de gradiente. Resultado típico: MAML ≪ Reptile < pré-treino joint < aleatória em MSE após poucos passos.

```bash
python maml_reptile_sine.py
```

## Referências

- Finn, Abbeel & Levine, *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (2017).
- Nichol, Achiam & Schulman, *On First-Order Meta-Learning Algorithms* (Reptile, 2018).
- Snell et al., *Prototypical Networks for Few-shot Learning* (2017); Hospedales et al., *Meta-Learning in Neural Networks: A Survey* (2021).
