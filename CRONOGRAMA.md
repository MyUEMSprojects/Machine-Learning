# Rotina e Estrutura de Estudos: ML, Data Science & Data Engineering

Estudar três frentes técnicas distintas no mesmo dia frequentemente resulta em sobrecarga cognitiva e perda de rendimento por troca frequente de contexto (*context switching*). O objetivo desta estrutura é maximizar a profundidade teórica e a prática aplicada, organizando os estudos em blocos temáticos bem definidos.

---

## Estratégias de Organização

### Abordagem 1: Separação por Dias (Recomendada)
Dedica cada dia da semana a uma linha de raciocínio específica, eliminando o atrito da troca de foco mental.

| Dia | Foco Principal | Repositório | Atividades Práticas |
| :--- | :--- | :--- | :--- |
| **Segunda** | Matemática + Algoritmo Base | `Machine-Learning` | Teoria, cálculo/álgebra do algoritmo e implementação do zero (*scratch*). |
| **Terça** | Estatística & EDA | `Data-Science` | Distribuições, limpeza de dados, formulação de hipóteses e tratamento de features. |
| **Quarta** | Modelagem & Métricas | `Machine-Learning` | Scikit-learn/PyTorch, treinamento com datasets reais e tuning de hiperparâmetros. |
| **Quinta** | Engenharia de Dados & Armazenamento | `data-engineering` | SQL avançado, modelagem relacional (PostgreSQL), Docker e ingestão de dados. |
| **Sexta** | Pipelines & Integração | `data-engineering` / MLOps | Orquestração (scripts/Airflow), exportação de artefatos de modelos ou API de inferência. |
| **Fim de semana** | Revisão & Consolidação (Opcional) | Todos / Nenhum | Leitura técnica, resolução de exercícios pontuais ou descanso. |

---

### Abordagem 2: Divisão Diária em 2 Blocos
Para rotinas que demandam contato contínuo com ambas as áreas, limitando o dia a apenas dois contextos:

* **Bloco 1: Eixo Analítico / ML (~60% do tempo total)**
  * **Momento:** Início do dia ou da sessão de estudo (mente descansada).
  * **Escopo:** Alternância entre `Machine-Learning` e `Data-Science`.
  * *Exemplo:* 1h30 focada no conceito de Regressão Logística, matriz de confusão e cálculo de entropia/perda.
* **Bloco 2: Eixo Engenharia (~40% do tempo total)**
  * **Momento:** Segunda metade da sessão (trabalho procedimental e estrutural).
  * **Escopo:** Foco em `data-engineering`.
  * *Exemplo:* 1h construindo queries com Window Functions, configurando PostgreSQL via Docker Compose ou estruturando um pipeline ETL.

---

## Ciclo de Estudo Ativo por Sessão

Para evitar o estudo estritamente passivo, cada bloco deve seguir a divisão tripartida:

1. **Fundamentação Teórica (Até 30% do bloco):**
   * Leitura de documentações, artigos técnicos ou capítulos conceituais.
   * Compreensão das derivações matemáticas ou comandos de arquitetura.
2. **Exercício Guiado / Scratch (30% do bloco):**
   * Implementação de funções de perda, gradientes ou matrizes sem uso de bibliotecas de alto nível.
   * Resolução de exercícios analíticos ou problemas de modelagem SQL.
3. **Commit no Repositório (40% do bloco):**
   * **`Machine-Learning`:** Notebooks ou scripts testando algoritmos contra modelos *baseline*.
   * **`Data-Science`:** Análise exploratória com gráficos analíticos e tratamento de dados reais.
   * **`data-engineering`:** Schemas SQL otimizados, tabelas normalizadas, arquivos de configuração ou scripts de ingestão.

---

## Plano de Ação: Primeiras 4 Semanas

### Semana 1: Vetores, Álgebra Linear e SQL Base
* **`Machine-Learning`:** Operações vetoriais e matriciais com NumPy puro + implementação da Regressão Linear do zero.
* **`Data-Science`:** Análise exploratória univariada/bivariada, medidas de dispersão e correlação de Pearson.
* **`data-engineering`:** Ambiente PostgreSQL local via Docker; queries intermediárias e avançadas (`JOINs`, `GROUP BY`, Window Functions).

### Semana 2: Otimização, Pré-processamento e Modelagem Relacional
* **`Machine-Learning`:** Derivadas parciais, regra da cadeia e algoritmo do Gradiente Descendente + validação com Scikit-Learn.
* **`Data-Science`:** Tratamento sistemático de valores nulos, One-Hot Encoding e padronização com `StandardScaler` / `MinMaxScaler`.
* **`data-engineering`:** Modelagem de dados relacional (chaves primárias/estrangeiras, integridade referencial, normalização até 3FN) e scripts de carga em Python.

### Semana 3: Probabilidade, Classificação e Pipelines de Ingestão
* **`Machine-Learning`:** Probabilidade condicional, Regressão Logística, Teorema de Bayes e classificador Naive Bayes.
* **`Data-Science`:** Avaliação de modelos de classificação: Precision, Recall, F1-Score, Curva ROC e cálculo de AUC.
* **`data-engineering`:** Automação de extração/carga via Python com `psycopg2` ou `SQLAlchemy` e logging estruturado.

### Semana 4: Modelos em Árvore, Feature Selection e Integração
* **`Machine-Learning`:** Árvores de Decisão (entropia e ganho de informação) e ensemble com Random Forest.
* **`Data-Science`:** Detecção de outliers (IQR, Z-score), relevância de atributos (*feature importance*) e seleção de variáveis.
* **`data-engineering`:** Introdução à orquestração de tarefas ou construção de uma API simples (FastAPI) para servir predições do modelo treinado.